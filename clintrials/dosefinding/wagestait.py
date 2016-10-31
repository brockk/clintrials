__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'

""" An implementation of Wages & Tait's adaptive Bayesian design for dose-finding in clinical trials.

See:
Wages, N.A. and Tait, C. (2015). Seamless Phase I/II Adaptive Design For Oncology Trials of Molecularly Targeted Agents,
        Journal of Biopharmiceutical Statistics (preprint)

"""


import numpy as np
from scipy.stats import norm, beta
from scipy.integrate import quad, trapz
from random import sample

from clintrials.common import empiric, inverse_empiric
from clintrials.dosefinding.efficacytoxicity import EfficacyToxicityDoseFindingTrial
#from clintrials.util import correlated_binary_outcomes_from_uniforms
from clintrials.dosefinding.crm import CRM


_min_theta, _max_theta = -10, 10


def _wt_lik(cases, skeleton, theta, F=empiric, a0=0):
    """ Calculate the compound likelihood for many dose & efficacy pairs in Wages & Tait dose-finding method.

    Params:
    cases, list of 3-tuples, (dose, toxicity, efficacy), where dose is 1-based index of dose level received,
                                toxicity is 1 for toxic event, 0 for tolerance event
                                and efficacy is 1 for efficacious outcome, 0 for alternative.
    skeleton, list of prior efficacy probabilities
    theta, the third parameter to F, the slope
    F, a link function like logistic or empiric that takes params x, intercept, slope and returns a probability
    a0, the second parameter to F, the intercept

    Returns a probability.

    """

    l = 1
    for dose, tox, eff in cases:
        p = F(skeleton[dose-1], a0=a0, beta=theta)
        l = l * p**eff * (1-p)**(1-eff)
    return l


def _wt_get_theta_hat(cases, skeletons, theta_prior, F=empiric, use_quick_integration=False, estimate_var=False):
    """ Get posterior estimates of theta hat (and optionally, variance) in Wages & Tait dose-finding method.

    See Wages, N.A. & Tait, C. - Seamless Phase I/II Adaptive Design For Oncology Trials
                of Molecularly Targeted Agents, Journal of Biopharmaceutical Statistics

    :param cases: list of 3-tuples, (dose, toxicity, efficacy), where dose is 1-based index of dose level received,
                                toxicity is 1 for toxic event, 0 for tolerance event
                                and efficacy is 1 for efficacious outcome, 0 for alternative.
    :type cases: list
    :param skeletons: 2-d matrix of skeletons, i.e. list of lists, one prior efficacy scenario per row
    :type skeletons: list
    :param theta_prior: prior distibution for theta parameter, assumes interface like scipy.stats.rv_continuous
    :type theta_prior: scipy.stats.rv_continuous
    :param F: a link function like logistic or empiric that takes params x, intercept, slope and returns a probability
    :type F: func
    :param use_quick_integration: True to use a faster but slightly less accurate estimate of the integrals;
                                  False to use a slower but more accurate method.
    :type use_quick_integration: bool
    :param estimate_var: True to estimate the posterior variance of theta
    :type estimate_var: bool
    :return: 3-tuple, vectors for (posterior means, posterior variances, posterior model probabilities)
    :rtype: tuple

    """

    theta_hats = []
    for skeleton in skeletons:
        if use_quick_integration:
            n = 100 * max(np.log(len(cases) + 1) / 2, 1)  # My own rule of thumb for num points needed
            z, dz = np.linspace(_min_theta, _max_theta, num=n, retstep=1)
            denom_y = _wt_lik(cases, skeleton, z, F) * theta_prior.pdf(z)
            num_y = z * denom_y
            num = trapz(num_y, z, dz)
            denom = trapz(denom_y, z, dz)
            theta_hat = num / denom
            if estimate_var:
                num2_y = z**2 * denom_y
                num2 = trapz(num2_y, z, dz)
                exp_x2 = num2 / denom
                var = exp_x2 - theta_hat**2
                theta_hats.append((theta_hat, var, denom))
            else:
                theta_hats.append((num / denom, None, denom))
        else:
            num = quad(lambda t: t * _wt_lik(cases, skeleton, t, F) * theta_prior.pdf(t), -np.inf, np.inf)
            denom = quad(lambda t: _wt_lik(cases, skeleton, t, F) * theta_prior.pdf(t), -np.inf, np.inf)
            theta_hat = num[0] / denom[0]
            if estimate_var:
                num2 = quad(lambda t: t**2 * _wt_lik(cases, skeleton, t, F) * theta_prior.pdf(t), -np.inf, np.inf)
                exp_x2 = num2[0] / denom[0]
                var = exp_x2 - theta_hat**2
                theta_hats.append((theta_hat, var, denom[0]))
            else:
                theta_hats.append((theta_hat, None, denom[0]))
    return theta_hats


def _get_post_eff_bayes(cases, skeleton, dose_labels, theta_prior, F=empiric, use_quick_integration=False):
    """ Calculate the posterior probability of efficacy at doses using the Bayesian integral

    :param cases: list of 3-tuples, (dose, toxicity, efficacy), where dose is 1-based index of dose level received,
                                toxicity is 1 for toxic event, 0 for tolerance event
                                and efficacy is 1 for efficacious outcome, 0 for alternative.
    :type cases: list
    :param skeleton: list of prior efficacy probabilities
    :type skeleton: list
    :param dose_labels: dose-labels (often derived by backwards substitution) for which to estimate associated toxicity
    :type dose_labels: list
    :param theta_prior: prior distibution for theta parameter, assumes interface like scipy.stats.rv_continuous
    :type theta_prior: scipy.stats.rv_continuous
    :param F: a link function like logistic or empiric that takes params x, intercept, slope and returns a probability
    :type F: func
    :param use_quick_integration: True to use a faster but slightly less accurate estimate of the integrals;
                                  False to use a slower but more accurate method.
    :type use_quick_integration: bool
    :return: estimates of Pr(Eff) at each dose
    :rtype: list

    """

    post_eff = []
    intercept = 0
    if use_quick_integration:
        # This method uses simple trapezium quadrature. It is quite accurate and pretty fast.
        n = 100 * max(np.log(len(cases) + 1) / 2, 1)  # My own rule of thumb for num points needed
        z, dz = np.linspace(_min_theta, _max_theta, num=n, retstep=1)
        denom_y = _wt_lik(cases, skeleton, z, F) * theta_prior.pdf(z)
        denom = trapz(denom_y, z, dz)
        for x in dose_labels:
            num_y = F(x, a0=intercept, beta=z) * denom_y
            num = trapz(num_y, z, dz)
            post_eff.append(num / denom)
    else:
        # This method uses numpy's adaptive quadrature method. Superior accuracy but quite slow
        denom = quad(lambda t: theta_prior.pdf(t) * _wt_lik(cases, skeleton, t, F), -np.inf, np.inf)
        for x in dose_labels:
            num = quad(lambda t: F(x, a0=intercept, beta=t) * theta_prior.pdf(t) * _wt_lik(cases, skeleton, t, F),
                       -np.inf, np.inf)
            post_eff.append(num[0] / denom[0])

    return np.array(post_eff)


class WagesTait(EfficacyToxicityDoseFindingTrial):
    """ This is an object-oriented implementation of Wages & Tait adaptive phase I/II design for oncology MTAs.

    See Wages, N.A. & Tait, C. - Seamless Phase I/II Adaptive Design For Oncology Trials
                    of Molecularly Targeted Agents, to appear in Journal of Biopharmaceutical Statistics

    e.g. general usage
    >>> trial_size = 30
    >>> first_dose = 3
    >>> tox_target = 0.35
    >>> tox_limit = 0.40
    >>> eff_limit = 0.45
    >>> stage_1_size = 0  # Else initial dose choices will be random; not good for testing!
    >>> skeletons = [
    ...                 [.6, .5, .3, .2],
    ...                 [.5, .6, .5, .3],
    ...                 [.3, .5, .6, .5],
    ...                 [.2, .3, .5, .6],
    ...                 [.3, .5, .6, .6],
    ...                 [.5, .6, .6, .6],
    ...                 [.6, .6, .6, .6]
    ...             ]
    >>> prior_tox_probs = [0.025, 0.05, 0.1, 0.25]
    >>> trial = WagesTait(skeletons, prior_tox_probs, tox_target, tox_limit, eff_limit, first_dose, trial_size,
    ...                     stage_1_size)
    >>> trial.update([(3, 0, 1), (3, 1, 1), (3, 0, 0)])
    3
    >>> trial.has_more()
    True
    >>> trial.size(), trial.max_size()
    (3, 30)
    >>> trial.admissable_set()
    [1, 2, 3]
    >>> trial.update([(3, 1, 1), (3, 1, 1), (3, 0, 0)])
    2
    >>> trial.next_dose()
    2
    >>> trial.admissable_set()
    [1, 2]
    >>> trial.most_likely_model_index
    2

    """

    def __init__(self, skeletons, prior_tox_probs, tox_target, tox_limit, eff_limit,
                 first_dose, max_size, randomisation_stage_size,
                 F_func=empiric, inverse_F=inverse_empiric,
                 theta_prior=norm(0, np.sqrt(1.34)), beta_prior=norm(0, np.sqrt(1.34)),
                 excess_toxicity_alpha=0.025, deficient_efficacy_alpha=0.025,
                 model_prior_weights=None, use_quick_integration=False, estimate_var=False, plugin_mean=False):
        """

        Params:
        :param skeletons: 2-d matrix of skeletons, i.e. list of lists, one prior efficacy scenario per row
        :type skeletons: list
        :param prior_tox_probs: list of prior probabilities of toxicity, from least toxic to most.
        :type prior_tox_probs: list
        :param tox_target: target toxicity rate
        :type tox_target: float
        :param tox_limit: the maximum acceptable probability of toxicity
        :type tox_limit: float
        :param eff_limit: the minimium acceptable probability of efficacy
        :type eff_limit: float
        :param first_dose: starting dose level, 1-based. I.e. intcpt=3 means the middle dose of 5.
        :type first_dose: int
        :param max_size: maximum number of patients to use
        :type max_size: int
        :param randomisation_stage_size: number of patients to randomise in first stage of trial
        :type randomisation_stage_size: int
        :param F_func: the link function for CRM-like methods, e.g. empiric
        :type F_func: func
        :param inverse_F: the inverse link function for CRM-like methods method, e.g. inverse_empiric
        :type inverse_F: func
        :param theta_prior: prior distibution for theta parameter, the single parameter in the efficacy models
        :type theta_prior: scipy.stats.rv_continuous
        :param beta_prior: prior distibution for beta parameter, the single parameter in the toxicity CRM model
        :type beta_prior: scipy.stats.rv_continuous
        :param excess_toxicity_alpha: significance to use when testing that lowest dose exceeds toxicity limit
        :type excess_toxicity_alpha: float
        :param deficient_efficacy_alpha: significance to use when testing that optimal dose has efficacy less than
        efficacy limit
        :type deficient_efficacy_alpha: float
        :param model_prior_weights: list of prior probabilities that each model is correct. None to use uniform weights
        :type model_prior_weights: list
        :param use_quick_integration: numerical integration is slow. Set this to False to use the most accurate (& slow)
                                method; False to use a quick but approximate method.
                                In simulations, fast and approximate often suffices.
                                In trial scenarios, use slow and accurate!
        :type use_quick_integration: bool
        :param estimate_var: True to estimate the posterior variance of beta and theta
        :type estimate_var: bool
        :param plugin_mean: True to estimate event curves by plugging parameter estimate into function;
                            False to estimate using full Bayesian integral (default).
        :type plugin_mean: bool

        """

        EfficacyToxicityDoseFindingTrial.__init__(self, first_dose, len(prior_tox_probs), max_size)

        self.skeletons = skeletons
        self.K, self.I = np.array(skeletons).shape
        if self.I != len(prior_tox_probs):
            ValueError('prior_tox_probs should have %s items.' % self.I)
        if tox_target > tox_limit:
            ValueError('tox_target is greater than tox_limit. That does not sound clever.')
        self.prior_tox_probs = np.array(prior_tox_probs)
        self.tox_limit = tox_limit
        self.eff_limit = eff_limit
        self.randomisation_stage_size = randomisation_stage_size
        self.F_func = F_func
        self.inverse_F = inverse_F
        self.theta_prior = theta_prior
        self.beta_prior = beta_prior
        self.excess_toxicity_alpha = excess_toxicity_alpha
        self.deficient_efficacy_alpha = deficient_efficacy_alpha
        if model_prior_weights is not None:
            if self.K != len(model_prior_weights):
                ValueError('model_prior_weights should have %s items.' % self.K)
            if sum(model_prior_weights) == 0:
                ValueError('model_prior_weights cannot sum to zero.')
            self.model_prior_weights = model_prior_weights / sum(model_prior_weights)
        else:
            self.model_prior_weights = np.ones(self.K) / self.K
        self.use_quick_integration = use_quick_integration
        self.estimate_var = estimate_var
        self.plugin_mean = plugin_mean

        # Reset
        self.most_likely_model_index = \
            np.random.choice(np.array(range(self.K))[self.model_prior_weights == max(self.model_prior_weights)], 1)[0]
        self.w = np.zeros(self.K)
        if first_dose is None:
            self._next_dose = self._randomise_next_dose(prior_tox_probs, self.skeletons[self.most_likely_model_index])
            self.randomise_at_start = True
        else:
            # _next_dose is set in this case by parent class
            self.randomise_at_start = False
        self.crm = CRM(prior=prior_tox_probs, target=tox_target, first_dose=first_dose, max_size=max_size,
                       F_func=empiric, inverse_F=inverse_empiric, beta_prior=beta_prior,
                       use_quick_integration=use_quick_integration, estimate_var=estimate_var, plugin_mean=plugin_mean)
        self.post_tox_probs = np.zeros(self.I)
        self.post_eff_probs = np.zeros(self.I)
        self.theta_hats = np.zeros(self.K)

    def dose_toxicity_lower_bound(self, dose_level, alpha=0.025):
        """ Get lower bound of toxicity probability at a dose-level using the Clopper-Pearson aka Beta aka exact method.

        Params:
        dose-level, 1-based index of dose level
        alpha, significance level, i.e. {100*alpha}% of probabilities will be lower than the returned value

        Returns: a probability

        """
        if 0 < dose_level <= len(self.post_tox_probs):
            n = self.treated_at_dose(dose_level)
            x = self.toxicities_at_dose(dose_level)
            if n > 0:
                return beta(x, n-x+1).ppf(alpha)
        # Default
        return 0

    def dose_efficacy_upper_bound(self, dose_level, alpha=0.025):
        """ Get upper bound of efficacy probability at a dose-level using the Clopper-Pearson aka Beta aka exact method.

        Params:
        dose-level, 1-based index of dose level
        alpha, significance level, i.e. {100*alpha}% of probabilities will be greater than the returned value

        Returns: a probability

        """
        if 0 < dose_level <= len(self.post_eff_probs):
            n = self.treated_at_dose(dose_level)
            x = self.efficacies_at_dose(dose_level)
            if n > 0:
                return beta(x+1, n-x).ppf(1-alpha)
        # Default
        return 1

    def model_theta_hat(self):
        """ Return theta hat for the model with the highest posterior likelihood, i.e. the current model. """
        return self.theta_hats[self.most_likely_model_index]

    def _EfficacyToxicityDoseFindingTrial__calculate_next_dose(self):
        cases = list(zip(self._doses, self._toxicities, self._efficacies))
        toxicity_cases = []
        for (dose, tox, eff) in cases:
            toxicity_cases.append((dose, tox))
        self.crm.reset()
        self.crm.update(toxicity_cases)

        # Update parameters for efficacy estimates
        integrals = _wt_get_theta_hat(cases, self.skeletons, self.theta_prior,
                                      use_quick_integration=self.use_quick_integration, estimate_var=False)
        theta_hats, theta_vars, model_probs = zip(*integrals)

        self.theta_hats = theta_hats
        w = self.model_prior_weights * model_probs
        self.w = w / sum(w)
        most_likely_model_index = np.argmax(w)
        self.most_likely_model_index = most_likely_model_index
        self.post_tox_probs = np.array(self.crm.prob_tox())
        if self.plugin_mean:
            self.post_eff_probs = empiric(self.skeletons[most_likely_model_index],
                                          beta=theta_hats[most_likely_model_index])
        else:
            a0 = 0
            theta0 = self.theta_prior.mean()
            dose_labels = [self.inverse_F(p, a0=a0, beta=theta0) for p in self.skeletons[most_likely_model_index]]
            self.post_eff_probs = _get_post_eff_bayes(cases, self.skeletons[most_likely_model_index], dose_labels,
                                                      self.theta_prior, use_quick_integration=self.use_quick_integration
                                                      )

        # Update combined model
        if self.size() < self.randomisation_stage_size:
            self._next_dose = self._randomise_next_dose(self.post_tox_probs, self.post_eff_probs)
        else:
            self._next_dose = self._maximise_next_dose(self.post_tox_probs, self.post_eff_probs)

        # Stop if lower bound of probability at lowest dose exceeds tox_limit:
        if self.dose_toxicity_lower_bound(1, self.excess_toxicity_alpha) > self.tox_limit:
            self._status = -3
            self._next_dose = -1
            self._admissable_set = []
        # Stop if upper bound of efficacy at optimum dose is less than eff_limit
        if self.size() >= self.randomisation_stage_size:
            if self.dose_efficacy_upper_bound(self._next_dose, self.deficient_efficacy_alpha) < self.eff_limit:
                self._status = -4
                self._next_dose = -1
                self._admissable_set = []

        return self._next_dose

    def _EfficacyToxicityDoseFindingTrial__reset(self):
        """ Opportunity to run implementation-specific reset operations. """
        self.most_likely_model_index = \
            sample(np.array(range(self.K))[self.model_prior_weights == max(self.model_prior_weights)], 1)[0]
        self.w = np.zeros(self.K)
        self.post_tox_probs = np.zeros(self.I)
        self.post_eff_probs = np.zeros(self.I)
        self.theta_hats = np.zeros(self.K)
        self.crm.reset()
        if self.randomise_at_start:
            self._next_dose = self._randomise_next_dose(self.prior_tox_probs,
                                                        self.skeletons[self.most_likely_model_index])

    def has_more(self):
        return EfficacyToxicityDoseFindingTrial.has_more(self)

    def optimal_decision(self, prob_tox, prob_eff):
        """ Get the optimal dose choice for a given dose-toxicity curve.

        .. note:: Ken Cheung (2014) presented the idea that the optimal behaviour of a dose-finding
        design can be calculated for a given set of patients with their own specific tolerances by
        invoking the dose decicion on the complete (and unknowable) toxicity and efficacy curves.

        :param prob_tox: collection of toxicity probabilities
        :type prob_tox: list
        :param prob_tox: collection of efficacy probabilities
        :type prob_tox: list
        :return: the optimal (1-based) dose decision
        :rtype: int

        """

        admiss = prob_tox<= self.tox_limit
        if sum(admiss) > 0:
            wt_obd = np.nanargmax(np.where(admiss, prob_eff, np.nan)) + 1
        else:
            wt_obd = -1
        return wt_obd

    # Private interface
    def _randomise_next_dose(self, tox_probs, eff_probs):
        acceptable_doses = tox_probs <= self.tox_limit
        if sum(acceptable_doses) > 0:
            # There are acceptable doses
            prob_randomise = []
            for acc, eff in zip(acceptable_doses, eff_probs):
                if acc:
                    prob_randomise.append(eff)
                else:
                    prob_randomise.append(0)
            prob_randomise = np.array(prob_randomise) / sum(prob_randomise)
            self._status = 1
            self._admissable_set = [i for (acc, i) in zip(acceptable_doses, range(1, self.num_doses+1)) if acc]
            return np.random.choice(range(1, self.I+1), p=prob_randomise)
        else:
            # No acceptable doses, stop trial
            self._status = -1
            self._admissable_set = []
            return -1

    def _maximise_next_dose(self, tox_probs, eff_probs):
        acceptable_doses = tox_probs <= self.tox_limit
        if sum(acceptable_doses) > 0:
            # There are acceptable doses
            self._status = 1
            self._admissable_set = [i for (acc, i) in zip(acceptable_doses, range(1, self.num_doses+1)) if acc]
            return np.argmax(np.array(eff_probs)[acceptable_doses]) + 1
        else:
            # No acceptable doses, stop trial
            self._status = -1
            self._admissable_set = []
            return -1
