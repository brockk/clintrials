__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'

from collections import OrderedDict
import logging
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad, trapz
from scipy.optimize import minimize

from clintrials.dosefinding import DoseFindingTrial
from clintrials.common import empiric, logistic, inverse_empiric, inverse_logistic
from clintrials.util import atomic_to_json, iterable_to_json


_min_beta, _max_beta = -10, 10


def _toxicity_likelihood(link_func, a0, beta, dose, tox, log=False):
    """ Calculate likelihood of toxicity outcome in CRM given link_func plus parameters and a dose & tox pair.

    Note: this method can be Memoized to save iterations at the expense of memory.
    Also, this method allows vectorisation (in beta, importantly) if your link_func allows vectorisation.

    Params:
    link_func, link function like logistic or empiric, taking params (dose_label, intercept, slope), returning a probability
    a0, the second parameter to link_func, the intercept
    beta, the third parameter to link_func, the slope
    dose, the first parameter to link_func, the dose label (often derived by backwards substitution!)
    tox, use 1 if toxicity observed, 0 if not.
    log, True to return log likelihood

    Returns a probability (or the log of a probability)

    """

    p = link_func(dose, a0, beta)
    if log:
        return tox * np.log(p) + (1 - tox) * np.log(1 - p)
    else:
        return p ** tox * (1 - p) ** (1 - tox)


def _compound_toxicity_likelihood(link_func, a0, beta, doses, toxs, log=False):
    """ Calculate the compound likelihood of many toxicity outcomes in CRM given many dose & tox pairs.

    Params:
    link_func, link function like logistic or empiric, taking params (dose_label, intercept, slope), returning a probability
    a0, the second parameter to link_func, the intercept
    beta, the third parameter to link_func, the slope
    doses, a list of the first parameters to link_func, the dose labels (often derived by backwards substitution!)
    toxs, a list of toxicity markers. Use 1 if toxicity observed, 0 if not. Should be same length as doses.
    log, True to return log likelihood

    Returns a probability (or the log of a probability)

    """

    if len(doses) != len(toxs):
        raise ValueError('doses and toxs should be same length.')

    if log:
        l = 0
        for dose, tox in zip(doses, toxs):
            l += _toxicity_likelihood(link_func, a0, beta, dose, tox, log=True)
        return l
    else:
        l = 1
        for dose, tox in zip(doses, toxs):
            l *= _toxicity_likelihood(link_func, a0, beta, dose, tox, log=False)
        return l


def _get_beta_hat_bayes(F, intercept, codified_doses_given, toxs, beta_pdf, use_quick_integration=False,
                        estimate_var=False):
    """ Get posterior estimate of beta parameter (and optionally its variance) in Bayesian CRM.

    :param F: link function like logistic or empiric, taking params (dose_label, intercept, slope), returns probability
    :type F: func
    :param intercept: the second parameter to F, the intercept
    :type intercept: float
    :param codified_doses_given: doses given to patients on codified scale, i.e. each a valid first parameter to F
    :type codified_doses_given: list
    :param toxs: observed toxicity events. Use 1 if toxicity observed, else 0. Congruent to codified_doses_given.
    :type toxs: list
    :param beta_pdf: PDF of beta's prior distribution
    :type beta_pdf: func
    :param use_quick_integration: True to use a faster but slightly less accurate estimate of the integrals;
                                  False to use a slower but more accurate method.
    :type use_quick_integration: bool
    :param estimate_var: True to estimate beta variance. False by default for speed.
    :type estimate_var: bool
    :return: a 2-tuple, (posterior mean, posterior variance)
    :rtype: tuple

    """

    if use_quick_integration:
        # This method uses simple trapezium quadrature. It is quite accurate and pretty fast.
        n = 100 * max(np.log(len(codified_doses_given) + 1) / 2, 1)  # My own rule of thumb
        z, dz = np.linspace(_min_beta, _max_beta, num=n, retstep=1)
        num_y = z * _compound_toxicity_likelihood(F, intercept, z, codified_doses_given, toxs) * beta_pdf(z)
        denom_y = _compound_toxicity_likelihood(F, intercept, z, codified_doses_given, toxs) * beta_pdf(z)
        num = trapz(num_y, z, dz)
        denom = trapz(denom_y, z, dz)
        beta_hat = num / denom
        if estimate_var:
            num2_y = z ** 2 * _compound_toxicity_likelihood(F, intercept, z, codified_doses_given, toxs) * beta_pdf(z)
            num2 = trapz(num2_y, z, dz)
            exp_x2 = num2 / denom
            var = exp_x2 - beta_hat ** 2
        else:
            var = None
    else:
        # This method uses numpy's adaptive quadrature method. Superior accuracy but quite slow
        num = quad(lambda t: t * _compound_toxicity_likelihood(F, intercept, t, codified_doses_given, toxs) * \
                             beta_pdf(t), -np.inf, np.inf)
        denom = quad(lambda t: _compound_toxicity_likelihood(F, intercept, t, codified_doses_given, toxs) * \
                               beta_pdf(t), -np.inf, np.inf)
        beta_hat = num[0] / denom[0]
        if estimate_var:
            num2 = quad(lambda t: t ** 2 * _compound_toxicity_likelihood(F, intercept, t, codified_doses_given, toxs) *
                                  beta_pdf(t), -np.inf, np.inf)
            exp_x2 = num2[0] / denom[0]
            var = exp_x2 - beta_hat ** 2
        else:
            var = None

    return beta_hat, var


def _get_beta_hat_mle(F, intercept, codified_doses_given, toxs, estimate_var=False):
    """ Get maximum likelihood estimate of beta parameter (and optionally its variance) in MLE CRM.

    TODO: how is estimating variance possible?

    :param F: link function like logistic or empiric, taking params (dose_label, intercept, slope), returns probability
    :type F: func
    :param intercept: the second parameter to F, the intercept
    :type intercept: float
    :param codified_doses_given: doses given to patients on codified scale, i.e. each a valid first parameter to F
    :type codified_doses_given: list
    :param toxs: observed toxicity events. Use 1 if toxicity observed, else 0. Congruent to codified_doses_given.
    :type toxs: list
    :param beta_pdf: PDF of beta's prior distribution
    :type beta_pdf: func
    :param use_quick_integration: True to use a faster but slightly less accurate estimate of the integrals;
                                  False to use a slower but more accurate method.
    :type use_quick_integration: bool
    :param estimate_var: True to estimate beta variance. False by default for speed.
    :type estimate_var: bool
    :return: a 2-tuple, (posterior mean, posterior variance)
    :rtype: tuple

    """
    if sum(np.array(toxs) == 1) == 0 or sum(np.array(toxs) == 0) == 0:
        msg = 'Need heterogeneity in toxic events (i.e. toxic and non-toxic outcomes must be observed) for MLE to ' \
              'exist. See Cheung p.23.'
        logging.warn()
        return np.nan, None

    f = lambda beta: -1 * _compound_toxicity_likelihood(F, intercept, beta, codified_doses_given, toxs, log=True)
    res = minimize(f, x0=0, method='nelder-mead')
    if estimate_var:
        logging.warn('Variance estimation in MLE mode is not implemented because I do not think it makes sense. \
        In dfcrm, Ken Cheung uses a method that I do not understand. It yields comparatively huge variances.')
    return res.x[0], None


def _estimate_prob_tox_from_param(F, intercept, beta_hat, dose_labels):
    """ Estimate the probability of toxicity at doses by plugging in an estimate for beta.

    :param F: link function like logistic or empiric, taking params (dose_label, intercept, slope), returns probability
    :type F: func
    :param intercept: the second parameter to F, the intercept
    :type intercept: float
    :param beta_hat: the third parameter to F, estimate for beta, be it posterior- or maximum-likelihood-
    :type beta_hat: float
    :param dose_labels: dose-labels (often derived by backwards substitution) for which to estimate associated toxicity
    :type dose_labels: list
    :return: estimates of Pr(Tox) at each dose
    :rtype: list

    """

    post_tox = [F(x, a0=intercept, beta=beta_hat) for x in dose_labels]
    return post_tox


def _get_post_tox_bayes(F, intercept, dose_labels, codified_doses_given, toxs, beta_pdf, use_quick_integration=False):
    """ Calculate the posterior probability of toxicity at doses using the Bayesian integral

    :param F: link function like logistic or empiric, taking params (dose_label, intercept, slope), returns probability
    :type F: func
    :param intercept: the second parameter to F, the intercept
    :type intercept: float
    :param dose_labels: dose-labels (often derived by backwards substitution) for which to estimate associated toxicity
    :type dose_labels: list
    :param codified_doses_given: doses given to patients on codified scale, i.e. each a valid first parameter to F
    :type codified_doses_given: list
    :param toxs: observed toxicity events. Use 1 if toxicity observed, else 0. Congruent to codified_doses_given.
    :type toxs: list
    :param beta_pdf: PDF of beta's prior distribution
    :type beta_pdf: func
    :param use_quick_integration: True to use a faster but slightly less accurate estimate of the integrals;
                                  False to use a slower but more accurate method.
    :type use_quick_integration: bool
    :return: estimates of Pr(Tox) at each dose
    :rtype: list

    """

    post_tox = []
    if use_quick_integration:
        # This method uses simple trapezium quadrature. It is quite accurate and pretty fast.
        n = 100 * max(np.log(len(codified_doses_given) + 1) / 2, 1)  # My own rule of thumb
        z, dz = np.linspace(_min_beta, _max_beta, num=n, retstep=1)
        denom_y = _compound_toxicity_likelihood(F, intercept, z, codified_doses_given, toxs) * beta_pdf(z)
        denom = trapz(denom_y, z, dz)
        # num_scale = _compound_toxicity_likelihood(F, intercept, z, codified_doses_given, toxs) * beta_pdf(z)
        for x in dose_labels:
            num_y = F(x, a0=intercept, beta=z) * denom_y
            num = trapz(num_y, z, dz)
            post_tox.append(num / denom)
    else:
        # This method uses numpy's adaptive quadrature method. Superior accuracy but quite slow
        denom = quad(lambda t: beta_pdf(t) * \
                               _compound_toxicity_likelihood(F, intercept, t, codified_doses_given, toxs),
                     -np.inf, np.inf)
        for x in dose_labels:
            num = quad(lambda t: F(x, a0=intercept, beta=t) * beta_pdf(t) * \
                                 _compound_toxicity_likelihood(F, intercept, t, codified_doses_given, toxs),
                       -np.inf, np.inf)
            post_tox.append(num[0] / denom[0])

    return post_tox


def crm(prior, target, toxicities, dose_levels, intercept=3, F_func=logistic, inverse_F=inverse_logistic,
        beta_dist=norm(loc=0, scale=np.sqrt(1.34)), method="bayes", use_quick_integration=False,
        estimate_var=False, plugin_mean=True):
    """
    Run CRM calculation on observed dosages and toxicities.

    This method is similar to Ken Cheung's method in the R-package dfcrm. Take a look at that
    or his book for more information.

    Params:
    :param prior: list of prior probabilities of toxicity at each dose, from least toxic to most.
    :type prior: list
    :param target: target toxicity rate
    :type target: float
    :param toxicities: observed toxicity events. Use 1 if toxicity observed, else 0. Congruent to codified_doses_given.
    :type toxicities: list
    :param dose_levels: list of given 1-based dose levels, NOT the actual doses. See Cheung. Same length as toxicities.
    :type dose_levels: list
    :param intercept: the second parameter to F, the intercept. Only pertinent under logistic method.
    :type intercept: float
    :param F_func: the link function and inverse for CRM method, e.g. logistic
    :type F_func: func
    :param inverse_F: the inverse link function for CRM method, e.g. inverse_logistic
    :type inverse_F: func
    :param beta_dist: prior distibution for beta parameter, assumes interface like scipy.stats.rv_continuous
    :type beta_dist: scipy.stats.rv_continuous
    :param method: one of "bayes" or "mle"
    :type method: str
    :param use_quick_integration: True to use a faster but slightly less accurate estimate of the integrals;
                                  False to use a slower but more accurate method.
    :type use_quick_integration: bool
    :param estimate_var: True to estimate the posterior variance of beta
    :type estimate_var: bool
    :param plugin_mean: True to estimate toxicity curve by plugging beta estimate (posterior mean or mle) into function;
                        False to estimate using full Bayesian integral (only applies when method="bayes")
    :type plugin_mean: bool
    :return: 4-tuple, (recommended dose index (1-based), beta hat estimate, beta variance estimate, Pr(Tox) estimates)
    :rtype: tuple

    I omitted Ken's parameters:
    n=length(level), dosename=NULL, include=1:n, pid=1:n, conf.level=0.9, model.detail=TRUE, patient.detail=TRUE

    """

    if len(dose_levels) != len(toxicities):
        raise ValueError('toxicities and dose_levels should be same length.')

    beta0 = beta_dist.mean()
    codified_doses = [inverse_F(prior[dl - 1], a0=intercept, beta=beta0) for dl in dose_levels]
    dose_labels = [inverse_F(p, a0=intercept, beta=beta0) for p in prior]
    if method == 'bayes':
        beta_hat, var = _get_beta_hat_bayes(F_func, intercept, codified_doses, toxicities, beta_dist.pdf,
                                            use_quick_integration, estimate_var)
        if plugin_mean:
            post_tox = _estimate_prob_tox_from_param(F_func, intercept, beta_hat, dose_labels)
        else:
            # Bayesian integral
            post_tox = _get_post_tox_bayes(F_func, intercept, dose_labels, codified_doses, toxicities, beta_dist.pdf,
                                           use_quick_integration)
    elif method == 'mle':
        beta_hat, var = _get_beta_hat_mle(F_func, intercept, codified_doses, toxicities, estimate_var)
        post_tox = _estimate_prob_tox_from_param(F_func, intercept, beta_hat, dose_labels)
    else:
        msg = "Only 'bayes' and 'mle' methods are implemented."
        raise ValueError(msg)

    abs_distance_from_target = [abs(x - target) for x in post_tox]
    dose = np.argmin(abs_distance_from_target) + 1
    return dose, beta_hat, var, post_tox


class CRM(DoseFindingTrial):
    """ This is an object-oriented attempt at the CRM method.

    e.g. general usage

    >>> prior_tox_probs = [0.025, 0.05, 0.1, 0.25]
    >>> tox_target = 0.35
    >>> first_dose = 3
    >>> trial_size = 30
    >>> trial = CRM(prior_tox_probs, tox_target, first_dose, trial_size)
    >>> trial.next_dose()
    3
    >>> trial.update([(3,0), (3,0), (3,0)])
    4
    >>> trial.size(), trial.max_size()
    (3, 30)
    >>> trial.update([(4,0), (4,1), (4,1)])
    4
    >>> trial.update([(4,0), (4,1), (4,1)])
    3
    >>> trial.has_more()
    True

    """

    def __init__(self, prior, target, first_dose, max_size, F_func=empiric, inverse_F=inverse_empiric,
                 beta_prior=norm(0, np.sqrt(1.34)), method="bayes", use_quick_integration=False, estimate_var=True,
                 avoid_skipping_untried_escalation=False, avoid_skipping_untried_deescalation=False,
                 lowest_dose_too_toxic_hurdle=0.0, lowest_dose_too_toxic_certainty=0.0,
                 coherency_threshold=0.0, principle_escalation_func=None, termination_func=None, plugin_mean=True,
                 intercept=3):
        """

        Params:
        :param prior: list of prior probabilities of toxicity, from least toxic to most.
        :type prior: list
        :param target: target toxicity rate
        :type target: float
        :param first_dose: starting dose level, 1-based. I.e. first_dose=3 means the middle dose of 5.
        :type first_dose: int
        :param F_func: the link function for CRM method, e.g. logistic
        :type F_func: func
        :param inverse_F: the inverse link function for CRM method, e.g. inverse_logistic
        :type inverse_F: func
        :param beta_prior: prior distibution for beta parameter
        :type beta_prior: scipy.stats.rv_continuous
        :param max_size: maximum number of patients to use in trial
        :type max_size: int
        :param method: one of "bayes" or "mle"
        :type method: str
        :param use_quick_integration: numerical integration is slow. Set this to False to use the most accurate (& slow)
                                method; False to use a quick but approximate method.
                                In simulations, fast and approximate often suffices.
                                In trial scenarios, use slow and accurate!
        :type use_quick_integration: bool
        :param estimate_var: True to estimate the posterior variance of beta
        :type estimate_var: bool
        :param avoid_skipping_untried_escalation: True to avoid skipping untried doses in escalation
        :type avoid_skipping_untried_escalation: bool
        :param avoid_skipping_untried_deescalation: True to avoid skipping untried doses in de-escalation
        :type avoid_skipping_untried_deescalation: bool
        :param lowest_dose_too_toxic_hurdle: used with lowest_dose_too_toxic_certainty to facilitate stopping the trial
                    when the rate of estimated toxicity at the lowest dose is too high. Trial stops if:
                        Prob( Prob(Tox at d1) > lowest_dose_too_toxic_hurdle | X) > lowest_dose_too_toxic_certainty
                    Both must be positive for test to be invoked.
        :type lowest_dose_too_toxic_hurdle: float
        :param lowest_dose_too_toxic_certainty: see above
        :type lowest_dose_too_toxic_certainty: float
        :param coherency_threshold: if positive, the design is prevented from escalating when the observed toxicity rate
                                    at a dose exceeds this value. For instance, you might not want to escalate if the
                                    observed toxicity rate exceeds the target rate, because that would be 'incoherent'
                                    to the objectives of the trial.
        :type coherency_threshold: float
        :param principle_escalation_func: an optional function that takes cases (i.e., a list of
                            (1-based dose-level, boolean DLT dummies) like [(1,0), (2,0), (3,1)] )
                and returns either a) the next dose to be given, or b) None, to signify that principle escalation does
                not apply and that the general CRM method should be used.
                This function lets users specify their desired escalation that will take priority over the CRM strategy.
                For example, some users like to specify an initial escalation strategy that escalates until it
                observes the first toxicity. This function allows that behaviour in a flexible way.
                The principle_escalation_func is checked at every update so, if you use it, be mindful that it yields
                to the CRM model when you want it to by returning None.
        :type principle_escalation_func: func
        :param termination_func: an optional function that takes this trial instance as a sole parameter and returns
                True if trial should terminate, else False. The function is invoked when trial is asked whether it has
                more. This function gives trials a general facility to terminate early if certain conditions are met.
        :type termination_func: func
        :param plugin_mean: True to estimate toxicity curve by plugging beta estimate (posterior mean or mle) into func;
                        False to estimate using full Bayesian integral (only applies when method="bayes")
        :type plugin_mean: bool
        :param intercept: the second parameter to F, the intercept. Only pertinent under logistic method.
        :type intercept: float

        """

        DoseFindingTrial.__init__(self, first_dose=first_dose, num_doses=len(prior), max_size=max_size)

        self.prior = prior
        self.target = target
        self.F_func = F_func
        self.inverse_F = inverse_F
        self.beta_prior = beta_prior
        self.method = method
        self.use_quick_integration = use_quick_integration
        self.estimate_var = estimate_var
        self.avoid_skipping_untried_escalation = avoid_skipping_untried_escalation
        self.avoid_skipping_untried_deescalation = avoid_skipping_untried_deescalation
        self.lowest_dose_too_toxic_hurdle = lowest_dose_too_toxic_hurdle
        self.lowest_dose_too_toxic_certainty = lowest_dose_too_toxic_certainty
        self.coherency_threshold = coherency_threshold
        self.principle_escalation_func = principle_escalation_func
        self.termination_func = termination_func
        self.plugin_mean = plugin_mean
        self.intercept = intercept
        if lowest_dose_too_toxic_hurdle and lowest_dose_too_toxic_certainty:
            if not self.estimate_var:
                logging.warn('To monitor toxicity at lowest dose, I had to enable beta variance estimation.')
            self.estimate_var = True
        # Reset
        self.beta_hat, self.beta_var = beta_prior.mean(), beta_prior.var()
        self.post_tox = list(self.prior)

    def _DoseFindingTrial__reset(self):
        self.beta_hat, self.beta_var = self.beta_prior.mean(), self.beta_prior.var()
        self.post_tox = self.prior

    def _DoseFindingTrial__calculate_next_dose(self):

        if self.principle_escalation_func:
            cases = zip(self._doses, self._toxicities)
            proposed_dose = self.principle_escalation_func(cases)
            if proposed_dose is not None:
                return proposed_dose

        current_dose = self.next_dose()
        max_dose_given = self.maximum_dose_given()
        min_dose_given = self.minimum_dose_given()
        proposed_dose, beta_hat, beta_var, post_tox = crm(prior=self.prior, target=self.target,
                                                          toxicities=self._toxicities, dose_levels=self._doses,
                                                          intercept=self.intercept, F_func=self.F_func,
                                                          inverse_F=self.inverse_F,
                                                          beta_dist=self.beta_prior, method=self.method,
                                                          use_quick_integration=self.use_quick_integration,
                                                          estimate_var=self.estimate_var, plugin_mean=self.plugin_mean)
        self.beta_hat = beta_hat
        self.beta_var = beta_var
        self.post_tox = post_tox

        # Excess toxicity at lowest dose?
        if self.lowest_dose_too_toxic_hurdle and self.lowest_dose_too_toxic_certainty:
            labels = [self.inverse_F(p, a0=self.intercept, beta=self.beta_prior.mean()) for p in self.prior]
            beta_sample = norm(loc=beta_hat, scale=np.sqrt(beta_var)).rvs(1000000)  # N.b. normal sample a la prior
            p0_sample = self.F_func(labels[0], a0=self.intercept, beta=beta_sample)
            p0_tox = np.mean(p0_sample > self.lowest_dose_too_toxic_hurdle)

            if p0_tox > self.lowest_dose_too_toxic_certainty:
                proposed_dose = 0
                self._status = -1
                return proposed_dose

        # Coherency
        if self.coherency_threshold and proposed_dose > current_dose:
            # print 'Testing coherence'
            tox_rate_at_current = self.observed_toxicity_rates()[current_dose - 1]
            # print 'Tox at current', tox_rate_at_current
            if not np.isnan(tox_rate_at_current) and tox_rate_at_current > self.coherency_threshold:
                # Avoid escalation. Stay at current
                # print 'Throttling for coherence'
                proposed_dose = current_dose
                return proposed_dose

        # Skipping doses
        if self.avoid_skipping_untried_escalation and max_dose_given and proposed_dose - max_dose_given > 1:
            # Avoid skipping untried doses in escalation by setting proposed dose to max_dose_given + 1
            proposed_dose = max_dose_given + 1
            return proposed_dose
        elif self.avoid_skipping_untried_deescalation and min_dose_given and min_dose_given - proposed_dose > 1:
            # Avoid skipping untried doses in de-escalation by setting proposed dose to min_dose_given - 1
            proposed_dose = min_dose_given - 1
            return proposed_dose
        # Note: other methods of limiting dose escalation and de-escalation are possible.

        return proposed_dose

    def prob_tox(self):
        return list(self.post_tox)

    def prob_tox_exceeds(self, tox_cutoff, n=10 ** 6):
        if self.estimate_var:
            # Estimate probability of toxicity exceeds tox_cutoff using plug-in mean and variance for beta, and randomly
            # sampling values from normal. Why normal? Because the prior for is normal and the posterior
            # is asymptotically normal. For low n, non-normality may lead to bias.

            # TODO: research replacing this with a proper posterior integral when in bayes mode.

            labels = [self.inverse_F(p, a0=self.intercept, beta=self.beta_prior.mean()) for p in self.prior]
            beta_sample = norm(loc=self.beta_hat, scale=np.sqrt(self.beta_var)).rvs(n)
            p0_sample = [self.F_func(label, a0=self.intercept, beta=beta_sample) for label in labels]
            return np.array([np.mean(x > tox_cutoff) for x in p0_sample])
        else:
            raise Exception('CRM can only estimate posterior probabilities when estimate_var=True')

    def has_more(self):
        """ Is the trial ongoing? """
        if not DoseFindingTrial.has_more(self):
            return False
        if self.termination_func:
            return not self.termination_func(self)
        else:
            return True

    def optimal_decision(self, prob_tox):
        """ Get the optimal dose choice for a given dose-toxicity curve.

        .. note:: Ken Cheung (2014) presented the idea that the optimal behaviour of a dose-finding
        design can be calculated for a given set of patients with their own specific tolerances by
        invoking the dose decicion on the complete (and unknowable) toxicity curve.

        :param prob_tox: collection of toxicity probabilities
        :type prob_tox: list
        :return: the optimal (1-based) dose decision
        :rtype: int

        """

        return np.argmin(np.abs(prob_tox - self.target)) + 1

    def get_tox_prob_quantile(self, p):
        """ Get the quantiles of the probabilities of toxicity at each dose using normal approximation.
        :param p: probability, i.e. 0.05 means 5th quantile, i.e. 95% of values are greater
        :type p: float
        :return: the quantiles of the probabilities of toxicity at each dose
        :rtype: list
        """
        norm_crit = norm.ppf(p)
        beta_est = self.beta_hat - norm_crit * np.sqrt(self.beta_var)
        labels = [self.inverse_F(p, a0=self.intercept, beta=self.beta_prior.mean()) for p in self.prior]
        p = [self.F_func(x, a0=self.intercept, beta=beta_est) for x in labels]
        return p

    def plot_toxicity_probabilities(self, chart_title=None, use_ggplot=False):
        """ Plot prior and posterior dose-toxicity curves.

        :param chart_title: optional chart title. Default is fairly verbose
        :type chart_title: str
        :param use_ggplot: True to use ggplot, else matplotlib
        :type use_ggplot: bool
        :return: plot of toxicity curves

        """

        if not chart_title:
            chart_title = "Prior (dashed) and posterior (solid) dose-toxicity curves"
            chart_title = chart_title + "\n"

        if use_ggplot:
            from ggplot import (ggplot, ggtitle, geom_line, geom_hline, aes, ylim)
            import numpy as np
            import pandas as pd
            data = pd.DataFrame({'Dose level': self.dose_levels(),
                                 'Prior': self.prior,
                                 'Posterior': self.prob_tox(),
                                 #                      'Lower': crm.get_tox_prob_quantile(0.05),
                                 #                      'Upper': crm.get_tox_prob_quantile(0.95)
                                 })
            var_name = 'Type'
            value_name = 'Probability of toxicity'
            melted_data = pd.melt(data, id_vars='Dose level', var_name=var_name, value_name=value_name)
            # melted_data['LineType'] =  np.where(melted_data.Type=='Posterior', '--', np.where(melted_data.Type=='Prior', '-', '..'))
            # melted_data['LineType'] =  np.where(melted_data.Type=='Posterior', '--', np.where(melted_data.Type=='Prior', '-', '..'))
            # melted_data['Col'] =  np.where(melted_data.Type=='Posterior', 'green', np.where(melted_data.Type=='Prior', 'blue', 'yellow'))
            # np.where(melted_data.Type=='Posterior', '--', '-')

            p = ggplot(melted_data, aes(x='Dose level', y=value_name, linetype=var_name)) + geom_line() \
                + ggtitle(chart_title) + ylim(0, 1) + geom_hline(yintercept=self.target, color='black')
            # Can add confidence intervals once I work out linetype=??? in ggplot

            return p
        else:
            import matplotlib.pyplot as plt
            import numpy as np
            dl = self.dose_levels()
            prior_tox = self.prior
            post_tox = self.prob_tox()
            post_tox_lower = self.get_tox_prob_quantile(0.05)
            post_tox_upper = self.get_tox_prob_quantile(0.95)
            plt.plot(dl, prior_tox, '--', c='black')
            plt.plot(dl, post_tox, '-', c='black')
            plt.plot(dl, post_tox_lower, '-.', c='black')
            plt.plot(dl, post_tox_upper, '-.', c='black')
            plt.scatter(dl, prior_tox, marker='x', s=300, facecolors='none', edgecolors='k')
            plt.scatter(dl, post_tox, marker='o', s=300, facecolors='none', edgecolors='k')
            plt.axhline(self.target)
            plt.ylim(0, 1)
            plt.xlim(np.min(dl), np.max(dl))
            plt.xticks(dl)
            plt.ylabel('Probability of toxicity')
            plt.xlabel('Dose level')
            plt.title(chart_title)

            p = plt.gcf()
            phi = (np.sqrt(5) + 1) / 2.
            p.set_size_inches(12, 12 / phi)
            # return p


def crm_dtp_detail(trial):
    """ Performs the CRM-specific extra reporting when calculating DTPs
    :param trial: instance of CRM
    :return: OrderedDict

    """

    to_return = OrderedDict()

    if trial.beta_hat is not None:
        to_return['BetaHat'] = atomic_to_json(trial.beta_hat)
    if trial.beta_var is not None:
        to_return['BetaVar'] = atomic_to_json(trial.beta_var)

    if trial.prob_tox() is not None:
        to_return['ProbTox'] = iterable_to_json(trial.prob_tox())
        for i, dl in enumerate(trial.dose_levels()):
            to_return['ProbTox{}'.format(dl)] = trial.prob_tox()[i]

    return to_return
