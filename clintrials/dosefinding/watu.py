__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'


""" Brock & Yap's novel seamless phase I/II efficacy/toxicity design, fusing elements of Wages & Tait's design with
elements of Thall & Cook's EffTox design.

See:
Wages, N.A. & Tait, C. - Seamless Phase I/II Adaptive Design For Oncology Trials
                    of Molecularly Targeted Agents, to appear in Journal of Biopharmaceutical Statistics
Thall, P.F. and Cook, J.D. (2004). Dose-Finding Based on Efficacy-Toxicity Trade-Offs, Biometrics, 60: 684-693.
Cook, J.D. Efficacy-Toxicity trade-offs based on L^p norms, Technical Report UTMDABTR-003-06, April 2006
Berry, Carlin, Lee and Mueller. Bayesian Adaptive Methods for Clinical Trials, Chapman & Hall / CRC Press

"""


import numpy as np
from scipy.stats import norm, beta
from random import sample

from clintrials.common import empiric, inverse_empiric
from clintrials.dosefinding.crm import CRM
from clintrials.dosefinding.efficacytoxicity import EfficacyToxicityDoseFindingTrial
from clintrials.dosefinding.efftox import solve_metrizable_efftox_scenario
from clintrials.dosefinding.wagestait import _wt_get_theta_hat, _get_post_eff_bayes


class WATU(EfficacyToxicityDoseFindingTrial):
    """ Brock & Yap's fusion of Wages & Tait's phase I/II design with Thall & Cook's EffTox utility contours.

    See Wages, N.A. & Tait, C. - Seamless Phase I/II Adaptive Design For Oncology Trials
                    of Molecularly Targeted Agents, to appear in Journal of Biopharmaceutical Statistics
        Thall, P.F. and Cook, J.D. (2004). Dose-Finding Based on Efficacy-Toxicity Trade-Offs, Biometrics, 60: 684-693.
        Cook, J.D. Efficacy-Toxicity trade-offs based on L^p norms, Technical Report UTMDABTR-003-06, April 2006

    e.g. general usage
    >>> trial_size = 30
    >>> first_dose = 3
    >>> tox_target = 0.35
    >>> tox_limit = 0.40
    >>> eff_limit = 0.45
    >>> stage_one_size = 0  # Else initial dose choices will be random; not good for testing!
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
    >>> from clintrials.dosefinding.efftox import LpNormCurve
    >>> hinge_points = [(0.4, 0), (1, 0.7), (0.5, 0.4)]
    >>> metric = LpNormCurve(hinge_points[0][0], hinge_points[1][1], hinge_points[2][0], hinge_points[2][1])
    >>> trial = BrockYapEfficacyToxicityDoseFindingTrial(skeletons, prior_tox_probs, tox_target, tox_limit,
    ...                                                  eff_limit, metric, first_dose, trial_size, stage_one_size)
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

    def __init__(self, skeletons, prior_tox_probs, tox_target, tox_limit, eff_limit, metric,
                 first_dose, max_size, stage_one_size=0,
                 F_func=empiric, inverse_F=inverse_empiric,
                 theta_prior=norm(0, np.sqrt(1.34)), beta_prior=norm(0, np.sqrt(1.34)),
                 tox_certainty=0.05, eff_certainty=0.05,
                 model_prior_weights=None, use_quick_integration=True, estimate_var=True,
                 avoid_skipping_untried_escalation_stage_1=True, avoid_skipping_untried_deescalation_stage_1=True,
                 avoid_skipping_untried_escalation_stage_2=True, avoid_skipping_untried_deescalation_stage_2=True,
                 must_try_lowest_dose=True,
                 plugin_mean=False
                 ):
        """

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
        :param metric: instance of LpNormCurve or InverseQuadraticCurve, used to calculate utility
                        of efficacy/toxicity probability pairs.
        :type metric: clintrials.dosefinding.efftox.LpNormCurve
        :param first_dose: starting dose level, 1-based. I.e. intcpt=3 means the middle dose of 5.
        :type first_dose: int
        :param max_size: maximum number of patients to use
        :type max_size: int
        :param stage_one_size: size of the first stage of the trial, where dose is set by a latent CRM model only,
                        i.e. only toxicity monitoring is performed with no attempt to monitor efficacy. 0 by default
        :type stage_one_size: int
        :param F_func: the link function for CRM-like methods, e.g. empiric
        :type F_func: func
        :param inverse_F: the inverse link function for CRM-like methods method, e.g. inverse_empiric
        :type inverse_F: func
        :param theta_prior: prior distibution for theta parameter, the single parameter in the efficacy models
        :type theta_prior: scipy.stats.rv_continuous
        :param beta_prior: prior distibution for beta parameter, the single parameter in the toxicity CRM model
        :type beta_prior: scipy.stats.rv_continuous
        :param tox_certainty: the posterior certainty required that toxicity is less than cutoff
        :type tox_certainty: float
        :param model_prior_weights: list of prior probabilities that each model is correct. None to use uniform weights
        :type model_prior_weights: list
        :param use_quick_integration: numerical integration is slow. Set this to False to use the most accurate (& slow)
                                method; False to use a quick but approximate method.
                                In simulations, fast and approximate often suffices.
                                In trial scenarios, use slow and accurate!
        :type use_quick_integration: bool
        :param estimate_var: True to estimate the posterior variance of beta and theta
        :type estimate_var: bool
        :param avoid_skipping_untried_escalation_stage_1: True to avoid skipping untried doses in escalation in stage 1
        :type avoid_skipping_untried_escalation_stage_1: bool
        :param avoid_skipping_untried_deescalation_stage_1: True to avoid skipping untried doses in de-escalation in
        stage 1
        :type avoid_skipping_untried_deescalation_stage_1: bool
        :param avoid_skipping_untried_escalation_stage_2: True to avoid skipping untried doses in escalation in stage 2
        :type avoid_skipping_untried_escalation_stage_2: bool
        :param avoid_skipping_untried_deescalation_stage_2: True to avoid skipping untried doses in de-escalation in
        stage 2
        :type avoid_skipping_untried_deescalation_stage_2: bool
        :param must_try_lowest_dose: Unused, TODO
        :type must_try_lowest_dose: bool
        :param plugin_mean: True to estimate event curves by plugging parameter estimate into function;
                            False to estimate using full Bayesian integral (default).
        :type plugin_mean: bool

        """

        EfficacyToxicityDoseFindingTrial.__init__(self, first_dose, len(prior_tox_probs), max_size)

        self.skeletons = skeletons
        self.K, self.I = np.array(skeletons).shape
        if self.I != len(prior_tox_probs):
            ValueError('prior_tox_probs should have %s items.' % self.I)
        self.prior_tox_probs = prior_tox_probs
        self.tox_target = tox_target
        self.tox_limit = tox_limit
        self.eff_limit = eff_limit
        self.metric = metric
        self.stage_one_size = stage_one_size
        self.F_func = F_func
        self.inverse_F = inverse_F
        self.theta_prior = theta_prior
        self.beta_prior = beta_prior
        self.tox_certainty = tox_certainty
        self.eff_certainty = eff_certainty
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
        self.avoid_skipping_untried_escalation_stage_1 = avoid_skipping_untried_escalation_stage_1
        self.avoid_skipping_untried_deescalation_stage_1 = avoid_skipping_untried_deescalation_stage_1
        self.avoid_skipping_untried_escalation_stage_2 = avoid_skipping_untried_escalation_stage_2
        self.avoid_skipping_untried_deescalation_stage_2 = avoid_skipping_untried_deescalation_stage_2
        self.must_try_lowest_dose = must_try_lowest_dose
        self.plugin_mean = plugin_mean

        # Reset
        self.most_likely_model_index = \
            np.random.choice(np.array(range(self.K))[self.model_prior_weights == max(self.model_prior_weights)], 1)[0]
        self.w = np.zeros(self.K)
        self.crm = CRM(prior=prior_tox_probs, target=tox_target, first_dose=first_dose, max_size=max_size,
                       F_func=empiric, inverse_F=inverse_empiric, beta_prior=beta_prior,
                       use_quick_integration=use_quick_integration, estimate_var=estimate_var,
                       avoid_skipping_untried_escalation=avoid_skipping_untried_escalation_stage_1,
                       avoid_skipping_untried_deescalation=avoid_skipping_untried_deescalation_stage_1,
                       plugin_mean=plugin_mean)
        self.post_tox_probs = np.zeros(self.I)
        self.post_eff_probs = np.zeros(self.I)
        self.theta_hats = np.zeros(self.K)
        self.theta_vars = np.zeros(self.K)

        self.utility = []
        self.dose_allocation_mode = 0

    def model_theta_hat(self):
        """ Return theta hat for the model with the highest posterior likelihood, i.e. the current model. """
        return self.theta_hats[self.most_likely_model_index]

    def model_theta_var(self):
        """ Return theta var for the model with the highest posterior likelihood, i.e. the current model. """
        return self.theta_vars[self.most_likely_model_index]

    def _EfficacyToxicityDoseFindingTrial__calculate_next_dose(self):
        cases = list(zip(self._doses, self._toxicities, self._efficacies))
        toxicity_cases = []
        for (dose, tox, eff) in cases:
            toxicity_cases.append((dose, tox))
        self.crm.reset()
        self.crm.update(toxicity_cases)

        # Update parameters for efficacy estimates
        integrals = _wt_get_theta_hat(cases, self.skeletons, self.theta_prior,
                                      use_quick_integration=self.use_quick_integration, estimate_var=True)
        theta_hats, theta_vars, model_probs = zip(*integrals)
        self.theta_hats = theta_hats
        self.theta_vars = theta_vars
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
        if self.size() < self.stage_one_size:
            self._next_dose = self._stage_one_next_dose()
        else:
            self._next_dose = self._stage_two_next_dose(self.post_tox_probs, self.post_eff_probs)

        return self._next_dose

    def _EfficacyToxicityDoseFindingTrial__reset(self):
        """ Opportunity to run implementation-specific reset operations. """
        self.most_likely_model_index = \
            sample(np.array(range(self.K))[self.model_prior_weights == max(self.model_prior_weights)], 1)[0]
        self.w = np.zeros(self.K)
        self.post_tox_probs = np.zeros(self.I)
        self.post_eff_probs = np.zeros(self.I)
        self.theta_hats = np.zeros(self.K)
        self.theta_vars = np.zeros(self.K)
        self.crm.reset()

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

        admiss, u, u_star, obd, u_cushtion = solve_metrizable_efftox_scenario(prob_tox, prob_eff, self.metric,
                                                                              self.tox_limit, self.eff_limit)
        return obd

    def prob_eff_exceeds(self, eff_cutoff, n=10**6):

        # Estimate probability of efficacy exceeds eff_cutoff using plug-in mean and variance for theta
        # and randomly sampling values from normal. Why normal? Because the prior for is normal and the posterior
        # is asymptotically normal. For low n, non-normality may lead to bias.
        # TODO: research replacing this with a proper posterior integral
        theta_sample = norm(loc=self.model_theta_hat(), scale=np.sqrt(self.model_theta_var())).rvs(n)
        p0_sample = [empiric(prob, beta=theta_sample) for prob in self.skeletons[self.most_likely_model_index]]
        return np.array([np.mean(x > eff_cutoff) for x in p0_sample])

    def prob_acc_eff(self, threshold=None, **kwargs):
        if threshold is None:
            threshold = self.eff_limit
        return self.prob_eff_exceeds(threshold, **kwargs)

    def prob_acc_tox(self, threshold=None, **kwargs):
        if threshold is None:
            threshold = self.tox_limit
        return 1 - self.crm.prob_tox_exceeds(threshold, **kwargs)

    # Private interface
    def _stage_one_next_dose(self):

        prob_unacc_tox = self.crm.prob_tox_exceeds(self.tox_limit, n=10**5)  # TODO: make n=10**5 editable
        prob_unacc_eff = 1 - self.prob_eff_exceeds(self.eff_limit, n=10**5)
        admissable = [(prob_tox < (1-self.tox_certainty)) and (prob_eff < (1-self.eff_certainty))
                      for (prob_eff, prob_tox) in zip(prob_unacc_eff, prob_unacc_tox)]
        admissable_set = [i+1 for i, x in enumerate(admissable) if x]
        self._admissable_set = admissable_set

        if self.size() > 0:
            # Trial has started so modelling may commence
            max_dose_given = self.maximum_dose_given()
            min_dose_given = self.minimum_dose_given()
            attractiveness = np.abs(np.array(self.crm.prob_tox()) - self.tox_target)  # Rank as proximity to tox target
            for i in np.argsort(attractiveness):  # dose-indices from closest to farthest from tox target
                dose_level = i+1
                if dose_level in admissable_set:
                    if (self.avoid_skipping_untried_escalation_stage_1 and max_dose_given
                        and dose_level - max_dose_given > 1):
                        pass  # No skipping in escalation
                    elif (self.avoid_skipping_untried_deescalation_stage_1 and min_dose_given
                          and min_dose_given - dose_level > 1):
                        pass  # No skipping in de-escalation
                    else:
                        self._status = 1
                        self._next_dose = dose_level
                        break
            else:
                if self.must_try_lowest_dose and self.treated_at_dose(1) <= 0:
                    # Lowest dose has not been tried. Try it now rather than stop:
                    self._next_dose = 1
                    self._status = 1
                else:
                    # No dose can be selected so stop
                    self._next_dose = -1
                    self._status = -1
        else:
            # Trial has not yet started
            self._next_dose = self.first_dose()
            self._status = -10

        return self._next_dose

    def _stage_two_next_dose(self, tox_probs, eff_probs):

        prob_unacc_tox = self.crm.prob_tox_exceeds(self.tox_limit, n=10**5) # TODO: make n=10**5 editable
        prob_unacc_eff = 1 - self.prob_eff_exceeds(self.eff_limit, n=10**5)
        admissable = [(prob_tox < (1-self.tox_certainty)) and (prob_eff < (1-self.eff_certainty))
                      for (prob_eff, prob_tox) in zip(prob_unacc_eff, prob_unacc_tox)]
        admissable_set = [i+1 for i, x in enumerate(admissable) if x]
        self._admissable_set = admissable_set
        # Beware: I normally use (tox, eff) pairs but the metric expects (eff, tox) pairs, driven
        # by the equation form that Thall & Cook chose.
        utility = np.array([self.metric(x[0], x[1]) for x in zip(eff_probs, tox_probs)])
        self.utility = utility

        if self.size() > 0:
            # Trial has started so modelling may commence
            max_dose_given = self.maximum_dose_given()
            min_dose_given = self.minimum_dose_given()
            for i in np.argsort(-utility):  # dose-indices from highest to lowest utility
                dose_level = i+1
                if dose_level in admissable_set:
                    if (self.avoid_skipping_untried_escalation_stage_2 and max_dose_given
                            and dose_level - max_dose_given > 1):
                        pass  # No skipping in escalation
                    elif (self.avoid_skipping_untried_deescalation_stage_2 and min_dose_given
                          and min_dose_given - dose_level > 1):
                        pass  # No skipping in de-escalation
                    else:
                        self._status = 1
                        self._next_dose = dose_level
                        break
            else:
                if self.must_try_lowest_dose and self.treated_at_dose(1) <= 0:
                    # Lowest dose has not been tried. Try it now rather than stop:
                    self._next_dose = 1
                    self._status = 1
                else:
                    # No dose can be selected so stop
                    self._next_dose = -1
                    self._status = -1
        else:
            # Trial has not yet started
            self._next_dose = self.first_dose()
            self._status = -10

        return self._next_dose
