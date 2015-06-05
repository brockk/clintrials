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
# import pandas as pd
from scipy.stats import norm, beta
from random import sample

from clintrials.common import empiric, inverse_empiric
from clintrials.dosefinding import EfficacyToxicityDoseFindingTrial
from clintrials.dosefinding.crm import CRM
from clintrials.dosefinding.efftox import solve_metrizable_efftox_scenario
from clintrials.dosefinding.wagestait import wt_get_theta_hat
from clintrials.util import correlated_binary_outcomes_from_uniforms


class BrockYapEfficacyToxicityDoseFindingTrial(EfficacyToxicityDoseFindingTrial):
    """ This is Brock & Yap's fusion of Wages & Tait adaptive phase I/II design with EffTox's utility contours.

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
    >>> from crctu.trialdesigns.dosefinding.efftox import LpNormCurve
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
                 tox_certainty=0.05, deficient_efficacy_alpha=0.05,
                 model_prior_weights=None, use_quick_integration=True, estimate_var=True,
                 prevent_skipping_untolerated=True, must_try_lowest_dose=True):
        """

        Params:
        skeletons, 2-d matrix of skeletons, i.e. list of prior efficacy scenarios, one scenario per row
        prior_tox_probs, list of prior probabilities of toxicity, from least toxic to most
        tox_target, target probability of toxicity in CRM
        tox_limit, the maximum acceptable probability of toxicity
        eff_limit, the minimum acceptable probability of efficacy
        metric, instance of LpNormCurve or InverseQuadraticCurve, used to calculate utility
                of efficacy/toxicity probability pairs.
        first_dose, starting dose level, 1-based. I.e. first_dose=3 means the middle dose of 5.
        max_size, maximum number of patients to use in trial
        stage_one_size, size of the first stage of the trial, where dose is set by a latent CRM model only,
                        i.e. only toxicity monitoring is performed with no attempt to monitor efficacy. 0 by default
        F_func and inverse_F, the link function and inverse for CRM method, e.g. logistic and inverse_logistic
        theta_prior, prior distibution for theta parameter, the single parameter in the efficacy models
        beta_prior, prior distibution for beta parameter, the single parameter in the toxicity CRM model
        tox_certainty, significance to use when testing that lowest dose exceeds toxicity limit
        deficient_efficacy_alpha, significance to use when testing that optimal dose has efficacy less than
                                    efficacy limit
        model_prior_weights, vector of prior probabilities that each model is correct. None to use uniform weights
        use_quick_integration, numerical integration is slow. Set this to False to use the most accurate (slowest)
                                method; False to use a quick but approximate method.
                                In simulations, fast and approximate often suffices.
                                In trial scenarios, use slow and accurate!
        estimate_var, True to estimate the posterior variances of theta and beta
        prevent_skipping_untolerated, Unused, TODO
        must_try_lowest_dose, Unused, TODO

        """

        EfficacyToxicityDoseFindingTrial.__init__(self, first_dose, len(prior_tox_probs), max_size)

        self.skeletons = skeletons
        self.K, self.I = np.array(skeletons).shape
        if self.I != len(prior_tox_probs):
            ValueError('prior_tox_probs should have %s items.' % self.I)
        self.prior_tox_probs = prior_tox_probs
        self.tox_limit = tox_limit
        self.eff_limit = eff_limit
        self.metric = metric
        self.stage_one_size = stage_one_size
        self.F_func = F_func
        self.inverse_F = inverse_F
        self.theta_prior = theta_prior
        self.beta_prior = beta_prior
        self.tox_certainty = tox_certainty
        self.deficient_efficacy_alpha = deficient_efficacy_alpha
        if model_prior_weights:
            if self.K != len(model_prior_weights):
                ValueError('model_prior_weights should have %s items.' % self.K)
            if sum(model_prior_weights) == 0:
                ValueError('model_prior_weights cannot sum to zero.')
            self.model_prior_weights = model_prior_weights / sum(model_prior_weights)
        else:
            self.model_prior_weights = np.ones(self.K) / self.K
        self.use_quick_integration = use_quick_integration
        self.estimate_var = estimate_var

        # Reset
        self.most_likely_model_index = \
            sample(np.array(range(self.K))[self.model_prior_weights == max(self.model_prior_weights)], 1)[0]
        self.w = np.zeros(self.K)
        self.crm = CRM(prior=prior_tox_probs, target=tox_target, first_dose=first_dose, max_size=max_size,
                       F_func=empiric, inverse_F=inverse_empiric, beta_prior=beta_prior,
                       use_quick_integration=use_quick_integration, estimate_var=estimate_var,
                       avoid_skipping_untried_escalation=True)
        self.post_tox_probs = np.zeros(self.I)
        self.post_eff_probs = np.zeros(self.I)
        self.theta_hats = np.zeros(self.K)

        self.utility = []
        self.dose_allocation_mode = 0

        self.prevent_skipping_untolerated = prevent_skipping_untolerated  # TODO: plumb these in
        self.must_try_lowest_dose = must_try_lowest_dose

    def dose_toxicity_lower_bound(self, dose_level, alpha=0.05):
        """ Get lower bound of toxicity probability at a dose-level using the Clopper-Pearson aka Beta aka exact method.

        Params:
        dose-level, 1-based index of dose level
        alpha, significance level, i.e. alpha% of probabilities will be less than response

        Returns: a probability

        """
        if 0 < dose_level <= len(self.post_tox_probs):
            n = self.treated_at_dose(dose_level)
            x = self.toxicities_at_dose(dose_level)
            if n > 0:
                ci = beta(x, n-x+1).ppf(alpha/2), beta(x+1, n-x).ppf(1-alpha/2)
                return ci[0]
        # Default
        return np.NaN

    def dose_efficacy_upper_bound(self, dose_level, alpha=0.05):
        """ Get upper bound of efficacy probability at a dose-level using the Clopper-Pearson aka Beta aka exact method.

        Params:
        dose-level, 1-based index of dose level
        alpha, significance level, i.e. alpha% of probabilities will be greater than response

        Returns: a probability

        """
        if 0 < dose_level <= len(self.post_eff_probs):
            n = self.treated_at_dose(dose_level)
            x = self.efficacies_at_dose(dose_level)
            if n > 0:
                ci = beta(x, n-x+1).ppf(alpha/2), beta(x+1, n-x).ppf(1-alpha/2)
                return ci[1]
        # Default
        return np.NaN

    def model_theta_hat(self):
        """ Return theta hat for the model with the highest posterior likelihood, i.e. the current model. """
        return self.theta_hats[self.most_likely_model_index]

    def _EfficacyToxicityDoseFindingTrial__calculate_next_dose(self):
        cases = zip(self._doses, self._toxicities, self._efficacies)
        # Update parameters for efficacy estimates
        integrals = wt_get_theta_hat(cases, self.skeletons, self.theta_prior,
                                     use_quick_integration=self.use_quick_integration, estimate_var=False)
        theta_hats, theta_vars, model_probs = zip(*integrals)
        self.theta_hats = theta_hats
        w = self.model_prior_weights * model_probs
        self.w = w / sum(w)
        most_likely_model_index = np.argmax(w)
        self.most_likely_model_index = most_likely_model_index
        # TODO: Plug-in means. Better way?
        self.post_eff_probs = empiric(self.skeletons[most_likely_model_index],
                                      beta=theta_hats[most_likely_model_index])
        self.post_tox_probs = empiric(self.prior_tox_probs, beta=self.crm.beta_hat)

        # Update combined model
        if self.size() < self.stage_one_size:
            # self._next_dose = self._stage_one_next_dose(self.post_tox_probs)
            self._next_dose = self._stage_one_next_dose()
        else:
            self._next_dose = self._stage_two_next_dose(self.post_tox_probs, self.post_eff_probs)

        # Stop if upper bound of efficacy at optimum dose is less than eff_limit
        # TODO: In time, change this rule to use posterior probabilities too, like the way toxicity is
        # tested in _stage_one_next_dose and _stage_two_next_dose.
        if self.size() >= self.stage_one_size:
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

    def _EfficacyToxicityDoseFindingTrial__process_cases(self, cases):
        """ Subclasses should override this method to perform an cases-specific processing. """
        # Update CRM toxicity model
        toxicity_cases = []
        for (dose, tox, eff) in cases:
            toxicity_cases.append((dose, tox))
        self.crm.update(toxicity_cases)

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

    # Private interface
    def _stage_one_next_dose(self):
        # There is no scrutiny of what is efficable in 'admissable'
        prob_unacc_tox = self.crm.prob_tox_exceeds(self.tox_limit, n=10**5)
        admissable = [x < (1-self.tox_certainty) for x in prob_unacc_tox]
        admissable_set = [i+1 for i, x in enumerate(admissable) if x]
        self._admissable_set = admissable_set
        if sum(admissable) > 0:
            self._status = 1
            self.dose_allocation_mode = 0.5  # TODO
            return self.crm.next_dose()
        else:
            # All doses are too toxic so stop trial
            self._status = -1
            self.dose_allocation_mode = 0
            return -1

    def _stage_two_next_dose(self, tox_probs, eff_probs):
        # There is no scrutiny of what is efficable in 'admissable'
        prob_unacc_tox = self.crm.prob_tox_exceeds(self.tox_limit, n=10**5)
        admissable = [x < (1-self.tox_certainty) for x in prob_unacc_tox]
        admissable_set = [i+1 for i, x in enumerate(admissable) if x]
        self._admissable_set = admissable_set
        # Beware: I normally use (tox, eff) pairs but the metric expects (eff, tox) pairs, driven
        # by the equation form that Thall & Cook chose.
        utility = np.array([self.metric(x[0], x[1]) for x in zip(eff_probs, tox_probs)])
        self.utility = utility
        if sum(admissable) > 0:
            # Select most desirable dose from admissable set
            masked_util = np.where(admissable, utility, np.nan)
            ideal_dose = np.nanargmax(masked_util) + 1
            max_dose_given = self.maximum_dose_given()
            if max_dose_given and ideal_dose - max_dose_given > 1:
                # Prevent skipping untried doses in escalation
                self._status = 1
                self.dose_allocation_mode = 2
                return max_dose_given + 1
            else:
                self._status = 1
                self.dose_allocation_mode = 1
                return ideal_dose
        else:
            # All doses are too toxic so stop trial
            self._status = -1
            self.dose_allocation_mode = 0
            return -1
