__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'

__all__ = ["peps2v1", "peps2v2"]

"""

 BeBOP: Bayesian design with Bivariate Outcomes and Predictive variables
 Brock, et al. To be published.

 BeBOP studies the dual primary outcomes efficacy and toxicity.
 The two events can be associated to reflect the potential for correlated
 outcomes. The design models the probabilities of efficacy and toxicity
 using logistic models so that the information in predictive variables
 can be incorporated to tailor the treatment acceptance / rejection
 decision.

 This is a generalisation of the design that was used in the PePS2 trial.
 PePS2 studies the efficacy and toxicity of a drug in a population of
 performance status 2 lung cancer patients. Patient outcomes may plausibly
 be effected by whether or not they have been treated before, and the
 expression rate of PD-L1 in their cells.

 PePS2 uses Brock et al's BeBOP design to incorporate the potentially
 predictive data in PD-L1 expression rate and whether or not a patient has
 been pre-treated to find the sub-population(s) where the drug works
 and is tolerable.

"""

import numpy
import pandas as pd

from clintrials.stats import ProbabilityDensitySample


class BeBOP():
    """
    """

    def __init__(self, theta_priors, efficacy_model, toxicity_model, joint_model):
        """

        Params:
        :param theta_priors: list of prior distributions for elements of parameter vector, theta.
                        Each prior object should support obj.ppf(x) and obj.pdf(x) like classes in scipy
        :param efficacy_model: func with signature x, theta; where x is a case vector and theta a 2d array of
          parameter values, the first column containing values for the first parameter, the second column the
          second parameter, etc, so that each row in theta is a single parameter set. Function should return probability
          of efficacy of case x under each parameter set (i.e. each row of theta) so that a 1*len(theta) array should
          be returned.
        :param toxicity_model: func with signature x, theta; where x is a case vector and theta a 2d array of
          parameter values, the first column containing values for the first parameter, the second column the
          second parameter, etc, so that each row in theta is a single parameter set. Function should return probability
          of toxicity of case x under each parameter set (i.e. each row of theta) so that a 1*len(theta) array should
          be returned.
        :param joint_model: func with signature x, theta; where x is a case vector and theta a 2d array of
          parameter values, the first column containing values for the first parameter, the second column the
          second parameter, etc, so that each row in theta is a single parameter set. Function should return the joint
          probability of efficacy and toxicity of case x under each parameter set (i.e. each row of theta) so that
          a 1*len(theta) array should be returned. Generally this method would use efficacy_model and toxicity_model.
          For non-associated events, for instance, the simple product of efficacy_model(x, theta) and
          toxicity_model(x, theta) would do the job. For associated events, more complexity is required.

        In case vector x, the element x[0] should be boolean efficacy variable, with 1 showing efficacy occurred.
        In case vector x, the element x[1] should be boolean toxicity variable, with 1 showing toxicity occurred.

        See clintrials.phase2.bebop.peps2v2 for a working trio of efficacy_model, toxicity_model and joint_model that
        allow for associated efficacy and toxicity events.

        Note: efficacy_model, toxicity_model and joint_model should be vectorised to work with one case and many
        parameter sets (rather than just many cases and one parameter set) for quick integration using Monte Carlo.

        """

        self.priors = theta_priors
        self._pi_e = efficacy_model
        self._pi_t = toxicity_model
        self._pi_ab = joint_model
        # Initialise model
        self.reset()

    def reset(self):
        self.cases = []
        self._pds = None

    def _l_n(self, D, theta):
        if len(D) > 0:
            lik = numpy.array(map(lambda x: self._pi_ab(x, theta), D))
            return lik.prod(axis=0)
        else:
            return numpy.ones(len(theta))

    def size(self):
        return len(self.cases)

    def efficacies(self):
        return [case[0] for case in self.cases]

    def toxicities(self):
        return [case[1] for case in self.cases]

    def get_case_elements(self, i):
        return [case[i] for case in self.cases]

    def update(self, cases, n=10**6, epsilon = 0.00001, **kwargs):
        """ TODO

        :param n:
        :param epsilon:

        """

        self.cases.extend(cases)
        limits = [(dist.ppf(epsilon), dist.ppf(1-epsilon)) for dist in self.priors]
        samp = numpy.column_stack([numpy.random.uniform(*limit_pair, size=n) for limit_pair in limits])
        lik_integrand = lambda x: self._l_n(cases, x) * numpy.prod(numpy.array([dist.pdf(col) for (dist, col) in zip(self.priors, x.T)]), axis=0)
        self._pds = ProbabilityDensitySample(samp, lik_integrand)
        return

    def _predict_case(self, case, eff_cutoff, tox_cutoff, pds, samp, estimate_ci=False):
        x = case
        eff_probs = self._pi_e(x, samp)
        tox_probs = self._pi_t(x, samp)
        from collections import OrderedDict
        predictions = OrderedDict([
            ('Pr(Eff)', pds.expectation(eff_probs)),
            ('Pr(Tox)', pds.expectation(tox_probs)),
            ('Pr(AccEff)', pds.expectation((eff_probs > eff_cutoff))),
            ('Pr(AccTox)', pds.expectation((tox_probs < tox_cutoff))),
        ])

        if estimate_ci:
            predictions['Pr(Eff) Lower'] = pds.quantile_vector(eff_probs, 0.05, start_value=0.05)
            predictions['Pr(Eff) Upper'] = pds.quantile_vector(eff_probs, 0.95, start_value=0.95)
            predictions['Pr(Tox) Lower'] = pds.quantile_vector(tox_probs, 0.05, start_value=0.05)
            predictions['Pr(Tox) Upper'] = pds.quantile_vector(tox_probs, 0.95, start_value=0.95)
        return predictions

    def predict(self, cases, eff_cutoff, tox_cutoff, to_pandas=False, estimate_ci=False):
        if self._pds is not None:
            pds = self._pds
            samp = pds._samp
            fitted = [self._predict_case(x, eff_cutoff, tox_cutoff, pds, samp, estimate_ci=estimate_ci) for x in cases]
            if to_pandas:
                if estimate_ci:
                    return pd.DataFrame(fitted, columns=['Pr(Eff)', 'Pr(Tox)', 'Pr(AccEff)', 'Pr(AccTox)',
                                                         'Pr(Eff) Lower', 'Pr(Eff) Upper', 'Pr(Tox) Lower', 'Pr(Tox) Upper'])
                else:
                    return pd.DataFrame(fitted, columns=['Pr(Eff)', 'Pr(Tox)', 'Pr(AccEff)', 'Pr(AccTox)'])
            else:
                return fitted
        else:
            return None

    def get_posterior_param_means(self):
        if self._pds:
            return numpy.apply_along_axis(lambda x: self._pds.expectation(x), 0, self._pds._samp)
        else:
            return []

    def theta_estimate(self, i, alpha=0.05):
        """ Get posterior confidence interval and mean estimate of element i in parameter vector.

        Returns (lower, mean, upper)

        """

        if j < len(self.priors):
            mu = self._pds.expectation(self._pds._samp[:,i])
            return numpy.array([self._pds.quantile(i, alpha/2), mu, self._pds.quantile(i, 1-alpha/2)])
        else:
            return (0,0,0)

#     def efficacy_effect(self, j, alpha=0.05):
#         """ Get confidence interval and mean estimate of the effect on efficacy, expressed as odds-ratios.

#         Use:
#         - j=0, to get treatment effect of the intercept variable
#         - j=1, to get treatment effect of the pre-treated status variable
#         - j=2, to get treatment effect of the mutation status variable

#         """

#         if j==0:
#             expected_log_or = self._pds.expectation(self._pds._samp[:,1])
#             return np.exp([self._pds.quantile(1, alpha/2), expected_log_or, self._pds.quantile(1, 1-alpha/2)])
#         elif j==1:
#             expected_log_or = self._pds.expectation(self._pds._samp[:,2])
#             return np.exp([self._pds.quantile(2, alpha/2), expected_log_or, self._pds.quantile(2, 1-alpha/2)])
#         elif j==2:
#             expected_log_or = self._pds.expectation(self._pds._samp[:,3])
#             return np.exp([self._pds.quantile(3, alpha/2), expected_log_or, self._pds.quantile(3, 1-alpha/2)])
#         else:
#             return (0,0,0)

#     def toxicity_effect(self, j=0, alpha=0.05):
#         """ Get confidence interval and mean estimate of the effect on toxicity, expressed as odds-ratios.

#         Use:
#         - j=0, to get effect on toxicity of the intercept variable

#         """

#         if j==0:
#             expected_log_or = self._pds.expectation(self._pds._samp[:,0])
#             return np.exp([self._pds.quantile(0, alpha/2), expected_log_or, self._pds.quantile(0, 1-alpha/2)])
#         else:
#             return (0,0,0)

#     def correlation_effect(self, alpha=0.05):
#         """ Get confidence interval and mean estimate of the correlation between efficacy and toxicity. """
#         expected_psi = self._pds.expectation(self._pds._samp[:,4])
#         psi_levels = np.array([self._pds.quantile(4, alpha/2), expected_psi, self._pds.quantile(4, 1-alpha/2)])
#         return (np.exp(psi_levels) - 1) / (np.exp(psi_levels) + 1)