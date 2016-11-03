__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'

""" An implementation of Thall & Cook's EffTox design for dose-finding in clinical trials.

See:
Thall, P.F. and Cook, J.D. (2004). Dose-Finding Based on Efficacy-Toxicity Trade-Offs, Biometrics, 60: 684-693.
Cook, J.D. Efficacy-Toxicity trade-offs based on L^p norms, Technical Report UTMDABTR-003-06, April 2006
Berry, Carlin, Lee and Mueller. Bayesian Adaptive Methods for Clinical Trials, Chapman & Hall / CRC Press

"""

from collections import OrderedDict
import logging

import numpy as np
from scipy.optimize import brentq

from clintrials.common import inverse_logit
from clintrials.dosefinding.efficacytoxicity import EfficacyToxicityDoseFindingTrial
from clintrials.stats import ProbabilityDensitySample
from clintrials.util import atomic_to_json, iterable_to_json


def scale_doses(real_doses):
    """
    :param real_doses:
    :return:
    """

    return np.log(real_doses) - np.mean(np.log(real_doses))


def _eta_T(scaled_dose, mu, beta):
    return mu + beta * scaled_dose


def _eta_E(scaled_dose, mu, beta1, beta2):
    return mu + beta1 * scaled_dose + beta2 * scaled_dose**2


def _pi_T(scaled_dose, mu, beta):
    return inverse_logit(_eta_T(scaled_dose, mu, beta))


def _pi_E(scaled_dose, mu, beta1, beta2):
    return inverse_logit(_eta_E(scaled_dose, mu, beta1, beta2))


def _pi_ab(scaled_dose, tox, eff, mu_T, beta_T, mu_E, beta1_E, beta2_E, psi):
    """ Calculate likelihood of observing toxicity and efficacy with given parameters. """
    a, b = eff, tox
    p1 = _pi_E(scaled_dose, mu_E, beta1_E, beta2_E)
    p2 = _pi_T(scaled_dose, mu_T, beta_T)
    response = p1**a * (1-p1)**(1-a) * p2**b * (1-p2)**(1-b)
    response += -1**(a+b) * p1 * (1-p1) * p2 * (1-p2) * (np.exp(psi) - 1) / (np.exp(psi) + 1)
    return response


def _L_n(D, mu_T, beta_T, mu_E, beta1_E, beta2_E, psi):
    """ Calculate compound likelihood of observing cases D with given parameters.

    Params:
    D, list of 3-tuples, (dose, toxicity, efficacy), where dose is on Thall & Cook's codified scale (see below),
                                toxicity = 1 for toxic event, 0 for tolerance event,
                                and efficacy = 1 for efficacious outcome, 0 for alternative.

    Note: Thall & Cook's codified scale is thus:
    If doses 10mg, 20mg and 25mg are given so that d = [10, 20, 25], then the codified doses, x, are
    x = ln(d) - mean(ln(dose)) = [-0.5365, 0.1567, 0.3798]

    """

    response = np.ones(len(mu_T))
    for scaled_dose, tox, eff in D:
        p = _pi_ab(scaled_dose, tox, eff, mu_T, beta_T, mu_E, beta1_E, beta2_E, psi)
        response *= p
    return response


def efftox_get_posterior_probs(cases, priors, scaled_doses, tox_cutoff, eff_cutoff, n=10**5):
    """ Get the posterior probabilities after having observed cumulative data D in an EffTox trial.

    Note: This function evaluates the posterior integrals using Monte Carlo integration. Thall & Cook
    use the method of Monahan & Genz. I imagine that is quicker and more accurate but it is also
    more difficult to program, so I have skipped it. It remains a medium-term aim, however, because
    this method is slow.

    Params:
    cases, list of 3-tuples, (dose, toxicity, efficacy), where dose is the given (1-based) dose level,
                    toxicity = 1 for a toxicity event; 0 for a tolerance event,
                    efficacy = 1 for an efficacy event; 0 for a non-efficacy event.
    priors, list of prior distributions corresponding to mu_T, beta_T, mu_E, beta1_E, beta2_E, psi respectively
            Each prior object should support obj.ppf(x) and obj.pdf(x)
    scaled_doses, ordered list of all possible doses where each dose is on Thall & Cook's codified scale (see below),
    tox_cutoff, the desired maximum toxicity
    eff_cutoff, the desired minimum efficacy
    n, number of random points to use in Monte Carlo integration.

    Returns:
    nested lists of posterior probabilities, [ Prob(Toxicity, Prob(Efficacy), Prob(Toxicity less than cutoff),
                Prob(Efficacy greater than cutoff)], for each dose in doses,
            i.e. returned obj is of length len(doses) and each interior list of length 4.

    Note: Thall & Cook's codified dose scale is thus:
    If doses 10mg, 20mg and 25mg are given so that d = [10, 20, 25], then the codified doses, x, are
    x = ln(d) - mean(ln(dose)) = [-0.5365, 0.1567, 0.3798]

    """
    if len(priors) != 6:
        raise ValueError('priors should have 6 items.')

    # Convert dose-levels given to dose amounts given
    if len(cases) > 0:
        dose_levels, tox_events, eff_events = zip(*cases)
        scaled_doses_given = [scaled_doses[x-1] for x in dose_levels]
        _cases = list(zip(scaled_doses_given, tox_events, eff_events))
    else:
        _cases = []

    # The ranges of integration must be specified. In truth, the integration range is (-Infinity, Infinity)
    # for each variable. In practice, though, integrating to infinity is problematic, especially in
    # 6 dimensions. The limits of integration should capture all probability density, but not be too
    # generous, e.g. -1000 to 1000 would be stupid because the density at most points would be practically zero.
    # I use percentage points of the various prior distributions. The risk is that if the prior
    # does not cover the posterior range well, it will not estimate it well. This needs attention. TODO
    epsilon = 0.000001
    limits = [(dist.ppf(epsilon), dist.ppf(1-epsilon)) for dist in priors]
    samp = np.column_stack([np.random.uniform(*limit_pair, size=n) for limit_pair in limits])

    lik_integrand = lambda x: _L_n(_cases, x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]) \
                              * priors[0].pdf(x[:, 0]) * priors[1].pdf(x[:, 1]) * priors[2].pdf(x[:, 2]) \
                              * priors[3].pdf(x[:, 3]) * priors[4].pdf(x[:, 4]) * priors[5].pdf(x[:, 5])
    pds = ProbabilityDensitySample(samp, lik_integrand)

    probs = []
    for x in scaled_doses:
        tox_probs = _pi_T(x, mu=samp[:,0], beta=samp[:,1])
        eff_probs = _pi_E(x, mu=samp[:,2], beta1=samp[:,3], beta2=samp[:,4])
        probs.append((
            pds.expectation(tox_probs),
            pds.expectation(eff_probs),
            pds.expectation(tox_probs < tox_cutoff),
            pds.expectation(eff_probs > eff_cutoff)
        ))

    return probs, pds
    
    
def efftox_get_posterior_params(cases, priors, scaled_doses, n=10**5):
    """ Get the posterior parameter estimates after having observed cumulative data D in an EffTox trial.

    Note: This function evaluates the posterior integrals using Monte Carlo integration. Thall & Cook
    use the method of Monahan & Genz. I imagine that is quicker and more accurate but it is also
    more difficult to program, so I have skipped it. It remains a medium-term aim, however, because
    this method is slow.

    Params:
    cases, list of 3-tuples, (dose, toxicity, efficacy), where dose is the given (1-based) dose level,
                    toxicity = 1 for a toxicity event; 0 for a tolerance event,
                    efficacy = 1 for an efficacy event; 0 for a non-efficacy event.
    priors, list of prior distributions corresponding to mu_T, beta_T, mu_E, beta1_E, beta2_E, psi respectively
            Each prior object should support obj.ppf(x) and obj.pdf(x)
    scaled_doses, ordered list of all possible doses where each dose is on Thall & Cook's codified scale (see below),
    tox_cutoff, the desired maximum toxicity
    eff_cutoff, the desired minimum efficacy
    n, number of random points to use in Monte Carlo integration.

    Returns:
    list of posterior parameters as tuples, [ (mu_T, beta_T, mu_E, beta_T_1, beta_T_2, psi) ], and that's it for now.
            i.e. returned obj is of length 1 and first interior tuple is of length 6.
            More objects might be added to the outer list eventually.

    Note: Thall & Cook's codified dose scale is thus:
    If doses 10mg, 20mg and 25mg are given so that d = [10, 20, 25], then the codified doses, x, are
    x = ln(d) - mean(ln(dose)) = [-0.5365, 0.1567, 0.3798]

    """

    if len(priors) != 6:
        raise ValueError('priors should have 6 items.')

    # Convert dose-levels given to dose amounts given
    if len(cases) > 0:
        dose_levels, tox_events, eff_events = zip(*cases)
        scaled_doses_given = [scaled_doses[x-1] for x in dose_levels]
        _cases = zip(scaled_doses_given, tox_events, eff_events)
    else:
        _cases = []

    # The ranges of integration must be specified. In truth, the integration range is (-Infinity, Infinity)
    # for each variable. In practice, though, integrating to infinity is problematic, especially in
    # 6 dimensions. The limits of integration should capture all probability density, but not be too
    # generous, e.g. -1000 to 1000 would be stupid because the density at most points would be practically zero.
    # I use percentage points of the various prior distributions. The risk is that if the prior
    # does not cover the posterior range well, it will not estimate it well. This needs attention. TODO
    epsilon = 0.000001
    limits = [(dist.ppf(epsilon), dist.ppf(1-epsilon)) for dist in priors]
    samp = np.column_stack([np.random.uniform(*limit_pair, size=n) for limit_pair in limits])

    lik_integrand = lambda x: _L_n(_cases, x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]) \
                              * priors[0].pdf(x[:, 0]) * priors[1].pdf(x[:, 1]) * priors[2].pdf(x[:, 2]) \
                              * priors[3].pdf(x[:, 3]) * priors[4].pdf(x[:, 4]) * priors[5].pdf(x[:, 5])
    pds = ProbabilityDensitySample(samp, lik_integrand)

    params = []
    params.append(
        (
            pds.expectation(samp[:, 0]),
            pds.expectation(samp[:, 1]),
            pds.expectation(samp[:, 2]),
            pds.expectation(samp[:, 3]),
            pds.expectation(samp[:, 4]),
            pds.expectation(samp[:, 5]),
        )
    )

    return params, pds


# Desirability metrics
class LpNormCurve:
    """ Fit an indifference contour using three points and an L-p norm.

    The three points are:
    * efficacy when toxicity is impossible;
    * toxicity when efficacy is guaranteed;
    * an equallty desirable hinge point (hinge_eff, hinge_tox) in (0,1)^2

    The official EffTox software has used L^p Norms in the trade-off analysis since approximately version 2.
    This is the current method as at Aug 2014.

    For more information, consult the Cook (2006) paper and the Bayesian methods book for more information.
    The hinge point is the equally attractive point of the three that is not on the x- or y-axis.
    The p-parameter in the L-p norm is initialised by setting r = 1 as per p.119 in Berry et al

    """

    def __init__(self, minimum_tolerable_efficacy, maximum_tolerable_toxicity, hinge_prob_eff, hinge_prob_tox):
        """
        Params:
        minimum_tolerable_efficacy, pi_E^*, the tolerable efficacy when toxicity is impossible
        maximum_tolerable_toxicity, pi_T^*, the tolerable toxicity when efficacy is guaranteed
        hinge_prob_eff, probability of efficacy at the hinge point
        hinge_prob_tox, probability of toxicity at the hinge point

        """

        if hinge_prob_tox >= maximum_tolerable_toxicity:
            raise ValueError('Probability of toxicity at hinge point should be less than toxicity upper bound.')
        if hinge_prob_eff <= minimum_tolerable_efficacy:
            raise ValueError('Probability of efficacy at hinge point should be greater than efficacy lower bound.')

        def _find_p(p):
            a = ((1-hinge_prob_eff)/(1-minimum_tolerable_efficacy))
            b = hinge_prob_tox / maximum_tolerable_toxicity
            return a**p + b**p - 1
        self.minimum_tolerable_efficacy = minimum_tolerable_efficacy
        self.maximum_tolerable_toxicity = maximum_tolerable_toxicity
        self.p = brentq(_find_p, 0, 100)
        self.hinge_points = [(minimum_tolerable_efficacy, 0), (1, maximum_tolerable_toxicity),
                             (hinge_prob_eff, hinge_prob_tox)]

    def __call__(self, prob_eff, prob_tox):
        x = prob_eff
        y = prob_tox
        if np.all(0 < x) and np.all(x < 1) and np.all(0 < y) and np.all(y < 1):
            a = ((1 - x) / (1 - self.minimum_tolerable_efficacy))
            b = y / self.maximum_tolerable_toxicity
            r_to_the_p = a**self.p + b**self.p
            return 1 - r_to_the_p ** (1./self.p)
        else:
            response = np.zeros_like(x)
            response *= np.nan
            return response

    def solve(self, prob_eff=None, prob_tox=None, delta=0):
        """ Specify exactly one of prob_eff or prob_tox and this will return the other, for given delta"""

        if prob_eff is None and prob_tox is None:
            raise Exception('Specify prob_eff or prob_tox')
        if prob_eff is not None and prob_tox is not None:
            raise Exception('Specify just one of prob_eff and prob_tox')

        x, y = prob_eff, prob_tox
        x_l, y_l = self.minimum_tolerable_efficacy, self.maximum_tolerable_toxicity
        scaled_delta = (1-delta)**self.p
        if x is None:
            # Solve for x
            b = y / y_l
            a = (scaled_delta - b**self.p)**(1/self.p)
            return 1 - (1 - x_l)*a
        else:
            # Solve for y
            a = ((1 - x) / (1 - x_l))
            b = (scaled_delta - a**self.p)**(1/self.p)
            return b*y_l

    def get_tox(self, eff, util=0.0):
        """ Get equivalent toxicity probability for given efficacy probability and utility value.
        :param eff: efficacy probability
        :type eff: float
        :param util: utility value, defaults to zerp for neutral utility
        :type util: float
        :return: toxicity probability
        :rtype: float

        """

        p = self.p
        eff0 = self.minimum_tolerable_efficacy
        tox1 = self.maximum_tolerable_toxicity
        a = ((1-eff) / (1-eff0))
        return tox1 * ((1-util)**p -a**p)**(1/p)

    def plot_contours(self, use_ggplot=False, prob_eff=None, prob_tox=None, n=1000,
                      util_lower=-0.8, util_upper=0.8, util_delta=0.2, title='EffTox utility contours',
                      custom_points_label = 'priors'):
        """
        :param use_ggplot: True to use ggplot, False to use matplotlib
        :type use_ggplot: bool
        :param prob_eff: optional
        :type prob_eff: list
        :param prob_tox:
        :type prob_tox: list
        :param n: number of points per line
        :type n: int
        :param util_lower: lowest utility value to plot contour for
        :type util_lower: float
        :param util_upper: lowest utility value to plot contour for
        :type util_upper: float
        :param util_delta: plot contours for each increment in utility
        :type util_delta: float
        :param title: chart title
        :type title: str
        :param custom_points_label: label for points provided via prob_eff and prob_tox
        :type custom_points_label: str

        :return: plot of efficacy-toxicity contours

        """

        eff_vals = np.linspace(0, 1, n)
        util_vals = np.linspace(util_lower, util_upper, ((util_upper-util_lower) / util_delta) + 1)

        if use_ggplot:
            raise NotImplementedError()
        else:
            import matplotlib.pyplot as plt
            # Plot general contours
            for u in util_vals:
                tox_vals = [self.get_tox(eff=x, util=u) for x in eff_vals]
                plt.plot(eff_vals, tox_vals, '-', c='k', lw=0.5)

            # Add neutral utility contour
            tox_vals = [self.get_tox(eff=x, util=0) for x in eff_vals]
            plt.plot(eff_vals, tox_vals, '-', c='k', lw=2, label='neutral utility')

            # Add hinge points
            hinge_prob_eff, hinge_prob_tox = zip(*self.hinge_points)
            plt.plot(hinge_prob_eff, hinge_prob_tox, 'ro', ms=10, label='hinge points')

            # Add custom points
            if prob_eff is not None and prob_tox is not None:
                plt.plot(prob_eff, prob_tox, 'b^', ms=10, label=custom_points_label)

            # Plot size
            plt.ylim(0, 1)
            plt.xlim(0, 1)
            plt.xlabel('Prob(Efficacy)')
            plt.ylabel('Prob(Toxicity)')
            plt.title(title)
            plt.legend()

            # Return
            p = plt.gcf()
            phi = (np.sqrt(5)+1)/2.
            p.set_size_inches(12, 12/phi)


class InverseQuadraticCurve:
    """ Fit an indifference contour of the type, y = a + b/x + c/x^2 where y = Prob(Tox) and x = Prob(Eff).

    The official EffTox software has used L^p Norms in the trade-off analysis since approximately version 2.
    This is the current method as at Aug 2014.

    The official EffTox software used inverse quadratics in the trade-off analysis from inception until
    approximately version 2. The method was ditched in favour of using L^p norms.
    This is the current method as at Aug 2014.

    For more information, consult the original EffTox paper (2004), the Cook update (2006) and the Bayesian book.

    """

    def __init__(self, points):
        """
        Params:
        Points, list of points in (prob_eff, prob_tox) tuple pairs.
        """

        x = np.array([z for z,_ in points])
        y = np.array([z for _,z in points])
        z = 1/x
        import statsmodels.api as sm
        lm = sm.OLS(y, np.column_stack((np.ones_like(z), z, z**2))).fit()
        a, b, c = lm.params
        f = lambda x: a + b/x + c/x**2
        # Check f is not a terrible fit
        if sum(np.abs(f(x) - y)) > 0.00001:
            ValueError('%s do not fit an ABC curve well' % points)
        self.f = f
        self.a, self.b, self.c = a, b, c

    def __call__(self, prob_eff, prob_tox):
        x = prob_eff
        y = prob_tox
        if 0 < x < 1 and 0 < y < 1:
            gradient = 1.0 * y / (x-1)
            def intersection_expression(x, m, f):
                return m*(x-1) - f(x)
            x_00 = brentq(intersection_expression, 0.0001, 1, args=(gradient, self.f))
            y_00 = self.f(x_00)
            d1 = np.sqrt((x_00-1)**2 + y_00**2)
            d2 = np.sqrt((x-1)**2 + y**2)

            return d1 / d2 - 1
        else:
            return np.nan

    def solve(self, prob_eff=None, prob_tox=None, delta=0):
        """ Specify exactly one of prob_eff or prob_tox and this will return the other, for given delta"""
        # TODO
        raise NotImplementedError()

    def plot_contours(self, use_ggplot=False, prior_eff_probs=None, prior_tox_probs=None, n=1000,
                      util_lower=-0.8, util_upper=0.8, util_delta=0.2, title='EffTox utility contours'):
        """
        :param use_ggplot: True to use ggplot, False to use matplotlib
        :type use_ggplot: bool
        :param prior_eff_probs: optional
        :type prior_eff_probs: list
        :param prior_tox_probs:
        :type prior_tox_probs: list
        :param n: number of points per line
        :type n: int
        :param util_lower: lowest utility value to plot contour for
        :type util_lower: float
        :param util_upper: lowest utility value to plot contour for
        :type util_upper: float
        :param util_delta: plot contours for each increment in utility
        :type util_delta: float

        :return: plot of efficacy-toxicity contours

        """

        raise NotImplementedError()

# I used to call the InverseQuadraticCurve an ABC_Curve because it uses three parameters, a, b and c.
# Similarly, I used to call the LpNormCurve a HingedCurve because it uses a hinge point.
# Mask those for backwards compatability in my code.
HingedCurve = LpNormCurve
ABC_Curve = InverseQuadraticCurve


class EffTox(EfficacyToxicityDoseFindingTrial):
    """ This is an object-oriented implementation of Thall & Cook's EffTox trial design.

    See Thall, P.F. & Cook, J.D. (2004) - Dose-Finding Based on Efficacy-Toxicity Trade-Offs

    e.g. general usage
    (for now, parameter means and standard deviations were fetched from MD Anderson's EffTox software. TODO)
    >>> real_doses = [7.5, 15, 30, 45]
    >>> tox_cutoff = 0.40
    >>> eff_cutoff = 0.45
    >>> tox_certainty = 0.05
    >>> eff_certainty = 0.05
    >>> mu_t_mean, mu_t_sd = -5.4317, 2.7643
    >>> beta_t_mean, beta_t_sd = 3.1761, 2.7703
    >>> mu_e_mean, mu_e_sd = -0.8442, 1.9786
    >>> beta_e_1_mean, beta_e_1_sd = 1.9857, 1.9820
    >>> beta_e_2_mean, beta_e_2_sd = 0, 0.2
    >>> psi_mean, psi_sd = 0, 1
    >>> from scipy.stats import norm
    >>> theta_priors = [
    ...                   norm(loc=mu_t_mean, scale=mu_t_sd),
    ...                   norm(loc=beta_t_mean, scale=beta_t_sd),
    ...                   norm(loc=mu_e_mean, scale=mu_e_sd),
    ...                   norm(loc=beta_e_1_mean, scale=beta_e_1_sd),
    ...                   norm(loc=beta_e_2_mean, scale=beta_e_2_sd),
    ...                   norm(loc=psi_mean, scale=psi_sd),
    ...                 ]
    >>> hinge_points = [(0.4, 0), (1, 0.7), (0.5, 0.4)]
    >>> metric = LpNormCurve(hinge_points[0][0], hinge_points[1][1], hinge_points[2][0], hinge_points[2][1])
    >>> trial = EffTox(real_doses, theta_priors, tox_cutoff, eff_cutoff, tox_certainty, eff_certainty, metric,
    ...                max_size=30, first_dose=3)
    >>> trial.next_dose()
    3
    >>> trial.update([(3, 0, 1), (3, 1, 1), (3, 0, 0)])
    4
    >>> trial.has_more()
    True
    >>> trial.size(), trial.max_size()
    (3, 30)

    """

    def __init__(self, real_doses, theta_priors, tox_cutoff, eff_cutoff,
                 tox_certainty, eff_certainty, metric, max_size, first_dose=1,
                 avoid_skipping_untried_escalation=True, avoid_skipping_untried_deescalation=True,
                 num_integral_steps=10**5):
        """

        Params:
        :param real_doses: list of actual doses. E.g. for 10mg and 25mg, use [10, 25].
        :type real_doses: list
        :param theta_priors: list of prior distributions corresponding to mu_T, beta_T, mu_E, beta1_E, beta2_E, psi
                                respectively. Each prior object should support obj.ppf(x) and obj.pdf(x)
        :type theta_priors: list
        :param tox_cutoff: the maximum acceptable probability of toxicity
        :type tox_cutoff: float
        :param eff_cutoff: the minimium acceptable probability of efficacy
        :type eff_cutoff: float
        :param tox_certainty: the posterior certainty required that toxicity is less than cutoff
        :type tox_certainty: float
        :param eff_certainty: the posterior certainty required that efficacy is greater than than cutoff
        :type eff_certainty: float
        :param metric: instance of LpNormCurve or InverseQuadraticCurve, used to calculate utility
                        of efficacy/toxicity probability pairs.
        :type metric: LpNormCurve
        :param max_size: maximum number of patients to use
        :type max_size: int
        :param first_dose: starting dose level, 1-based. I.e. intcpt=3 means the middle dose of 5.
        :type first_dose: int
        :param avoid_skipping_untried_escalation: True to avoid skipping untried doses in escalation
        :type avoid_skipping_untried_escalation: bool
        :param avoid_skipping_untried_deescalation: True to avoid skipping untried doses in de-escalation
        :type avoid_skipping_untried_deescalation: bool
        :param num_integral_steps: number of points to use in Monte Carlo integration.
        :type num_integral_steps: int

        Note: dose_allocation_mode has been suppressed. Remove once I know it is not needed. KB
        # Instances have a dose_allocation_mode property that is set according to this schedule:
        # 0, when no dose has been chosen
        # 1, when optimal dose is selected from non-trivial admissable set (this is normal operation)
        # 2, when next untried dose is selected to avoid skipping doses
        # 2.5, when dose is maintained because ideal dose would require skipping and intervening dose is inadmissable
        # 3, when admissable set is empty so lowest untried dose above starting dose that is probably tolerable is
        #     selected
        # 4, when admissable set is empty and there is no untested dose above first dose to try
        # 5, when admissable set is empty and all doses were probably too toxic
        # 6, when admissable set is not-empty but trial stops because 1) ideal dose requires skipping; 2) next
        #     -best dose is inadmissable; and 3) current dose is inadmissable. I question the validity of this
        #     scenario because avoiding stopping was possible. However, I am trying to faithfully reproduce
        #     the MD Anderson software so this scenario is programmed in. That may change. If you want to avoid
        #     this outcome, allow dose skipping.

        """

        EfficacyToxicityDoseFindingTrial.__init__(self, first_dose, len(real_doses), max_size)

        if len(theta_priors) != 6:
            raise ValueError('theta_priors should have 6 items.')

        self.real_doses = real_doses
        self._scaled_doses = np.log(real_doses) - np.mean(np.log(real_doses))
        self.priors = theta_priors
        self.tox_cutoff = tox_cutoff
        self.eff_cutoff = eff_cutoff
        self.tox_certainty = tox_certainty
        self.eff_certainty = eff_certainty
        self.metric = metric
        self.avoid_skipping_untried_escalation = avoid_skipping_untried_escalation
        self.avoid_skipping_untried_deescalation = avoid_skipping_untried_deescalation
        self.num_integral_steps = num_integral_steps

        # Reset
        self.reset()

    def _update_integrals(self, n=None):
        """ Method to recalculate integrals, thus updating probabilties of eff and tox, utilities, and
            admissable set.
        """
        if n is None:
            n = self.num_integral_steps
        cases = list(zip(self._doses, self._toxicities, self._efficacies))
        post_probs, _pds = efftox_get_posterior_probs(cases, self.priors, self._scaled_doses, self.tox_cutoff,
                                                     self.eff_cutoff, n)
        prob_tox, prob_eff, prob_acc_tox, prob_acc_eff = zip(*post_probs)
        admissable = np.array([(x >= self.tox_certainty and y >= self.eff_certainty) # Probably acceptable tox & eff
                               or (i==self.maximum_dose_given() and x >= self.tox_certainty) # lowest untried dose above
                                                                                             # starting dose and
                                                                                             # probably acceptable tox
                               for i, (x, y) in enumerate(zip(prob_acc_tox, prob_acc_eff))])
        admissable_set = [i+1 for i, x in enumerate(admissable) if x]
        # Beware: I normally use (tox, eff) pairs but the metric expects (eff, tox) pairs, driven
        # by the equation form that Thall & Cook chose.
        utility = np.array([self.metric(x[0], x[1]) for x in zip(prob_eff, prob_tox)])
        self.prob_tox = prob_tox
        self.prob_eff = prob_eff
        self.prob_acc_tox = prob_acc_tox
        self.prob_acc_eff = prob_acc_eff
        self._admissable_set = admissable_set
        self.utility = utility
        self.pds = _pds

    def _EfficacyToxicityDoseFindingTrial__calculate_next_dose(self, n=None):
        if n is None:
            n = self.num_integral_steps
        self._update_integrals(n)
        if self.treated_at_dose(self.first_dose()) > 0:
            # First dose has been tried so modelling may commence
            max_dose_given = self.maximum_dose_given()
            min_dose_given = self.minimum_dose_given()
            for i in np.argsort(-self.utility):  # dose-indices from highest to lowest utility
                dose_level = i+1
                if dose_level in self.admissable_set():
                    if self.avoid_skipping_untried_escalation and max_dose_given and dose_level - max_dose_given > 1:
                        pass  # No skipping
                    elif self.avoid_skipping_untried_deescalation and min_dose_given and min_dose_given - dose_level > 1:
                        pass  # No skipping
                    else:
                        self._status = 1
                        self._next_dose = dose_level
                        break
            else:
                # No dose can be selected
                self._next_dose = -1
                self._status = -1
        else:
            # First dose not given yet, so keep recommending that, like EffTox software does
            self._next_dose = self.first_dose()
            if self.size() > 0:
                self._status = -10
            else:
                self._status = 0

        return self._next_dose

    def _EfficacyToxicityDoseFindingTrial__reset(self):
        """ Opportunity to run implementation-specific reset operations. """
        self.prob_tox = []
        self.prob_eff = []
        self.prob_acc_tox = []
        self.prob_acc_eff = []
        self._admissable_set = []
        self.utility = []

    def has_more(self):
        return EfficacyToxicityDoseFindingTrial.has_more(self)

    def tabulate(self):
        # import pandas as pd
        # tab_data = OrderedDict()
        # treated_at_dose = [self.treated_at_dose(d) for d in self.dose_levels()]
        # eff_at_dose = [self.efficacies_at_dose(d) for d in self.dose_levels()]
        # tox_at_dose = [self.toxicities_at_dose(d) for d in self.dose_levels()]
        # tab_data['Dose'] = self.dose_levels()
        # tab_data['N'] = treated_at_dose
        # tab_data['Efficacies'] = eff_at_dose
        # tab_data['Toxicities'] = tox_at_dose
        # df = pd.DataFrame(tab_data)
        # df['EffRate'] = np.where(df.N > 0, df.Efficacies / df.N, np.nan)
        # df['ToxRate'] = np.where(df.N > 0, df.Toxicities / df.N, np.nan)

        df = EfficacyToxicityDoseFindingTrial.tabulate(self)

        df['P(Eff)'] = self.prob_eff
        df['P(Tox)'] = self.prob_tox
        df['P(AccEff)'] = self.prob_acc_eff
        df['P(AccTox)'] = self.prob_acc_tox
        df['Admissible'] = self.dose_admissability()
        df['Utility'] = self.utility

        return df

    def posterior_params(self, n=None):
        """ Get posterior parameter estimates """
        if n is None:
            n = self.num_integral_steps
        cases = list(zip(self._doses, self._toxicities, self._efficacies))
        post_params, pds = efftox_get_posterior_params(cases, self.priors, self._scaled_doses, n)
        return post_params

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

        admiss, u, u_star, obd, u_cushion = solve_metrizable_efftox_scenario(prob_tox, prob_eff, self.metric,
                                                                             self.tox_cutoff, self.eff_cutoff)
        return obd

    def scaled_doses(self):
        return self._scaled_doses

    def _post_density_plot(self, func=None, x_name='', plot_title='', include_doses=None, boot_samps=1000):

        from ggplot import aes, ggplot, geom_density, ggtitle
        import pandas as pd

        if include_doses is None:
            include_doses = range(1, self.num_doses + 1)

        def my_func(x, samp):
            tox_probs = _pi_T(x, mu=samp[:, 0], beta=samp[:, 1])
            eff_probs = _pi_E(x, mu=samp[:, 2], beta1=samp[:, 3], beta2=samp[:, 4])
            u = self.metric(eff_probs, tox_probs)
            return u
        if func is None:
            func = my_func

        x_boot = []
        dose_indices = []
        samp = self.pds._samp
        p = self.pds._probs
        p /= p.sum()
        for i, x in enumerate(self.scaled_doses()):
            dose_index = i+1
            if dose_index in include_doses:
                x = func(x, samp)
                x_boot.extend(np.random.choice(x, size=boot_samps, replace=True, p=p))
                dose_indices.extend(np.repeat(dose_index, boot_samps))
        df = pd.DataFrame({x_name: x_boot, 'Dose': dose_indices})
        return ggplot(aes(x=x_name, fill='Dose'), data=df) + geom_density(alpha=0.6) + ggtitle(plot_title)

    def plot_posterior_tox_prob_density(self, include_doses=None, boot_samps=1000):
        """ Plot the posterior densities of the dose probabilities of toxicity.

        .. note:: this method uses ggplot so that must be available on your system.
        Why ggplot and not matplotlib? Because I also use R so I prefer the commonality of ggplot.
        The length of time this method takes will be linked to number of points in last call to update().
        It is relatively slow so don't go nuts.

        :param include_doses: optional, list of dose levels to include, e.g. [1,2]. Default is None for all doses.
        :type include_doses: list
        :return: ggplot device
        :rtype: ggplot

        """

        def get_prob_tox(x, samp):
            tox_probs = _pi_T(x, mu=samp[:, 0], beta=samp[:, 1])
            return tox_probs

        return self._post_density_plot(func=get_prob_tox, x_name='Prob(Toxicity)',
                                       plot_title='Posterior densities of Prob(Toxicity)',
                                       include_doses=include_doses, boot_samps=boot_samps)

    def plot_posterior_eff_prob_density(self, include_doses=None, boot_samps=1000):
        """ Plot the posterior densities of the dose probabilities of efficacy.

        .. note:: this method uses ggplot so that must be available on your system.
        Why ggplot and not matplotlib? Because I also use R so I prefer the commonality of ggplot.
        The length of time this method takes will be linked to number of points in last call to update().
        It is relatively slow so don't go nuts.

        :param include_doses: optional, list of dose levels to include, e.g. [1,2]. Default is None for all doses.
        :type include_doses: list
        :return: ggplot device
        :rtype: ggplot

        """

        def get_prob_eff(x, samp):
            eff_probs = _pi_E(x, mu=samp[:, 2], beta1=samp[:, 3], beta2=samp[:, 4])
            return eff_probs

        return self._post_density_plot(func=get_prob_eff, x_name='Prob(Efficacy)',
                                       plot_title='Posterior densities of Prob(Efficacy)',
                                       include_doses=include_doses, boot_samps=boot_samps)

    def plot_posterior_utility_density(self, include_doses=None, boot_samps=1000):
        """ Plot the posterior densities of the dose utilities.

        .. note:: this method uses ggplot so that must be available on your system.
        Why ggplot and not matplotlib? Because I also use R so I prefer the commonality of ggplot.
        The length of time this method takes will be linked to number of points in last call to update().
        It is relatively slow so don't go nuts.

        :param include_doses: optional, list of dose levels to include, e.g. [1,2]. Default is None for all doses.
        :type include_doses: list
        :return: ggplot device
        :rtype: ggplot

        """

        def get_utility(x, samp):
            tox_probs = _pi_T(x, mu=samp[:, 0], beta=samp[:, 1])
            eff_probs = _pi_E(x, mu=samp[:, 2], beta1=samp[:, 3], beta2=samp[:, 4])
            u = self.metric(eff_probs, tox_probs)
            return u

        return self._post_density_plot(func=get_utility, x_name='Utility', plot_title='Posterior densities of Utility',
                                       include_doses=include_doses, boot_samps=boot_samps)

    def prob_superior_utility(self, dl1, dl2):
        """ Returns the probability that the utility of dose-level 1 (dl1) exceeds that of dose-level 2 (dl2)

        :param dl1: 1-based dose-level of dose 1
        :type dl1: int
        :param dl2: 1-based dose-level of dose 2
        :type dl2: int
        :return: probability that utility of dose-level 1 exceeds that of dose-level 2
        :rtype: float

        """

        if dl1 == dl2:
            return 0

        samp = self.pds._samp
        p = self.pds._probs
        p /= p.sum()

        x1 = self.scaled_doses()[dl1-1]
        x1_tox_probs = _pi_T(x1, mu=samp[:, 0], beta=samp[:, 1])
        x1_eff_probs = _pi_E(x1, mu=samp[:, 2], beta1=samp[:, 3], beta2=samp[:, 4])
        u1 = self.metric(x1_eff_probs, x1_tox_probs)

        x2 = self.scaled_doses()[dl2-1]
        x2_tox_probs = _pi_T(x2, mu=samp[:, 0], beta=samp[:, 1])
        x2_eff_probs = _pi_E(x2, mu=samp[:, 2], beta1=samp[:, 3], beta2=samp[:, 4])
        u2 = self.metric(x2_eff_probs, x2_tox_probs)

        return np.sum(p * (u1 > u2))

    def utility_superiority_matrix(self):
        superiority_mat = np.zeros((4, 4))
        superiority_mat[:] = np.nan
        for i in range(1, self.num_doses+1):
            for j in range(i+1, self.num_doses+1):
                p = self.prob_superior_utility(i, j)
                superiority_mat[i-1, j-1] = p
                superiority_mat[j-1, i-1] = 1-p
        return superiority_mat


def solve_metrizable_efftox_scenario(prob_tox, prob_eff, metric, tox_cutoff, eff_cutoff):
    """ Solve a metrizable efficacy-toxicity dose-finding scenario.


    Metrizable means that the priority of doses can be calculated using a metric.
    A dose is conformative if it has probability of toxicity less than some cutoff; and
    probability of efficacy greater than some cutoff.
    The OBD is the dose with the highest utility in the conformative set. The OBD does not
    necessarily have a positive utility.

    This function returns, as a 5-tuple, (an array of bools representing whether each dose is conformative, the array
    of utlities, the utility of the optimal dose, the 1-based OBD level, and the utility distance from the OBD to the
    next most preferable dose in the conformative set where there are several conformative doses)

    :param prob_tox: Probabilities of toxicity at each dose
    :type prob_tox: iterable
    :param prob_eff: Probabilities of efficacy at each dose
    :type prob_eff: iterable
    :param metric: Metric to score
    :type metric: class like clintrials.dosefinding.efftox.LpNormCurve or func(prob_eff, prob_tox) returning float
    :param tox_cutoff: maximum acceptable toxicity probability
    :type tox_cutoff: float
    :param eff_cutoff: minimum acceptable efficacy probability
    :type eff_cutoff: float

    """
    if len(prob_tox) != len(prob_eff):
        raise Exception('prob_tox and prob_eff should be lists or tuples of the same length.')

    t = prob_tox
    r = prob_eff
    # Probabilities of 0.0 and 1.0 in the prob_tox and eff vectors cause problems when calculating utilities.
    # Being pragmatic, the easiest way to deal with them is to swap them for some number that is
    # nearly 0.0 or 1.0
    t = np.where(t <= 0, 0.001, t)
    t = np.where(t >= 1, 0.999, t)
    r = np.where(r <= 0, 0.001, r)
    r = np.where(r >= 1, 0.999, r)

    conform = np.array([(eff >= eff_cutoff) and (tox <= tox_cutoff) for eff, tox in zip(r, t)])
    util = np.array([metric(eff, tox) for eff, tox in zip(r, t)])
    conform_util = np.where(conform, util, -np.inf)

    if np.all(np.isnan(util)):
        logging.warn('All NaN util encountered in solve_metrizable_efftox_scenario')
        return conform, util, np.nan, -1, np.nan
    elif np.all(np.isnan(conform_util)):
        logging.warn('All NaN conform_util encountered in solve_metrizable_efftox_scenario')
        return conform, util, np.nan, -1, np.nan
    else:
        if sum(conform) >= 2:
            obd = np.nanargmax(conform_util)+1
            u2, u1 = np.sort(conform_util)[-2:]
            u_cushion = u1 - u2
            return conform, util, u1, obd, u_cushion
        elif sum(conform) >= 1:
            obd = np.nanargmax(conform_util)+1
            u1 = np.nanmax(conform_util)
            return conform, util, u1, obd, np.nan

    # Default:
    return conform, util, np.nan, -1, np.nan


def get_obd(tox_curve, eff_curve, metric, tox_cutoff, eff_cutoff):
    X = solve_metrizable_efftox_scenario(tox_curve, eff_curve, metric,
                                         tox_cutoff, eff_cutoff)
    conform, util, u_star, obd, u_cushion = X
    return obd


def get_conformative_doses(tox_curve, eff_curve, metric, tox_cutoff, eff_cutoff):
    X = solve_metrizable_efftox_scenario(tox_curve, eff_curve, metric,
                                         tox_cutoff, eff_cutoff)
    conform, util, u_star, obd, u_cushion = X
    return [int(x) for x in conform]


def get_util(tox_curve, eff_curve, metric, tox_cutoff, eff_cutoff):
    X = solve_metrizable_efftox_scenario(tox_curve, eff_curve, metric,
                                         tox_cutoff, eff_cutoff)
    conform, util, u_star, obd, u_cushion = X
    return np.round(util, 2)


def classify_problem(delta, prob_tox, prob_eff, metric, tox_cutoff, eff_cutoff, text_label=True):
    X = solve_metrizable_efftox_scenario(prob_tox, prob_eff, metric, tox_cutoff, eff_cutoff)
    conform, util, u_star, obd, u_cushion = X
    dose_within_delta = np.array([u >= (1-delta)*u_star for u in util])
    if obd == -1:
        if text_label:
            return 'Stop'
        else:
            return 1
    elif sum(dose_within_delta) == 1:
        if text_label:
            return 'Optimal'
        else:
            return 2
    else:
        if text_label:
            return 'Desirable'
        else:
            return 3


def get_problem_class(delta, tox_curve, eff_curve, metric, tox_cutoff, eff_cutoff):
    return classify_problem(delta, tox_curve, eff_curve, metric, tox_cutoff, eff_cutoff)


def classify_tox_class(prob_tox, tox_cutoff, text_label=True):
    prob_tox = np.array(prob_tox)
    if sum(prob_tox < tox_cutoff) == len(prob_tox):
        if text_label:
            return 'Tolerable'
        else:
            return 1
    elif sum(prob_tox > tox_cutoff) == len(prob_tox):
        if text_label:
            return 'Toxic'
        else:
            return 2
    else:
        if text_label:
            return 'Mixed'
        else:
            return 3


def get_tox_class(tox_curve, tox_cutoff):
    prob_tox = tox_curve
    return classify_tox_class(prob_tox, tox_cutoff)


def classify_eff_class(prob_eff, eff_cutoff, text_label=True):
    prob_eff = np.array(prob_eff)
    max_eff = np.max(prob_eff)
    if np.all([prob_eff[i] > prob_eff[i-1] for i in range(1, len(prob_eff))]):
        if text_label:
            return 'Monotonic'
        else:
            return 1
    elif sum(prob_eff == max_eff) == 1:
        if text_label:
            return 'Unimodal'
        else:
            return 2
    elif sum(prob_eff == max_eff) > 1:
        if text_label:
            return 'Plateau'
        else:
            return 3
    else:
        if text_label:
            return 'Weird'
        else:
            return 4


def get_eff_class(eff_curve, eff_cutoff):
    prob_eff = eff_curve
    return classify_eff_class(prob_eff, eff_cutoff)


def efftox_dtp_detail(trial):
    """ Performs the EffTox-specific extra reporting when calculating DTPs
    :param trial: instance of EffTox
    :return: OrderedDict

    """

    to_return = OrderedDict()

    # Utility
    to_return['Utility'] = iterable_to_json(trial.utility)
    for i, dl in enumerate(trial.dose_levels()):
        to_return['Utility{}'.format(dl)] = trial.utility[i]

    # Prob(Eff)
    to_return['ProbEff'] = iterable_to_json(trial.prob_eff)
    for i, dl in enumerate(trial.dose_levels()):
        to_return['ProbEff{}'.format(dl)] = trial.prob_eff[i]

    # Prob(Acceptable Eff)
    to_return['ProbAccEff'] = iterable_to_json(trial.prob_acc_eff)
    for i, dl in enumerate(trial.dose_levels()):
        to_return['ProbAccEff{}'.format(dl)] = trial.prob_acc_eff[i]

    # Prob(Tox)
    to_return['ProbTox'] = iterable_to_json(trial.prob_tox)
    for i, dl in enumerate(trial.dose_levels()):
        to_return['ProbTox{}'.format(dl)] = trial.prob_tox[i]

    # Prob(Acceptable Eff)
    to_return['ProbAccTox'] = iterable_to_json(trial.prob_acc_tox)
    for i, dl in enumerate(trial.dose_levels()):
        to_return['ProbAccTox{}'.format(dl)] = trial.prob_acc_tox[i]

    # What is the probability that the utility of the top dose exceeds that of the next best dose?
    # I.e. how confident are we that OBD really is the shizzle?
    # u1_dose_index, u2_dose_index = np.argsort(-trial.utility)[:2]
    sup_mat = trial.utility_superiority_matrix()
    to_return['SuperiorityMatrix'] = [iterable_to_json(x) for x in sup_mat]

    obd = trial.next_dose()
    if obd > 0:
        min_sup = np.nanmin(sup_mat[obd-1])
    else:
        min_sup = np.nan
    to_return['MinProbSuperiority'] = atomic_to_json(min_sup)

    return to_return