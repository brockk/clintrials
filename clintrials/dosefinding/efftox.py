__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'

""" An implementation of Thall & Cook's EffTox design for dose-finding in clinical trials.

See:
Thall, P.F. and Cook, J.D. (2004). Dose-Finding Based on Efficacy-Toxicity Trade-Offs, Biometrics, 60: 684-693.
Cook, J.D. Efficacy-Toxicity trade-offs based on L^p norms, Technical Report UTMDABTR-003-06, April 2006
Berry, Carlin, Lee and Mueller. Bayesian Adaptive Methods for Clinical Trials, Chapman & Hall / CRC Press

"""

import numpy as np
from scipy.optimize import brentq

from clintrials.common import inverse_logit
from clintrials.dosefinding import EfficacyToxicityDoseFindingTrial
from clintrials.util import correlated_binary_outcomes_from_uniforms


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
        _cases = zip(scaled_doses_given, tox_events, eff_events)
    else:
        _cases = []

    # The ranges of integration must be specified. In truth, the integration range is (-Infinity, Infinity)
    # for each variable. In practice, though, integrating to infinity is problematic, especially in
    # 6 dimensions. The limits of integration should capture all probability density, but not be too
    # generous, e.g. -1000 to 1000 would be stupid because the density at most points would be practically zero.
    # I use percentage points of the various prior distributions. The risk is that if the prior
    # does not cover the posterior range well, it will not estimate it well. This needs attention. TODO
    epsilon = 0.0001
    limits = [(dist.ppf(epsilon), dist.ppf(1-epsilon)) for dist in priors]
    samp = np.column_stack([np.random.uniform(*limit_pair, size=n) for limit_pair in limits])

    lik_integrand = lambda x: _L_n(_cases, x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]) \
                              * priors[0].pdf(x[:, 0]) * priors[1].pdf(x[:, 1]) * priors[2].pdf(x[:, 2]) \
                              * priors[3].pdf(x[:, 3]) * priors[4].pdf(x[:, 4]) * priors[5].pdf(x[:, 5])
    lik = lik_integrand(samp)
    scale = lik.mean()

    probs = []
    for x in scaled_doses:
        tox_probs = _pi_T(x, mu=samp[:,0], beta=samp[:,1])
        eff_probs = _pi_E(x, mu=samp[:,2], beta1=samp[:,3], beta2=samp[:,4])
        probs.append((
            np.mean(tox_probs * lik / scale),
            np.mean(eff_probs * lik / scale),
            np.mean((tox_probs < tox_cutoff) * lik / scale),
            np.mean((eff_probs > eff_cutoff) * lik / scale)
        ))

    return probs
    
    
def efftox_get_posterior_params(cases, priors, scaled_doses, tox_cutoff, eff_cutoff, n=10**5):
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
    epsilon = 0.0001
    limits = [(dist.ppf(epsilon), dist.ppf(1-epsilon)) for dist in priors]
    samp = np.column_stack([np.random.uniform(*limit_pair, size=n) for limit_pair in limits])

    lik_integrand = lambda x: _L_n(_cases, x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]) \
                              * priors[0].pdf(x[:, 0]) * priors[1].pdf(x[:, 1]) * priors[2].pdf(x[:, 2]) \
                              * priors[3].pdf(x[:, 3]) * priors[4].pdf(x[:, 4]) * priors[5].pdf(x[:, 5])
    lik = lik_integrand(samp)
    scale = lik.mean()

    params = []
    params.append(
    	(
        	np.mean(samp[:, 0] * lik / scale),
	        np.mean(samp[:, 1] * lik / scale),
	        np.mean(samp[:, 2] * lik / scale),
	        np.mean(samp[:, 3] * lik / scale),
	        np.mean(samp[:, 4] * lik / scale),
	        np.mean(samp[:, 5] * lik / scale),
    	)
    )

    return params


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

    def __call__(self, prob_eff, prob_tox):
        x = prob_eff
        y = prob_tox
        if 0 < x < 1 and 0< y < 1:
            a = ((1 - x) / (1 - self.minimum_tolerable_efficacy))
            b = y / self.maximum_tolerable_toxicity
            r_to_the_p = a**self.p + b**self.p
            return 1 - r_to_the_p ** (1./self.p)
        else:
            return np.nan

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

# I used to call the InverseQuadraticCurve an ABC_Curve because it uses three parameters, a, b and c.
# Similarly, I used to call the LpNormCurve a HingedCurve because it uses a hinge point.
# Mask those for backwards compatability in my code.
HingedCurve = LpNormCurve
ABC_Curve = InverseQuadraticCurve


class EffTox(EfficacyToxicityDoseFindingTrial):
    """ This is an object-oriented attempt at Thall & Cook's EffTox trial design.

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
                 avoid_skipping_untried_escalation=True, avoid_skipping_untried_deescalation=True):
        """

        Params:
        real_doses, list of actual doses. E.g. for 10mg and 25mg, use [10, 25].
        theta_priors, list of prior distributions corresponding to mu_T, beta_T, mu_E, beta1_E, beta2_E, psi
                        respectively. Each prior object should support obj.ppf(x) and obj.pdf(x)
        tox_cutoff, the maximum acceptable probability of toxicity
        eff_cutoff, the minimium acceptable probability of efficacy
        tox_certainty, the posterior certainty required that toxicity is less than cutoff
        eff_certainty, the posterior certainty required that efficacy is greater than than cutoff
        metric, instance of LpNormCurve or InverseQuadraticCurve, used to calculate utility
                of efficacy/toxicity probability pairs.
        max_size, maximum number of patients to use
        first_dose, starting dose level, 1-based. I.e. intcpt=3 means the middle dose of 5.
        avoid_skipping_untried_escalation, True to avoid skipping untried doses in escalation
        avoid_skipping_untried_deescalation, True to avoid skipping untried doses in de-escalation

        Instances have a dose_allocation_mode property that is set according to this schedule:
        0, when no dose has been chosen
        1, when optimal dose is selected from non-trivial admissable set (this is normal operation)
        2, when next untried dose is selected to avoid skipping doses in escalation
        3, when admissable set is empty so lowest untried dose above starting dose
            that is probably tolerable is selected
        4, when admissable set is empty and there is no untested dose above first dose to try
        5, when admissable set is empty and all doses were probably too toxic

        """

        EfficacyToxicityDoseFindingTrial.__init__(self, first_dose, len(real_doses), max_size)

        if len(theta_priors) != 6:
            raise ValueError('theta_priors should have 6 items.')

        self._scaled_doses = np.log(real_doses) - np.mean(np.log(real_doses))
        self.priors = theta_priors
        self.tox_cutoff = tox_cutoff
        self.eff_cutoff = eff_cutoff
        self.tox_certainty = tox_certainty
        self.eff_certainty = eff_certainty
        self.metric = metric
        self.avoid_skipping_untried_escalation = avoid_skipping_untried_escalation
        self.avoid_skipping_untried_deescalation = avoid_skipping_untried_deescalation

        # Reset
        self.prob_tox = []
        self.prob_eff = []
        self.prob_acc_tox = []
        self.prob_acc_eff = []
        self.utility = []
        self.dose_allocation_mode = 0
        # Estimate integrals
        _ = self._update_integrals()

    def _update_integrals(self, n=10**5):
        """ Method to recalculate integrals, thus updating probabilties of eff and tox, utilities, and
            admissable set.
        """
        cases = zip(self._doses, self._toxicities, self._efficacies)
        post_probs = efftox_get_posterior_probs(cases, self.priors, self._scaled_doses, self.tox_cutoff,
                                                self.eff_cutoff, n)
        prob_tox, prob_eff, prob_acc_tox, prob_acc_eff = zip(*post_probs)
        admissable = np.array([x >= self.tox_certainty and y >= self.eff_certainty
                               for x, y in zip(prob_acc_tox, prob_acc_eff)])
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

    def _EfficacyToxicityDoseFindingTrial__calculate_next_dose(self, n=10**5):
        self._update_integrals(n)
        dose_is_admissable = np.array([x in self._admissable_set for x in range(1, self.num_doses+1)])
        if sum(dose_is_admissable) > 0:
            # At least one dose is admissable. Select most desirable dose from admissable set based on utility
            masked_utility = np.where(dose_is_admissable, self.utility, np.nan)
            ideal_dose = np.nanargmax(masked_utility) + 1
            max_dose_given = self.maximum_dose_given()
            min_dose_given = self.minimum_dose_given()
            if self.avoid_skipping_untried_escalation and max_dose_given and ideal_dose - max_dose_given > 1:
                # Prevent skipping untried doses in escalation
                self._next_dose = max_dose_given + 1
                self._status = 1
                self.dose_allocation_mode = 2
            elif self.avoid_skipping_untried_deescalation and min_dose_given and min_dose_given - ideal_dose > 1:
                # Prevent skipping untried doses in de-escalation
                self._next_dose = min_dose_given - 1
                self._status = 1
                self.dose_allocation_mode = 2
            else:
                self._next_dose = ideal_dose
                self._status = 1
                self.dose_allocation_mode = 1
        else:
            # No dose is admissable.
            tolerable = [x >= self.tox_certainty for x in self.prob_acc_tox]
            if sum(tolerable) > 0:
                # Select lowest untried dose above starting dose that is probably tolerable
                for i, tol in enumerate(tolerable):
                    dose_level = i+1
                    if tol and dose_level > self._first_dose and self.treated_at_dose(dose_level) == 0:
                        self._next_dose = dose_level
                        self._status = 1
                        self.dose_allocation_mode = 3
                        break
                else:
                    # All doses have been tried or are below first dose. There is nothing left to try, so...
                    self._next_dose = -1
                    self._status = -2
                    self.dose_allocation_mode = 4
            else:
                # All doses are too toxic. There is nothing left to try, so...
                self._next_dose = -1
                self._status = -1
                self.dose_allocation_mode = 5

        return self._next_dose

    def _EfficacyToxicityDoseFindingTrial__reset(self):
        """ Opportunity to run implementation-specific reset operations. """
        self.prob_tox = []
        self.prob_eff = []
        self.prob_acc_tox = []
        self.prob_acc_eff = []
        self.utility = []
        self.dose_allocation_mode = 0
        # Estimate integrals
        _ = self._update_integrals()

    def _EfficacyToxicityDoseFindingTrial__process_cases(self, cases):
        """ Subclasses should override this method to perform an cases-specific processing. """
        return

    def has_more(self):
        return EfficacyToxicityDoseFindingTrial.has_more(self)
    
    def posterior_params(self, n=10**5):
    	""" Get posterior parameter estimates """
    	cases = zip(self._doses, self._toxicities, self._efficacies)
    	return efftox_get_posterior_params(cases, self.priors, self._scaled_doses, self.tox_cutoff,
                                           self.eff_cutoff, n)

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


def efftox_sim(n_patients, true_toxicities, true_efficacies, first_dose,
               real_doses, theta_priors, tox_cutoff, eff_cutoff, tox_certainty, eff_certainty,
               metric, tox_eff_odds_ratio=1.0, tolerances=None, cohort_size=1, n=10**5):
    """ Simulate EffTox trials.

    Params:
    n_patients, number of patients
    true_toxicities, list of the true toxicity rates. Obviously these are unknown in real-life but
                    we use them in simulations to test the algorithm. Should be same length as prior.
    true_efficacies, list of the true efficacy rates. These are unknown in real-life as well.
                            Should be same length as prior.
    first_dose, starting dose level, 1-based. I.e. first_dose=3 means the middle dose of 5.
    real_doses, list of possible doses in regular units, e.g. use [10, 25] for dose of 10mg and 25mg
    theta_priors, list of prior distributions corresponding to mu_T, beta_T, mu_E, beta1_E, beta2_E, psi
                    respectively. Each prior object should support obj.ppf(x) and obj.pdf(x)
    tox_cutoff, the maximum acceptable probability of toxicity
    eff_cutoff, the minimium acceptable probability of efficacy
    tox_certainty, the posterior certainty required that toxicity is less than cutoff
    eff_certainty, the posterior certainty required that efficacy is greater than than cutoff
    metric, instance of LpNormCurve or InverseQuadraticCurve or similar
    tox_eff_odds_ratio, odds ratio of toxicity and efficacy events. Use 1. for no association
    tolerances, optional n_patients*3 array of uniforms used to infer correlated toxicity and efficacy events
                        for patients. This array is passed to function that calculates correlated binary events from
                        uniform variables and marginal probabilities.
                        Leave None to get randomly sampled data.
                        This parameter is specifiable so that dose-finding methods can be compared on same 'patients'.
    cohort_size, to add several patients at once
    n, number of points to use in numerical integration. Higher is more accurate but slower.

    Returns: a 3-tuple of lists:
        - a list of doses selected
        - a list of dictionaries, containing patient-specific data on trial progress
        - a list of trial outcome ids, explained below.

    Trial outcomes:
        0: trial not started
        100: trial completed normally
        -2: trial abandoned early because no doses were admissable

    """

    if len(true_efficacies) != len(true_toxicities):
        raise ValueError('true_efficacies and true_toxicities should be same length.')
    if len(true_toxicities) != len(real_doses):
        raise ValueError('true_toxicities and real_doses should be same length.')
    if tolerances is not None:
        if tolerances.ndim != 2 or tolerances.shape[0] < n_patients:
            raise ValueError('tolerances should be an n_patients*3 array')
    else:
        tolerances = np.random.uniform(size=3*n_patients).reshape(n_patients, 3)

    # trial = EffTox(real_doses, theta_priors, tox_cutoff, eff_cutoff, tox_certainty, eff_certainty,
    #                metric, first_dose)
    trial = EffTox(real_doses=real_doses, theta_priors=theta_priors, tox_cutoff=tox_cutoff,
                   eff_cutoff=eff_cutoff, tox_certainty=tox_certainty, eff_certainty=eff_certainty,
                   metric=metric, max_size=n_patients, first_dose=first_dose)
    if first_dose:
        if first_dose != trial.next_dose():
            raise ValueError('first dose was ignored!')
    dose_level = trial.next_dose()
    i, trial_outcome = 0, 0
    doses_given, prob_tox_hat, prob_eff_hat, prob_acc_tox_hat, prob_acc_eff_hat, utility = [], [], [], [], [], []
    admissable_sets, dose_allocation_modes = [], []
    while i < n_patients:
        u = (true_toxicities[dose_level-1], true_efficacies[dose_level-1])
        events = correlated_binary_outcomes_from_uniforms(tolerances[i:i+cohort_size, ], u,
                                                          psi=tox_eff_odds_ratio).astype(int)
        new_cases = np.column_stack(([dose_level] * cohort_size, events))
        doses_given.extend([dose_level] * cohort_size)
        dose_level = trial.update(new_cases, n=n)
        prob_tox_hat.extend([trial.prob_tox] * cohort_size)
        prob_eff_hat.extend([trial.prob_eff] * cohort_size)
        prob_acc_tox_hat.extend([trial.prob_acc_tox] * cohort_size)
        prob_acc_eff_hat.extend([trial.prob_acc_eff] * cohort_size)
        utility.extend([trial.utility] * cohort_size)
        admissable_sets.extend([trial.admissable_set()] * cohort_size)
        dose_allocation_modes.extend([trial.dose_allocation_mode] * cohort_size)

        if dose_level < 0:
            #trial_outcome = -2
            trial_outcome = trial.status()
            break
        trial_outcome = 100
        i += cohort_size

    # Trial-level pertinent data
    dose_label = 'Dose'
    tox_label, eff_label= 'Tox', 'Eff'
    prob_tox_label, prob_eff_label= 'P(Tox)', 'P(Eff)'
    prob_acc_tox_label, prob_acc_eff_label= 'P(AccTox)', 'P(AccEff)'
    utility_label = 'Util'
    as_label, dam_label = 'Admiss', 'Mode'
    import pandas as pd
    df1 = pd.DataFrame({tox_label: trial.toxicities(), eff_label: trial.efficacies(), dose_label: doses_given,
                        as_label: admissable_sets, dam_label: dose_allocation_modes},
                       index=range(1, trial.size()+1))
    thing = {tox_label: trial.toxicities(), eff_label: trial.efficacies(), dose_label: doses_given,
             as_label: admissable_sets, dam_label: dose_allocation_modes}
    df2 = pd.DataFrame(prob_tox_hat, columns=[prob_tox_label + str(x) for x in range(1, len(real_doses)+1)],
                       index=range(1, trial.size()+1))
    for k, x in enumerate(np.array(prob_tox_hat).T):
        thing[prob_tox_label + str(k+1)] = x
    df3 = pd.DataFrame(prob_eff_hat, columns=[prob_eff_label + str(x) for x in range(1, len(real_doses)+1)],
                       index=range(1, trial.size()+1))
    for k, x in enumerate(np.array(prob_eff_hat).T):
        thing[prob_eff_label + str(k+1)] = x
    for k, x in enumerate(np.array(prob_acc_tox_hat).T):
        thing[prob_acc_tox_label + str(k+1)] = x
    for k, x in enumerate(np.array(prob_acc_eff_hat).T):
        thing[prob_acc_eff_label + str(k+1)] = x
    df4 = pd.DataFrame(utility, columns=[utility_label + str(x) for x in range(1, len(real_doses)+1)],
                       index=range(1, trial.size()+1))
    for k, x in enumerate(np.array(utility).T):
        thing[utility_label + str(k+1)] = x
    return trial.next_dose(), pd.concat([df1, df2, df3, df4], axis=1), trial_outcome


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
    conform = np.array([(eff > eff_cutoff) and (tox < tox_cutoff) for eff, tox in zip(r, t)])
    util = np.array([metric(eff, tox) for eff, tox in zip(r, t)])
    conform_util = np.where(conform, util, -np.inf)
    if sum(conform) >= 2:
        obd = np.nanargmax(conform_util)+1
        u2, u1 = np.sort(conform_util)[-2:]
        u_cushion = u1 - u2
        return conform, util, u1, obd, u_cushion
    elif sum(conform) >= 1:
        obd = np.nanargmax(conform_util)+1
        u1 = np.nanmax(conform_util)
        return conform, util, u1, obd, np.nan
    else:
        return conform, util, np.nan, -1, np.nan