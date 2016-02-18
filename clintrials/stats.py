__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'

""" Classes and methods to perform general useful statistical routines. """


from collections import OrderedDict
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, chi2, norm
from scipy.optimize import fsolve


def bootstrap(x):
    """ Bootstrap sample a list.

    :param x: sample observations
    :type x: list
    :return: bootstrap sample
    :rtype: numpy.array

    """

    return np.random.choice(x, size=len(x), replace=1)


def density(x, n_points=100, covariance_factor=0.25):
    """ Calculate and plot approximate densoty function from a sample.

    :param x: sample observations
    :type x: list
    :param n_points: number of points in density function to estimate
    :type n_points: int
    :param covariance_factor: covariance factor in scipy routine, see scipy.stats.gaussian_kde
    :type covariance_factor: float
    :return: None (yet)
    :rtype: None

    """

    d = gaussian_kde(x)
    xs = np.linspace(min(x), max(x), n_points)
    d.covariance_factor = lambda : covariance_factor
    d._compute_covariance()
    plt.plot(xs, d(xs))
    plt.show()


def beta_like_normal(mu, sigma):
    """ If X ~ N(mu, sigma^2), get alpha and beta s.t. Y ~ Beta(alpha, beta) has:
        E[X] = E[Y] & Var[X] = Var[Y]

        This is useful for quickly estimating the effective sample size of a normal prior,
        using the principle that the effective sample size of Beta(a, b) is a+b.

    :param mu: Mean of a normal r.v.
    :type mu: float
    :param sigma: Standard deviation of a normal r.v.
    :type sigma: float
    :return: (alpha, beta) pair of floats
    :rtype: tuple

    """

    alpha = (mu/sigma)**2 * (1-mu) - mu
    beta = ((1-mu)/mu) * alpha
    return alpha, beta


def or_test(a, b, c, d, ci_alpha=0.05):
    """ Calculate odds ratio and asymptotic confidence interval for events with counts a, b, c, and d.

    :param a: Number of observations with positive exposure (e.g. treated) and positive outcome (e.g cured)
    :type a: int
    :param b: Number of observations with positive exposure (e.g. treated) and negative outcome (e.g not cured)
    :type b: int
    :param c: Number of observations with negative exposure (e.g. not treated) and positive outcome (e.g cured)
    :type c: int
    :param d: Number of observations with negative exposure (e.g. not treated) and negative outcome (e.g not cured)
    :type d: int
    :param ci_alpha: significance for asymptotic confidence ionterval of odds-ratio
    :type ci_alpha: float
    :return: A dict object with all available statistics
    :rtype: collections.OrderedDict

    """

    abcd = [a, b, c, d]
    to_return = OrderedDict()
    to_return['ABCD'] = abcd

    if np.any(np.array(abcd) < 0):
        logging.error('Negative event count. Garbage!')
    elif np.any(np.array(abcd) == 0):
        logging.info('At least one event count was zero. Added one to all counts.')
        abcd = np.array(abcd) + 1
        a,b,c,d = abcd

    odds_ratio = 1. * (a * d) / (c * b)
    log_or_se = np.sqrt(sum(1. / np.array(abcd)))
    ci_scalars = norm.ppf([ci_alpha/2, 1-ci_alpha/2])
    or_ci = np.exp(np.log(odds_ratio) + ci_scalars * log_or_se)

    to_return['OR'] = odds_ratio
    to_return['Log(OR) SE'] = log_or_se
    to_return['OR CI'] = list(or_ci)

    to_return['Alpha'] = ci_alpha
    return to_return


def chi_squ_test(x, y, x_positive_value=None, y_positive_value=None, ci_alpha=0.05):
    """ Run a chi-squared test for association between x and y.

    :param x:
    :type x: list
    :param y:
    :type y: list
    :param x_positive_value: item in x corresponding to positive event, 1 by default
    :type x_positive_value: object
    :param y_positive_value: item in y corresponding to positive event, 1 by default
    :type y_positive_value: object
    :param ci_alpha: significance for asymptotic confidence ionterval of odds-ratio
    :type ci_alpha: float
    :return: A dict object with all available statistics
    :rtype: collections.OrderedDict

    """
    sum_oe = 0.0
    x_set = set(x)
    y_set = set(y)
    for x_case in x_set:
        x_matches = [z == x_case for z in x]
        for y_case in y_set:
            y_matches = [z == y_case for z in y]
            obs = sum(np.array(x_matches) & np.array(y_matches))
            exp = 1. * sum(x_matches) * sum(y_matches) / len(x)
            oe = (obs - exp)**2 / exp
            sum_oe += oe
    num_df = (len(x_set)-1) * (len(y_set)-1)
    p = 1-chi2.cdf(sum_oe, num_df)
    to_return = OrderedDict([('TestStatistic', sum_oe), ('p', p), ('Df', num_df)])

    if len(x_set) == 2 and len(y_set)==2:
        x = np.array(x)
        y = np.array(y)
        if not x_positive_value:
            x_positive_value=1
        if not y_positive_value:
            y_positive_value=1
        x_pos_val, y_pos_val = x_positive_value, y_positive_value
        a, b, c, d = (sum((x == x_pos_val) & (y == y_pos_val)), sum((x == x_pos_val) & (y != y_pos_val)),
                      sum((x != x_pos_val) & (y == y_pos_val)), sum((x != x_pos_val) & (y != y_pos_val)))
        to_return['Odds'] = or_test(a, b, c, d, ci_alpha=ci_alpha)
    else:
        # There's no reason why the OR logic could not be calculated for each combination pair
        # in x and y, but it's more work so leave it for now.
        pass

    return to_return


class ProbabilityDensitySample:

    def __init__(self, samp, func):
        self._samp = samp
        self._probs = func(samp)
        self._scale = self._probs.mean()

    def expectation(self, vector):
        return np.mean(vector * self._probs / self._scale)

    def variance(self, vector):
        exp = self.expectation(vector)
        exp2 = self.expectation(vector**2)
        return exp2 - exp**2

    def cdf(self, i, y):
        """ Get the cumulative density of the parameter in position i that is less than y. """
        return self.expectation(self._samp[:,i]<y)

    def quantile(self, i, p, start_value=0.1):
        """ Get the value of the parameter at position i for which p of the probability mass is in the left-tail. """
        return fsolve(lambda z: self.cdf(i, z) - p, start_value)[0]

    def cdf_vector(self, vector, y):
        """ Get the cumulative density of sample vector that is less than y. """
        return self.expectation(vector < y)

    def quantile_vector(self, vector, p, start_value=0.1):
        """ Get the value of a vector for which p of the probability mass is in the left-tail. """
        return fsolve(lambda z: self.cdf_vector(vector, z) - p, start_value)[0]