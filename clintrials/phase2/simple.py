__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'

""" Implementations of simple phase II clinical trial designs. Long, complicated designs belong in own modules. """

from collections import OrderedDict
from itertools import product
import numpy as np
from scipy.stats import beta, binom, chi2


def bayesian_2stage_dich_design(theta, p0, p1, N0, N1, p, q, prior_a=1, prior_b=1,
                                labels=['StopAtInterim', 'StopAtFinal', 'GoAtFinal']):
    """ Calculate the outcome probabilities for a two-stage Bayesian trial of a dichotomous variable.

    We test the hypotheses H0: theta<p0 vs H1: theta>p1, stopping at interim if Prob(theta < p0 | data) > p,
     stopping at final analysis if Prob(theta < p0 | data) > q, otherwise concluding that theta > p1.

    .. note:: this is Prof Lucinda Billingham's dichotomous design used in the National Lung Matrix trial.

    :param theta: the true efficacy
    :type theta: float
    :param p0: hypothesised lower bound probability
    :type p0: float
    :param p1: hypothesised upper bound probability
    :type p1: float
    :param N0: number of participants at interim stage
    :type N0: int
    :param N1: number of participants at final stage
    :type N1: int
    :param p: certainty needed to reject H0 at end of interim stage
    :type p: float
    :param q: certainty needed to accept H1 at end of final stage
    :type q: float
    :param prior_a: first parameter to Beta distribution to describe prior beliefs about theta
    :type prior_a: float
    :param prior_b: second parameter to Beta distribution to describe prior beliefs about theta
    :type prior_b: float
    :param labels: labels for the cases of stopping at interim, stopping at final, and approving at final analysis.
    :type labels: list
    :return: dict, mapping outcome label to probability
    :rtype: dict

    e.g.

    >>> res = bayesian_2stage_dich_design(0.35, 0.2, 0.4, 15, 30, 0.8, 0.6)
    >>> res == {'GoAtFinal': 0.21978663862560768, 'StopAtFinal': 0.76603457678233555,
    ...         'StopAtInterim': 0.014178784592056803}
    True

    """

    a, b = prior_a, prior_b
    n0, n1 = zip(*product(range(N0+1), range(N1-N0+1)))
    n0, n1 = np.array(n0), np.array(n1)
    lik0 = binom.pmf(n0, n=N0, p=theta)
    lik1 = binom.pmf(n1, n=N1-N0, p=theta)
    joint_lik = lik0 * lik1
    prob_lt_p0 = beta.cdf(p0, a+n0, b+N0-n0)
    prob_gt_p1 = 1 - beta.cdf(p1, a+n0+n1, b+N1-n0-n1)
    stop_0 = sum(joint_lik[prob_lt_p0 > p])
    go_1 = sum(joint_lik[~(prob_lt_p0 > p) & (prob_gt_p1 > q)])
    stop_1 = 1 - stop_0 - go_1
    return { labels[0]: stop_0, labels[1]: stop_1, labels[2]: go_1}


def bayesian_2stage_dich_design_df(theta, p0, p1, N0, N1, p, q, prior_a=1, prior_b=1,
                                labels=['StopAtInterim', 'StopAtFinal', 'GoAtFinal']):
    """ Calculate the outcome probabilities for a two-stage Bayesian trial of a dichotomous variable.

    We test the hypotheses H0: theta<p0 vs H1: theta>p1, stopping at interim if Prob(theta < p0 | data) > p,
     stopping at final analysis if Prob(theta < p0 | data) > q, otherwise concluding that theta > p1.

    .. note:: this is Prof Lucinda Billingham's dichotomous design used in the National Lung Matrix trial.

    :param theta: the true efficacy
    :type theta: float
    :param p0: hypothesised lower bound probability
    :type p0: float
    :param p1: hypothesised upper bound probability
    :type p1: float
    :param N0: number of participants at interim stage
    :type N0: int
    :param N1: number of participants at final stage
    :type N1: int
    :param p: certainty needed to reject H0 at end of interim stage
    :type p: float
    :param q: certainty needed to accept H1 at end of final stage
    :type q: float
    :param prior_a: first parameter to Beta distribution to describe prior beliefs about theta
    :type prior_a: float
    :param prior_b: second parameter to Beta distribution to describe prior beliefs about theta
    :type prior_b: float
    :param labels: labels for the cases of stopping at interim, stopping at final, and approving at final analysis.
    :type labels: list
    :return: dict, mapping outcome label to probability
    :rtype: dict

    See bayesian_2stage_dich_design

    """

    a, b = prior_a, prior_b
    n0, n1 = zip(*product(range(N0+1), range(N1-N0+1)))
    n0, n1 = np.array(n0), np.array(n1)
    lik0 = binom.pmf(n0, n=N0, p=theta)
    lik1 = binom.pmf(n1, n=N1-N0, p=theta)
    joint_lik = lik0 * lik1
    prob_lt_p0 = beta.cdf(p0, a+n0, b+N0-n0)
    prob_gt_p1 = 1 - beta.cdf(p1, a+n0+n1, b+N1-n0-n1)
    stop_0 = prob_lt_p0 > p
    go_1 = ~stop_0 & (prob_gt_p1 > q)
    stop_1 = ~(stop_0 | go_1)

    dat = OrderedDict()
    dat['Successes0'] = n0
    dat['Pr(theta<p0)'] = prob_lt_p0
    dat['Successes1'] = n1
    dat['TotalSuccesses'] = n0+n1
    dat['Pr(theta>p1)'] = prob_gt_p1
    dat['Lik'] = joint_lik
    dat[labels[0]] = stop_0
    dat[labels[1]] = stop_1
    dat[labels[2]] = go_1
    import pandas as pd
    return pd.DataFrame(dat)


def chisqu_two_arm_comparison(p0, p1, n, alpha):
    """ Test that p1 exceeds p0 with n patients per arm using the chi-squared distribution.

    :param p0: first proportion
    :type p0: float
    :param p1: second proportion
    :type p1: float
    :param n: n patients per arm
    :type n: int
    :param alpha: significance
    :type alpha: float
    :param to_pandas: True to get results as pandas.DataFrame else dict
    :type to_pandas: bool
    :return: tuple -- (probability of rejecting, probability of not-rejecting)

    E.g.

    >>> chisqu_two_arm_comparison(0.3, 0.5, 20, 0.05)
    (0.34534530091794574, 0.65465469908205098)

    """

    n0, n1 = zip(*list(product(range(n+1), range(n+1))))
    n0 = np.array(n0)
    n1 = np.array(n1)
    lik0 = binom.pmf(n0, n, p0)
    lik1 = binom.pmf(n1, n, p1)
    lik = lik0 * lik1
    observed = np.column_stack([n0, n-n0, n1, n-n1])
    success = n0 + n1
    fail = 2*n - n0 - n1
    expected = np.column_stack([success/2., fail/2., success/2., fail/2.])
    test_stat = ((observed-expected)**2 / expected).sum(axis=1)
    p = 1-chi2.cdf(test_stat, 1)
    reject = (p < alpha*2) & (n0<n1)
    data = np.column_stack([n0, n1, lik, test_stat, p, reject])
    return sum(data[data[:, 5] == True, 2]), sum(data[data[:, 5] == False, 2])

