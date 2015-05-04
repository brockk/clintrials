__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'

""" Classes and methods to perform general useful statistical routines. """


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


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