__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'


""" Common, useful functions in the statistics and mathematics of clinical trials. """

import numpy as np


def inverse_logit(x):
    """ Get the inverse logit function value:

    :math:`\\frac{e^x}{e^x+1}`,
    or equivalently, :math:`\\frac{1}{1 + e^{-x}}`

    :param x: x-variable
    :type x: float
    :return: Inverse logit function value.
    :rtype: float

    >>> inverse_logit(0)
    0.5

    """

    return 1/(1+np.exp(-x))


# Two-parameter link functions used in CRM-style designs
# They are written in pairs and all use the same call signature.
# They take their lead from the same in the dfcrm R-package.
def empiric(x, a0=None, beta=0):
    """ Get the empiric function value:

    :math:`x^{e^\\beta}`

    :param x: x-variable
    :type x: float
    :param a0: intercept parameter. This param is ignored here but exists to match similar call signatures.
    :type a0: float
    :param beta: slope parameter
    :type beta: float
    :return: Empiric function value
    :rtype: float

    >>> import math
    >>> empiric(0.5, beta=math.log(2))
    0.25

    """

    return x ** np.exp(beta)


def inverse_empiric(x, a0=0, beta=0):
    """ Get the inverse empiric function value:

    :math:`x^{e^{-\\beta}}`

    .. note:: this function is the inverse of :func:`clintrials.common.empiric`.

    :param x: x-variable
    :type x: float
    :param a0: intercept parameter. This param is ignored here but exists to match similar call signatures.
    :type a0: float
    :param beta: slope parameter
    :type beta: float
    :return: Inverse empiric function value
    :rtype: float

    >>> import math
    >>> inverse_empiric(0.25, beta=math.log(2))
    0.5

    """

    return x ** np.exp(-beta)


def logistic(x, a0=0, beta=0):
    """ Get the logistic function value:

    :math:`\\frac{1}{1 + e^{-a_0 - e^\\beta x}}`

    :param x: x-variable
    :type x: float
    :param a0: intercept parameter.
    :type a0: float
    :param beta: slope parameter
    :type beta: float
    :return: Logistic function value
    :rtype: float

    >>> logistic(0.25, -1, 1)
    0.42057106852688747

    """

    return 1 / (1 + np.exp(-a0 - np.exp(beta)*x))


def inverse_logistic(x, a0=0, beta=0):
    """ Get the inverse logistic function value:

    :math:`\\frac{\\log(\\frac{x}{1-x}) - a_0}{e^\\beta}`

    .. note:: this function is the inverse of :func:`clintrials.common.logistic`.

    :param x: x-variable
    :type x: float
    :param a0: intercept parameter.
    :type a0: float
    :param beta: slope parameter
    :type beta: float
    :return: Inverse logistic function value
    :rtype: float

    >>> round(inverse_logistic(0.42057106852688747, -1, 1), 2)
    0.25

    """

    return (np.log(x/(1-x)) - a0) / np.exp(beta)


def hyperbolic_tan(x, a0=0, beta=0):
    return ((np.tanh(x) + 1) / 2) ** np.exp(beta)


def inverse_hyperbolic_tan(x, a0=0, beta=0):
    return np.arctanh(2*x**np.exp(-beta) - 1)
