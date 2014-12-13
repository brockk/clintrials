__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'


""" Common, useful functions in the statistics and mathematics of clinical trials. """

import numpy as np


def inverse_logit(x):
    return 1/(1+np.exp(-x))


# Two-parameter link functions used in CRM-style designs
def empiric(x, a0=0, beta=0):
    return x ** np.exp(beta)


def inverse_empiric(x, a0=0, beta=0):
    return x ** np.exp(-beta)


def logistic(x, a0=0, beta=0):
    return 1 / (1 + np.exp(-a0 - np.exp(beta)*x))


def inverse_logistic(x, a0=0, beta=0):
    return (np.log(x/(1-x)) - a0) / np.exp(beta)


def hyperbolic_tan(x, a0=0, beta=0):
    return ((np.tanh(x) + 1) / 2) ** np.exp(beta)


def inverse_hyperbolic_tan(x, a0=0, beta=0):
    return np.arctanh(2*x**np.exp(-beta) - 1)
