__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'


""" Probability models for the BeBOP model used in PePS2 trial.

In PePS2, the patient vector is x and D is a list of x instances, where:

x[0] is efficacy event
x[1] is toxicity event
x[2] is pre-treated group membership dummy
x[3] is low PD-L1 group membership dummy
x[4] is middle PD-L1 group membership dummy

The parameter vector is theta:
theta[0] is efficacy model intercept
theta[1] is efficacy model pre-treated group coeff
theta[2] is efficacy model low PD-L1 group coeff
theta[3] is efficacy model middle PD-L1 group coeff
theta[4] is toxicity model intercept
theta[5] is association param

"""

import numpy


def pi_e(x, theta):
    z = theta[:,0] + theta[:, 1]*x[2] + theta[:, 2]*x[3] + theta[:, 3]*x[4]
    return 1/(1+numpy.exp(-z))


def pi_t(x, theta):
    z = theta[:,4]
    return 1 / (1+numpy.exp(-z))


def pi_ab(x, theta):
    b = x[0] # had efficacy
    a = x[1] # had_toxicity
    psi = theta[:, 5]
    pe = pi_e(x, theta)
    pt = pi_t(x, theta)
    joint_prob = pe**b * (1-pe)**(1-b) * pt**a * (1-pt)**(1-a)
    joint_prob = joint_prob + (-1)**(a+b) * pe * (1-pe) * pt * (1-pt) * (numpy.exp(psi) - 1) / (numpy.exp(psi) + 1)
    return joint_prob


