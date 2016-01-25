__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'


""" Second attempt at classes and functions for the PePs2 trial.

    PePS2 studies the efficacy and toxicity of a drug in a population
    of performance status 2 lung cancer patients. Patient outcomes
    may plausibly be effected by whether or not they have been
    treated before, and the expression rate of PD-L1 in their cells.

    Our all-comers trial uses Brock et al's BeBOP design to incorporate
    this potentially predictive data to find the sub population(s)
    where the drug works and is tolerable.

    This script is version 2 and aims to de-couple the general design
    from the predictive variable chosen. It will not succeed completely,
    but will hopefully be a step in the right direction.

    """

from collections import OrderedDict
import datetime
import glob
from itertools import product
import json
import logging
import numpy as np
import pandas as pd

from clintrials.stats import chi_squ_test, or_test, ProbabilityDensitySample
from clintrials.util import correlated_binary_outcomes, atomic_to_json, iterable_to_json


# TODO