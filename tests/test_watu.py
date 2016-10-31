__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'

""" Tests of the clintrials.dosefindings.wagestait module. """

from nose.tools import with_setup
import numpy as np
from scipy.stats import norm

from clintrials.common import empiric, logistic, inverse_empiric, inverse_logistic
from clintrials.dosefinding.watu import WATU
from clintrials.dosefinding.efftox import LpNormCurve


def setup_func():
    pass


def teardown_func():
    pass


@with_setup(setup_func, teardown_func)
def test_watu_1():

    tox_prior = [0.01, 0.08, 0.15, 0.22, 0.29, 0.36]
    tox_cutoff = 0.33
    eff_cutoff = 0.05
    tox_target = 0.30

    skeletons = [
        [0.60, 0.50, 0.40, 0.30, 0.20, 0.10],
        [0.50, 0.60, 0.50, 0.40, 0.30, 0.20],
        [0.40, 0.50, 0.60, 0.50, 0.40, 0.30],
        [0.30, 0.40, 0.50, 0.60, 0.50, 0.40],
        [0.20, 0.30, 0.40, 0.50, 0.60, 0.50],
        [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        [0.20, 0.30, 0.40, 0.50, 0.60, 0.60],
        [0.30, 0.40, 0.50, 0.60, 0.60, 0.60],
        [0.40, 0.50, 0.60, 0.60, 0.60, 0.60],
        [0.50, 0.60, 0.60, 0.60, 0.60, 0.60],
        [0.60, 0.60, 0.60, 0.60, 0.60, 0.60],
    ]

    first_dose = 1
    trial_size = 64
    stage1_size = 16

    metric = LpNormCurve(0.05, 0.4, 0.25, 0.15)
    trial = WATU(skeletons, tox_prior, tox_target, tox_cutoff, eff_cutoff, metric, first_dose, trial_size, stage1_size)

    cases = [
        (1,1,0), (1,0,0), (1,0,0),
        (2,0,0), (2,0,0), (2,0,1),
        (3,1,1), (3,0,1),
    ]

    next_dose = trial.update(cases)
    assert next_dose == 2

    assert np.all(np.abs(trial.post_tox_probs - np.array([0.1376486, 0.3126617, 0.4095831, 0.4856057, 0.5506505,
                                                          0.6086650])) < 0.001)  #First one varies a bit more
    assert np.all(np.abs(trial.post_eff_probs - np.array([0.2479070, 0.3639813, 0.4615474, 0.5497718, 0.6321674,
                                                          0.7105235])) < 0.00001)
    assert np.all(np.abs(trial.w - np.array([0.01347890, 0.03951504, 0.12006585, 0.11798287, 0.11764227, 0.12346595,
                                      0.11764227, 0.11798287, 0.12006585, 0.07073296, 0.04142517])) < 0.00001)
    assert trial.most_likely_model_index == 5
    # All the same as Wages & Tait
    assert trial.admissable_set() == [1, 2, 3, 4, 5]  # More permissive than Wages & Tait
    assert np.all(np.abs(trial.prob_acc_tox() - np.array([0.9469, 0.5610, 0.2924, 0.1365, 0.0559, 0.0199]))
                  < 0.01)  # This is subject to random variation (estimation error) so varies a bit
    assert np.all(np.abs(trial.prob_acc_eff() - np.array([0.9497, 0.9928, 0.9991,  0.9999,  1.0000, 1.0000 ]))
                  < 0.01)  # This is subject to random variation (estimation error) so varies a bit
    assert trial.utility == []  # Empty as still in stage 1


def test_watu_2():

    tox_prior = [0.01, 0.08, 0.15, 0.22, 0.29, 0.36]
    tox_cutoff = 0.33
    eff_cutoff = 0.05
    tox_target = 0.30

    skeletons = [
        [0.60, 0.50, 0.40, 0.30, 0.20, 0.10],
        [0.50, 0.60, 0.50, 0.40, 0.30, 0.20],
        [0.40, 0.50, 0.60, 0.50, 0.40, 0.30],
        [0.30, 0.40, 0.50, 0.60, 0.50, 0.40],
        [0.20, 0.30, 0.40, 0.50, 0.60, 0.50],
        [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        [0.20, 0.30, 0.40, 0.50, 0.60, 0.60],
        [0.30, 0.40, 0.50, 0.60, 0.60, 0.60],
        [0.40, 0.50, 0.60, 0.60, 0.60, 0.60],
        [0.50, 0.60, 0.60, 0.60, 0.60, 0.60],
        [0.60, 0.60, 0.60, 0.60, 0.60, 0.60],
    ]

    first_dose = 1
    trial_size = 64
    stage1_size = 16

    metric = LpNormCurve(0.05, 0.4, 0.25, 0.15)
    trial = WATU(skeletons, tox_prior, tox_target, tox_cutoff, eff_cutoff, metric, first_dose, trial_size, stage1_size)

    cases = [
        (1,1,0), (1,0,0), (1,0,0),
        (2,0,0), (2,0,0), (2,0,1),
        (3,1,1), (3,0,1), (3,1,1),
        (2,0,0), (2,0,0), (2,1,1),
        (3,0,1), (3,0,0), (3,1,1),
        (4,1,1), (4,0,1), (4,0,1),
    ]

    next_dose = trial.update(cases)
    assert next_dose == 1
    assert np.all(trial.post_tox_probs - np.array([0.1292270, 0.3118713, 0.4124382, 0.4906020, 0.5569092, 0.6155877])
                  < 0.00001)
    assert np.all(trial.post_eff_probs - np.array([0.3999842, 0.4935573, 0.5830683, 0.6697644, 0.5830683, 0.4935573])
                  < 0.00001)
    assert np.all(trial.w - np.array([0.001653197, 0.006509789, 0.069328268, 0.156959090, 0.141296982, 0.144650706,
                                      0.141296982, 0.156959090, 0.117673776, 0.041764220, 0.021907900]) < 0.00001)
    assert trial.most_likely_model_index == 3
    assert trial.admissable_set() == [1, 2, 3, 4]  # More permissive than Wages & Tait
    assert np.all(np.abs(trial.prob_acc_tox() - np.array([0.9870, 0.5766, 0.2190, 0.0618,  0.0131,  0.0021]))
                  < 0.01)  # This is subject to random variation (estimation error) so varies a bit
    assert np.all(np.abs(trial.prob_acc_eff() - np.array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]))
                  < 0.01)  # This is subject to random variation (estimation error) so varies a bit

    assert np.all(np.abs(trial.utility - np.array([ 0.18320154, -0.11034328, -0.26984169, -0.39399425, -0.61068672,
                                                    -0.81190408])) < 0.00001)