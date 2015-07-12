__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'

# TODO

from nose.tools import with_setup
import numpy as np
from scipy.stats import norm

from clintrials.dosefinding.efftox import EffTox, LpNormCurve


# def setup_func():
#     pass
#
#
# def teardown_func():
#     pass


# @with_setup(setup_func, teardown_func)
def test_thall2014_efftox():

    # Recreate all params in a hypothetical path of the
    # trial described in Thall et al, 2014

    real_doses = [1, 2, 4, 6.6, 10]
    trial_size = 39
    first_dose = 1

    # Model params
    tox_cutoff = 0.3
    eff_cutoff = 0.5
    tox_certainty = 0.1
    eff_certainty = 0.1

    efftox_priors = [
        norm(loc=-7.9593, scale=3.5487),
        norm(loc=1.5482, scale=3.5018),
        norm(loc=0.7367, scale=2.5423),
        norm(loc=3.4181, scale=2.4406),
        norm(loc=0.0, scale=0.2),
        norm(loc=0.0, scale=1.0),
        ]

    hinge_points = [(0.5, 0), (1, 0.65), (0.7, 0.25)]
    metric = LpNormCurve(hinge_points[0][0], hinge_points[1][1], hinge_points[2][0], hinge_points[2][1])

    et = EffTox(real_doses, efftox_priors, tox_cutoff, eff_cutoff, tox_certainty, eff_certainty, metric, trial_size,
                first_dose)

    epsilon1 = 0.02
    epsilon2 = 0.04

    # Conduct a hypothetical trial and match the output to the official software

    # Cohort 1 - No responses or tox at dose 1
    cases = [(1, 0, 0), (1, 0, 0), (1, 0, 0)]
    next_dose = et.update(cases, n=10**6)
    assert next_dose == 2
    assert np.all(np.abs(et.prob_eff - np.array([0.04, 0.19, 0.57, 0.78, 0.87])) <= epsilon1)
    # 0.0411089,     0.192308,      0.57285,      0.78083,     0.865442
    # 0.0423735,     0.193038,     0.579136,     0.785387,     0.868304
    assert np.all(np.abs(et.prob_tox - np.array([0.01, 0.01, 0.02, 0.07, 0.13])) <= epsilon1)
    # 0.00891344,    0.0106823,    0.0253021,    0.0646338,     0.127027
    # 0.00819097,    0.0107802,    0.0290285,    0.0680472,     0.129004
    assert np.all(np.abs(et.utility - np.array([-0.93, -0.62, 0.11, 0.46, 0.53])) <= epsilon1)
    # -0.935017,    -0.630012,     0.108909,     0.455344,     0.521498
    # -0.936268,    -0.632437,     0.102517,     0.447415,     0.514455
    # -0.931457,    -0.634883,    0.0990699,     0.462706,     0.529655
    # -0.933477,    -0.634044,     0.102959,     0.456173,     0.528067
    # -0.929702,     -0.63275,     0.109402,     0.459879,     0.530763
    assert np.all(np.abs(et.prob_acc_eff - np.array([ 0.01, 0.12, 0.59, 0.82, 0.89])) <= epsilon1)
    # 0.00526376,     0.129856,     0.597466,     0.819071,     0.884521
    # 0.00552277,     0.132986,     0.588039,     0.810851,     0.889307
    # 0.00666152,     0.124646,     0.594309,     0.833948,     0.902025
    # 0.00678471,     0.127544,     0.588848,     0.823017,     0.890631
    # 0.00476136,     0.122447,     0.601623,     0.808115,     0.895993
    assert np.all(np.abs(et.prob_acc_tox - np.array([1.00, 0.99, 0.98, 0.93, 0.85])) <= epsilon1)
    # 0.995871,     0.995225,     0.978317,     0.929439,     0.853203
    # 0.996166,     0.995643,     0.973041,     0.930154,     0.851069

    # Cohort 2 - Singled response but no tox at dose 2
    cases = [(2, 0, 1), (2, 0, 0), (2, 0, 0)]
    next_dose = et.update(cases, n=10**6)
    assert next_dose == 3
    assert np.all(np.abs(et.prob_eff - np.array([0.05, 0.26, 0.72, 0.86, 0.91])) <= epsilon1)
    # 0.0509051,     0.269036,     0.722722,      0.86801,     0.914496]
    # 0.0510304,     0.265388,     0.722164,     0.864422,     0.914224
    # 0.0518509,     0.274869,      0.73628,     0.871351,     0.914808
    # 0.0538073,     0.276596,      0.73636,     0.876024,     0.921073
    assert np.all(np.abs(et.prob_tox - np.array([0.01, 0.01, 0.02, 0.06, 0.12])) <= epsilon1)
    # 0.0076164,   0.00930689,     0.022472,    0.0617848,     0.121213
    # 0.00648998,   0.00817005,    0.0202578,    0.0599878,     0.122688
    # 0.00725432,   0.00900773,    0.0197235,    0.0565496,     0.119785
    # 0.00733246,   0.00888673,    0.0206272,    0.0582661,     0.121853
    assert np.all(np.abs(et.utility - np.array([-0.91, -0.47, 0.42, 0.64, 0.64])) <= epsilon1)
    # -0.911645,      -0.4782,     0.407746,     0.636104,     0.636727
    # -0.909446,    -0.483551,     0.410265,     0.631728,     0.633869
    # -0.909128,    -0.466019,     0.439413,      0.65114,     0.639592
    # -0.905349,    -0.462358,      0.43809,      0.65783,     0.649114
    assert np.all(np.abs(et.prob_acc_eff - np.array([0.01, 0.13, 0.80, 0.91, 0.94])) <= epsilon1)
    # 0.00243223,     0.128105,     0.791986,     0.916741,     0.939606
    # 0.00307454,      0.13181,     0.789951,     0.902813,     0.938986
    # 0.00472987,     0.124916,     0.814938,     0.918152,     0.941961
    # 0.00350186,     0.138629,     0.821751,     0.926749,     0.948935
    assert np.all(np.abs(et.prob_acc_tox - np.array([1.00, 1.00, 0.98, 0.93, 0.86])) <= epsilon1)
    # 0.996671,     0.996452,     0.983693,     0.932407,     0.864749
    # 0.996752,     0.996427,     0.983369,     0.934628,     0.856087
    # 0.997235,       0.9968,     0.989923,     0.938403,     0.864412
    # 0.996549,     0.996075,     0.986417,     0.943325,     0.863644

    # Cohort 3 - Eff, Tox and a Both at dose level 3
    cases = [(3, 0, 1), (3, 1, 0), (3, 1, 1)]
    next_dose = et.update(cases, n=10**6)
    assert next_dose == 3
    assert np.all(np.abs(et.prob_eff - np.array([0.06, 0.24, 0.71, 0.89, 0.94])) <= epsilon1)
    # 0.0578593,     0.240559,     0.715176,     0.893277,     0.944129
    # 0.055577,     0.234091,      0.70965,     0.892396,     0.943455
    # 0.0549743,      0.22765,      0.69315,     0.887333,     0.940803
    assert np.all(np.abs(et.prob_tox - np.array([0.02, 0.06, 0.41, 0.77, 0.87])) <= epsilon1)
    # 0.0166051,     0.059379,       0.4087,     0.764791,     0.872101
    # 0.0163421,    0.0608998,     0.414188,     0.767363,     0.876163
    # 0.0162682,    0.0572041,     0.404346,     0.767712,     0.873865
    assert np.all(np.abs(et.utility - np.array([-0.92, -0.63, -0.23, -0.41, -0.47])) <= epsilon2)
    # -0.9131,    -0.618572,    -0.217798,    -0.404078,    -0.462769
    # -0.91722,    -0.634024,    -0.237611,    -0.409889,    -0.470461 # PROBLEM!
    # -0.918301,    -0.640853,    -0.255789,    -0.420991,    -0.472537
    assert np.all(np.abs(et.prob_acc_eff - np.array([0.01, 0.07, 0.84, 0.97, 0.98])) <= epsilon1)
    # 0.00431964,     0.079673,     0.854933,     0.972436,     0.982385
    # 0.0043408,    0.0706271,     0.854004,     0.966626,     0.980099
    # 0.00323427,     0.064967,     0.825731,     0.971935,      0.98238
    assert np.all(np.abs(et.prob_acc_tox - np.array([1.00, 0.98, 0.36, 0.08, 0.05])) <= epsilon2)
    # 0.995497,     0.976139,     0.373181,    0.0838482,     0.052491
    # 0.993254,     0.977806,     0.345383,     0.080502,    0.0480352 # PROBLEM!
    # 0.994217,     0.976738,     0.377723,    0.0835703,    0.0529918

    # assert next_dose == 3
    # assert np.all(np.abs(et.prob_eff - np.array([])) <= epsilon)
    # assert np.all(np.abs(et.prob_tox - np.array([])) <= epsilon)
    # assert np.all(np.abs(et.utility - np.array([])) <= epsilon)
    # assert np.all(np.abs(et.prob_acc_eff - np.array([])) <= epsilon)
    # assert np.all(np.abs(et.prob_acc_tox - np.array([])) <= epsilon)


def test_matchpoint_efftox():

    # Recreate all params in a hypothetical path of the
    # Matchpoint trial at CRCTU, University of Birmingham,
    # (publication in draft).

    mp_real_doses = [7.5, 15, 30, 45]
    mp_trial_size = 30
    mp_first_dose = 3

    mp_tox_cutoff = 0.40
    mp_eff_cutoff = 0.45

    mp_hinge_points = [(0.4, 0), (1, 0.7), (0.5, 0.4)]
    mp_metric = LpNormCurve(mp_hinge_points[0][0], mp_hinge_points[1][1], mp_hinge_points[2][0],
                            mp_hinge_points[2][1])

    mp_tox_certainty = 0.05
    mp_eff_certainty = 0.05

    mp_mu_t_mean, mp_mu_t_sd = -5.4317, 2.7643
    mp_beta_t_mean, mp_beta_t_sd = 3.1761, 2.7703
    mp_mu_e_mean, mp_mu_e_sd = -0.8442, 1.9786
    mp_beta_e_1_mean, mp_beta_e_1_sd = 1.9857, 1.9820
    mp_beta_e_2_mean, mp_beta_e_2_sd = 0, 0.2
    mp_psi_mean, mp_psi_sd = 0, 1
    mp_efftox_priors = [
              norm(loc=mp_mu_t_mean, scale=mp_mu_t_sd),
              norm(loc=mp_beta_t_mean, scale=mp_beta_t_sd),
              norm(loc=mp_mu_e_mean, scale=mp_mu_e_sd),
              norm(loc=mp_beta_e_1_mean, scale=mp_beta_e_1_sd),
              norm(loc=mp_beta_e_2_mean, scale=mp_beta_e_2_sd),
              norm(loc=mp_psi_mean, scale=mp_psi_sd),
              ]

    et = EffTox(mp_real_doses, mp_efftox_priors, mp_tox_cutoff, mp_eff_cutoff, mp_tox_certainty,
                mp_eff_certainty, mp_metric, mp_trial_size, mp_first_dose)

    epsilon1 = 0.02
    epsilon2 = 0.04

    # Cohort 1 - No responses and two toxes
    cases = [(3, 0, 0), (3, 1, 0), (3, 1, 0)]
    next_dose = et.update(cases, n=10**6)
    assert next_dose == 3
    assert np.all(np.abs(et.prob_eff - np.array([0.11, 0.10, 0.16, 0.25])) <= epsilon1)
    # 0.105846,    0.0997178,     0.158535,     0.243839
    # 0.113382,     0.105416,     0.159932,     0.245999]
    # 0.112774,     0.102533,     0.159932,     0.246534

    assert np.all(np.abs(et.prob_tox - np.array([0.06, 0.12, 0.52, 0.80])) <= epsilon1)
    # 0.0543127,     0.116322,     0.519326,     0.798698
    # 0.0573295,     0.123149,     0.524831,     0.804538
    # 0.0558935,     0.118977,     0.515849,     0.796918

    assert np.all(np.abs(et.utility - np.array([-0.48, -0.50, -0.57, -0.68])) <= epsilon1)
    # -0.49185,    -0.508098,    -0.572935,    -0.680738
    # -0.479495,    -0.499612,     -0.57442,     -0.68363
    # -0.480414,    -0.503796,    -0.568655,    -0.675756

    assert np.all(np.abs(et.prob_acc_eff - np.array([ 0.08, 0.04, 0.06, 0.20])) <= epsilon1)
    # 0.0737781,    0.0305425,    0.0602583,     0.198803
    # 0.0832185,    0.0443771,    0.0613655,     0.192502
    # 0.0817129,    0.0400428,    0.0657556,     0.203306

    assert np.all(np.abs(et.prob_acc_tox - np.array([0.95, 0.92, 0.33, 0.07])) <= epsilon2)
    # 0.952157,     0.920383,     0.332694,    0.0800697
    # 0.945771,     0.912801,     0.317847,    0.0621848 # Wide spread at dose 3
    # 0.951088,     0.912917,     0.346924,    0.0784506