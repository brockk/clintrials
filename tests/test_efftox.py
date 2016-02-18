__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'

# TODO

# from nose.tools import with_setup
from collections import OrderedDict
import numpy as np
from scipy.stats import norm

from clintrials.dosefinding.efftox import EffTox, LpNormCurve


def assess_efftox_trial(et):
    to_return = OrderedDict()
    to_return['NextDose'] = et.next_dose()
    to_return['ProbEff'] = et.prob_eff
    to_return['ProbTox'] = et.prob_tox
    to_return['ProbAccEff'] = et.prob_acc_eff
    to_return['ProbAccTox'] = et.prob_acc_tox
    to_return['Utility'] = et.utility
    return to_return

def run_trial(trial, cases, summary_func, **kwargs):
    trial.reset()
    trial.update(cases, **kwargs)
    return summary_func(trial)


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

    epsilon1 = 0.05
    epsilon2 = 0.05

    # Conduct a hypothetical trial and match the output to the official software


    # Cohort 1 - No responses or tox at dose 1
    cases = [(1, 0, 0), (1, 0, 0), (1, 0, 0)]
    trial_outcomes = [run_trial(et, cases, assess_efftox_trial, n=10**6) for i in range(5)]

    assert np.all([o['NextDose'] == 2 for o in trial_outcomes])
    assert np.all(np.array([list(o['ProbEff']) for o in trial_outcomes]).mean(axis=0) - [0.04, 0.19, 0.57, 0.78, 0.87] < epsilon1)
    # 0.0411089,     0.192308,      0.57285,      0.78083,     0.865442
    # 0.0423735,     0.193038,     0.579136,     0.785387,     0.868304
    assert np.all(np.array([list(o['ProbTox']) for o in trial_outcomes]).mean(axis=0) - [0.01, 0.01, 0.02, 0.07, 0.13] < epsilon1)
    # 0.00891344,    0.0106823,    0.0253021,    0.0646338,     0.127027
    # 0.00819097,    0.0107802,    0.0290285,    0.0680472,     0.129004
    assert np.all(np.array([list(o['Utility']) for o in trial_outcomes]).mean(axis=0) - [-0.93, -0.62, 0.11, 0.46, 0.53] < epsilon1)
    # -0.935017,    -0.630012,     0.108909,     0.455344,     0.521498
    # -0.936268,    -0.632437,     0.102517,     0.447415,     0.514455
    # -0.931457,    -0.634883,    0.0990699,     0.462706,     0.529655
    # -0.933477,    -0.634044,     0.102959,     0.456173,     0.528067
    # -0.929702,     -0.63275,     0.109402,     0.459879,     0.530763
    assert np.all(np.array([list(o['ProbAccEff']) for o in trial_outcomes]).mean(axis=0) - [ 0.01, 0.12, 0.59, 0.82, 0.89] < epsilon1)
    # 0.00526376,     0.129856,     0.597466,     0.819071,     0.884521
    # 0.00552277,     0.132986,     0.588039,     0.810851,     0.889307
    # 0.00666152,     0.124646,     0.594309,     0.833948,     0.902025
    # 0.00678471,     0.127544,     0.588848,     0.823017,     0.890631
    # 0.00476136,     0.122447,     0.601623,     0.808115,     0.895993
    assert np.all(np.array([list(o['ProbAccTox']) for o in trial_outcomes]).mean(axis=0) - [1.00, 0.99, 0.98, 0.93, 0.85] < epsilon1)
    # 0.995871,     0.995225,     0.978317,     0.929439,     0.853203
    # 0.996166,     0.995643,     0.973041,     0.930154,     0.851069


    # Cohort 2 - Singled response but no tox at dose 2
    cases = cases + [(2, 0, 1), (2, 0, 0), (2, 0, 0)]
    trial_outcomes = [run_trial(et, cases, assess_efftox_trial, n=10**6) for i in range(5)]

    assert np.all([o['NextDose'] == 3 for o in trial_outcomes])
    assert np.all(np.array([list(o['ProbEff']) for o in trial_outcomes]).mean(axis=0) - [0.05, 0.26, 0.72, 0.86, 0.91] < epsilon1)
    # 0.0509051,     0.269036,     0.722722,      0.86801,     0.914496]
    # 0.0510304,     0.265388,     0.722164,     0.864422,     0.914224
    # 0.0518509,     0.274869,      0.73628,     0.871351,     0.914808
    # 0.0538073,     0.276596,      0.73636,     0.876024,     0.921073
    assert np.all(np.array([list(o['ProbTox']) for o in trial_outcomes]).mean(axis=0) - [0.01, 0.01, 0.02, 0.06, 0.12] < epsilon1)
    # 0.0076164,   0.00930689,     0.022472,    0.0617848,     0.121213
    # 0.00648998,   0.00817005,    0.0202578,    0.0599878,     0.122688
    # 0.00725432,   0.00900773,    0.0197235,    0.0565496,     0.119785
    # 0.00733246,   0.00888673,    0.0206272,    0.0582661,     0.121853
    assert np.all(np.array([list(o['Utility']) for o in trial_outcomes]).mean(axis=0) - [-0.91, -0.47, 0.42, 0.64, 0.64] < epsilon1)
    # -0.911645,      -0.4782,     0.407746,     0.636104,     0.636727
    # -0.909446,    -0.483551,     0.410265,     0.631728,     0.633869
    # -0.909128,    -0.466019,     0.439413,      0.65114,     0.639592
    # -0.905349,    -0.462358,      0.43809,      0.65783,     0.649114
    assert np.all(np.array([list(o['ProbAccEff']) for o in trial_outcomes]).mean(axis=0) - [0.01, 0.13, 0.80, 0.91, 0.94] < epsilon1)
    # 0.00243223,     0.128105,     0.791986,     0.916741,     0.939606
    # 0.00307454,      0.13181,     0.789951,     0.902813,     0.938986
    # 0.00472987,     0.124916,     0.814938,     0.918152,     0.941961
    # 0.00350186,     0.138629,     0.821751,     0.926749,     0.948935
    assert np.all(np.array([list(o['ProbAccTox']) for o in trial_outcomes]).mean(axis=0) - [1.00, 1.00, 0.98, 0.93, 0.86] < epsilon1)
    # 0.996671,     0.996452,     0.983693,     0.932407,     0.864749
    # 0.996752,     0.996427,     0.983369,     0.934628,     0.856087
    # 0.997235,       0.9968,     0.989923,     0.938403,     0.864412
    # 0.996549,     0.996075,     0.986417,     0.943325,     0.863644


    # Cohort 3 - Eff, Tox and a Both at dose level 3
    cases = cases + [(3, 0, 1), (3, 1, 0), (3, 1, 1)]
    trial_outcomes = [run_trial(et, cases, assess_efftox_trial, n=10**6) for i in range(10)]

    assert np.all([o['NextDose'] == 3 for o in trial_outcomes])
    assert np.all(np.array([list(o['ProbEff']) for o in trial_outcomes]).mean(axis=0) - [0.06, 0.24, 0.71, 0.89, 0.94] < epsilon1)
    # 0.0578593,     0.240559,     0.715176,     0.893277,     0.944129
    # 0.055577,     0.234091,      0.70965,     0.892396,     0.943455
    # 0.0549743,      0.22765,      0.69315,     0.887333,     0.940803
    assert np.all(np.array([list(o['ProbTox']) for o in trial_outcomes]).mean(axis=0) - [0.02, 0.06, 0.41, 0.77, 0.87] < epsilon1)
    # 0.0166051,     0.059379,       0.4087,     0.764791,     0.872101
    # 0.0163421,    0.0608998,     0.414188,     0.767363,     0.876163
    # 0.0162682,    0.0572041,     0.404346,     0.767712,     0.873865
    assert np.all(np.array([list(o['Utility']) for o in trial_outcomes]).mean(axis=0) - [-0.92, -0.63, -0.24, -0.41, -0.47] < epsilon2)
    # -0.9131,    -0.618572,    -0.217798,    -0.404078,    -0.462769
    # -0.91722,    -0.634024,    -0.237611,    -0.409889,    -0.470461
    # -0.918301,    -0.640853,    -0.255789,    -0.420991,    -0.472537
    assert np.all(np.array([list(o['ProbAccEff']) for o in trial_outcomes]).mean(axis=0) - [0.01, 0.07, 0.84, 0.97, 0.98] < epsilon1)
    # 0.00431964,     0.079673,     0.854933,     0.972436,     0.982385
    # 0.0043408,    0.0706271,     0.854004,     0.966626,     0.980099
    # 0.00323427,     0.064967,     0.825731,     0.971935,      0.98238
    assert np.all(np.array([list(o['ProbAccTox']) for o in trial_outcomes]).mean(axis=0) - [1.00, 0.98, 0.36, 0.08, 0.05] < epsilon2)
    # 0.995497,     0.976139,     0.373181,    0.0838482,     0.052491
    # 0.993254,     0.977806,     0.345383,     0.080502,    0.0480352 # PROBLEM!
    # 0.994217,     0.976738,     0.377723,    0.0835703,    0.0529918


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

    epsilon1 = 0.05
    epsilon2 = 0.05

    # Cohort 1 - No responses and two toxes.
    # This situation tests scenario where a combination of avoidance of
    # dose-skipping and sparse admissable set => recommended dose stays put
    cases = [(3, 0, 0), (3, 1, 0), (3, 1, 0)]
    trial_outcomes = [run_trial(et, cases, assess_efftox_trial, n=10**6) for i in range(10)]

    # assert np.all([o['NextDose'] == 3 for o in trial_outcomes]) # Ambivalent
    assert np.all(np.array([list(o['ProbEff']) for o in trial_outcomes]).mean(axis=0) - [0.11, 0.10, 0.16, 0.25] < epsilon1)
    # 0.105846,    0.0997178,     0.158535,     0.243839
    # 0.113382,     0.105416,     0.159932,     0.245999
    # 0.112774,     0.102533,     0.159932,     0.246534
    assert np.all(np.array([list(o['ProbTox']) for o in trial_outcomes]).mean(axis=0) - [0.06, 0.12, 0.52, 0.80] < epsilon1)
    # 0.0543127,     0.116322,     0.519326,     0.798698
    # 0.0573295,     0.123149,     0.524831,     0.804538
    # 0.0558935,     0.118977,     0.515849,     0.796918
    assert np.all(np.array([list(o['Utility']) for o in trial_outcomes]).mean(axis=0) - [-0.49, -0.50, -0.57, -0.67] < epsilon1)
    # -0.49185,    -0.508098,    -0.572935,    -0.680738
    # -0.479495,    -0.499612,     -0.57442,     -0.68363
    # -0.480414,    -0.503796,    -0.568655,    -0.675756
    assert np.all(np.array([list(o['ProbAccEff']) for o in trial_outcomes]).mean(axis=0) - [ 0.08, 0.04, 0.06, 0.20] < epsilon1)
    # 0.0737781,    0.0305425,    0.0602583,     0.198803
    # 0.0832185,    0.0443771,    0.0613655,     0.192502
    # 0.0817129,    0.0400428,    0.0657556,     0.203306
    assert np.all(np.array([list(o['ProbAccTox']) for o in trial_outcomes]).mean(axis=0) - [0.95, 0.92, 0.33, 0.07] < epsilon2)
    # 0.952157,     0.920383,     0.332694,    0.0800697
    # 0.945771,     0.912801,     0.317847,    0.0621848 # Wide spread at dose 3
    # 0.951088,     0.912917,     0.346924,    0.0784506



# @with_setup(setup_func, teardown_func)
def test_thall2014_efftox_v2():

    # Recreate all params in a hypothetical path of the
    # trial described in Thall et al, 2014, with some
    # weirder behaviour like using weird doses and
    # bizarre eff & tox behaviour

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

    epsilon1 = 0.05
    epsilon2 = 0.05

    # Conduct a hypothetical trial and match the output to the official software


    # Patient 1 takes dose 4, experiences tox only
    cases = [(4, 1, 0)]
    trial_outcomes = [run_trial(et, cases, assess_efftox_trial, n=10**6) for i in range(10)]

    assert np.all([o['NextDose'] == 1 for o in trial_outcomes])
    assert np.all(np.array([list(o['ProbEff']) for o in trial_outcomes]).mean(axis=0) - [0.16, 0.18, 0.26, 0.40, 0.51] < epsilon1)
    # 0.157708,     0.175569,     0.262729,     0.401978,      0.51943
    # 0.159857,      0.18366,     0.257962,     0.385366,     0.496521
    # 0.156691,     0.176197,     0.262477,      0.40087,     0.522515
    assert np.all(np.array([list(o['ProbTox']) for o in trial_outcomes]).mean(axis=0) - [0.07, 0.10, 0.26, 0.58, 0.79] < epsilon1)
    # 0.0654221,    0.0971754,     0.261445,     0.587468,     0.793057
    # 0.0646629,    0.0944231,     0.241762,     0.566136,     0.782698
    # 0.0749463,     0.108187,     0.269988,     0.593154,       0.8015
    assert np.all(np.array([list(o['Utility']) for o in trial_outcomes]).mean(axis=0) - [-0.78, -0.80, -0.88, -1.10, -1.18] < epsilon1)
    # -0.78459359, -0.79681393, -0.86699412, -1.09067852, -1.17696273
    # -0.79223068, -0.80613904, -0.88895611, -1.10721898, -1.18218725
    # -0.7915823 , -0.8057034 , -0.88837279, -1.11576831, -1.19192616
    assert np.all(np.array([list(o['ProbAccEff']) for o in trial_outcomes]).mean(axis=0) - [ 0.14, 0.13, 0.19, 0.37, 0.52] < epsilon2)
    # 0.133412,     0.125565,     0.193301,     0.376064,     0.532416
    # 0.142883,     0.147857,      0.18962,     0.359766,     0.511193
    # 0.132598,     0.130498,     0.191643,     0.372066,     0.535538
    assert np.all(np.array([list(o['ProbAccTox']) for o in trial_outcomes]).mean(axis=0) - [0.92, 0.88, 0.69, 0.26, 0.11] < epsilon1)
    # 0.92232,     0.887738,     0.666631,     0.249994,     0.100707
    # 0.924049,     0.896069,     0.704535,     0.279793,    0.0774165
    # 0.913237,     0.878992,     0.664061,     0.234436,    0.0972018


    # Patient 2 takes dose 2, experiences nothing
    cases = cases  + [(2, 0, 0)]
    trial_outcomes = [run_trial(et, cases, assess_efftox_trial, n=10**6) for i in range(10)]

    assert np.all([o['NextDose'] == 1 for o in trial_outcomes])
    assert np.all(np.array([list(o['ProbEff']) for o in trial_outcomes]).mean(axis=0) - [0.08, 0.10, 0.20, 0.37, 0.52] < epsilon1)
    # 0.0821915,     0.100276,     0.198857,     0.367706,     0.513249
    # 0.0838232,    0.0974985,     0.201392,      0.37594,     0.516932
    # 0.0834354,     0.100993,     0.204322,     0.377156,     0.524105
    assert np.all(np.array([list(o['ProbTox']) for o in trial_outcomes]).mean(axis=0) - [0.02, 0.05, 0.21, 0.56, 0.79] < epsilon1)
    # 0.0237103,    0.0482733,     0.199567,      0.55032,     0.784568
    # 0.0255792,     0.053126,     0.211518,     0.552868,     0.775685
    # 0.0258052,    0.0515479,     0.206791,     0.560613,     0.795126
    assert np.all(np.array([list(o['Utility']) for o in trial_outcomes]).mean(axis=0) - [-0.88, -0.88, -0.93, -1.13, -1.18] < epsilon1)
    # -0.87822996 -0.88044002 -0.93536759 -1.13446895 -1.18717248
    # -0.87524417 -0.87153713 -0.91525599 -1.12004315 -1.17972149
    # -0.88015356 -0.88096721 -0.93826181 -1.13637188 -1.18179096
    assert np.all(np.array([list(o['ProbAccEff']) for o in trial_outcomes]).mean(axis=0) - [ 0.05, 0.04, 0.11, 0.33, 0.52] < epsilon2)
    # 0.0523111,    0.0451266,     0.105403,     0.317057,      0.51977
    # 0.0444401,    0.0372249,     0.107815,     0.340655,     0.517309
    # 0.0568997,    0.0448017,     0.117971,     0.333277,     0.527102
    assert np.all(np.array([list(o['ProbAccTox']) for o in trial_outcomes]).mean(axis=0) - [0.97, 0.95, 0.74, 0.27, 0.10] < epsilon1)
    # 0.97697,     0.953224,     0.752848,     0.281933,    0.0961928
    # 0.971225,      0.94334,     0.737988,     0.280542,      0.11164
    # 0.976007,     0.943854,       0.7497,     0.260749,     0.096534


    # Patient 3 takes dose 1, experiences nothing
    cases = cases + [(1, 0, 0)]
    trial_outcomes = [run_trial(et, cases, assess_efftox_trial, n=10**6) for i in range(10)]

    assert np.all([o['NextDose'] in [3, 4] for o in trial_outcomes])
    assert np.all(np.array([list(o['ProbEff']) for o in trial_outcomes]).mean(axis=0) - [0.05, 0.07, 0.19, 0.38, 0.54] < epsilon1)
    # 0.0444924,     0.070795,     0.184645,     0.372773,     0.536132
    # 0.048781,    0.0715875,     0.184968,     0.380816,     0.542991
    # 0.0486916,    0.0742152,      0.18994,     0.379419,     0.543241
    assert np.all(np.array([list(o['ProbTox']) for o in trial_outcomes]).mean(axis=0) - [0.01, 0.04, 0.20, 0.56, 0.79] < epsilon1)
    # 0.013422,    0.0371188,     0.195556,      0.56251,     0.792571
    # 0.012601,    0.0407719,     0.207403,     0.555442,     0.781372
    # 0.0143285,    0.0359184,     0.190005,     0.557048,     0.784581
    assert np.all(np.array([list(o['Utility']) for o in trial_outcomes]).mean(axis=0) - [-0.93, -0.92, -0.95, -1.14, -1.16] < epsilon1)
    # -0.934429,    -0.921661,    -0.951229,     -1.15337,      -1.1814
    # -0.924449,    -0.926155,    -0.969584,     -1.12598,     -1.14996
    # -0.927573,    -0.912815,    -0.931684,     -1.13133,     -1.15445
    assert np.all(np.array([list(o['ProbAccEff']) for o in trial_outcomes]).mean(axis=0) - [ 0.02, 0.02, 0.09, 0.34, 0.55] < epsilon1)
    # 0.0146855,     0.016538,    0.0825426,     0.326103,     0.547288
    # 0.019816,     0.018858,    0.0906095,     0.347732,     0.559773
    #  0.0204063,    0.0163605,    0.0974005,     0.345784,     0.553082
    assert np.all(np.array([list(o['ProbAccTox']) for o in trial_outcomes]).mean(axis=0) - [0.98, 0.97, 0.75, 0.28, 0.11] < epsilon1)
    # 0.988605,     0.967294,     0.758932,     0.272984,    0.0985633
    # 0.991555,     0.961945,     0.732714,     0.290823,     0.113183
    # 0.986746,      0.97316,     0.762996,     0.282144,     0.115626


    # Patient 4 takes dose 5, experiences both
    cases = cases + [(5, 1, 1)]
    trial_outcomes = [run_trial(et, cases, assess_efftox_trial, n=10**6) for i in range(5)]

    assert np.all([o['NextDose'] == 3 for o in trial_outcomes])
    assert np.all(np.array([list(o['ProbEff']) for o in trial_outcomes]).mean(axis=0) - [0.03, 0.08, 0.26, 0.54, 0.75] < epsilon1)
    # 0.0326091,    0.0770342,     0.265108,     0.549073,     0.750367]
    # 0.0319256,    0.0774786,     0.262769,     0.542528,     0.746857
    # 0.0323994,     0.076549,     0.258008,     0.541642,     0.750589
    assert np.all(np.array([list(o['ProbTox']) for o in trial_outcomes]).mean(axis=0) - [0.01, 0.04, 0.22, 0.64, 0.89] < epsilon1)
    # 0.0102802,     0.035909,     0.214656,     0.636154,     0.886352]
    # 0.0102499,    0.0384096,     0.229782,     0.635306,     0.886731
    # 0.0108129,    0.0354585,     0.215542,     0.646399,     0.892663
    assert np.all(np.array([list(o['Utility']) for o in trial_outcomes]).mean(axis=0) - [-0.95, -0.91, -0.84, -0.92, -0.89] < epsilon1)
    # -0.952828,    -0.907157,    -0.820185,    -0.910981,    -0.888259]
    # -0.954143,    -0.910433,     -0.84906,    -0.922971,     -0.89608
    # -0.954161,    -0.907378,     -0.83587,    -0.942101,    -0.897581
    assert np.all(np.array([list(o['ProbAccEff']) for o in trial_outcomes]).mean(axis=0) - [ 0.01, 0.02, 0.16, 0.57, 0.84] < epsilon1)
    # 0.00952869,    0.0185933,     0.161803,     0.573807,     0.827742]
    # 0.00766963,    0.0132403,     0.164568,     0.565208,     0.829171
    # 0.00805502,    0.0175303,     0.154376,     0.564132,     0.846368
    assert np.all(np.array([list(o['ProbAccTox']) for o in trial_outcomes]).mean(axis=0) - [0.99, 0.97, 0.72, 0.16, 0.02] < epsilon1)
    # 0.994051,     0.974491,     0.733143,     0.150137,    0.0162891]
    # 0.992878,     0.968902,     0.700709,     0.166045,     0.015269
    # 0.99071,     0.970019,      0.73017,     0.153644,    0.0195214
