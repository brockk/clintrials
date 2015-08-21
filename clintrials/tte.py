__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'

""" Time-to-event trial designs """

from collections import OrderedDict
import numpy as np
from scipy.stats import expon, poisson, invgamma

from clintrials.util import atomic_to_json, iterable_to_json


class BayesianTimeToEvent():
    """ An object-oriented implementation of a simple adaptive Bayesian design for time-to-event endpoints using a
    model assuming exponentially distributed event times and inverse-gamma prior beliefs on median survival time.

    .. note:: See Thall, P.F., Wooten, L.H., & Tannir, N.M. (2005) - *Monitoring Event Times in Early Phase Clinical
                Trials: Some Practical Issues* for full information.

    This class satisfies the interface for a time-to-event trial in the clintrials package, i.e. it supports methods:

    - event_times()
    - recruitment_times()
    - update(cases)
    - test(time, kwargs)

    .. note:: the event times are time-deltas *relative to the recruitment times*. E.g. recruitment at t=1
                and event at t=2 means the event took place at absolute time t=3. Using deltas gets around
                the silly scenario where events might occur before recruitment.

    """

    def __init__(self, alpha_prior, beta_prior):
        """ Create an instance.

        :param alpha_prior: first parameter in beta distribution for prior beliefs on median time-to-event
        :type alpha_prior: float
        :param beta_prior: second parameter in beta distribution for prior beliefs on median time-to-event
        :type beta_prior: float

        """

        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self._times_to_event = []
        self._recruitment_times = []

    def event_times(self):
        """ Get list of the times at which events occurred.

        :return: list of event times in the order they were provided
        :rtype: list

        """

        return self._times_to_event

    def recruitment_times(self):
        """ Get list of the times at which patients were recruited.

        :return: list of recruitment times in the order they were provided
        :rtype: list

        """

        return self._recruitment_times

    def update(self, cases):
        """ Update the trial with new patient cases.

        :param cases: list of cases expressed as 2-tuples, (event_time, recruitment_time)
        :type cases: list
        :return: Nothing
        :rtype: None

        """

        for event_time, recruitment_time in cases:
            self._times_to_event.append(event_time)
            self._recruitment_times.append(recruitment_time)

    def test(self, time, cutoff, probability, less_than=True):
        """ Test posterior belief that median time-to-event parameter is less than or greater than some boundary value.

        :param time: test at this time
        :type time: float
        :param cutoff: test median time against this critical value
        :type cutoff: float
        :param probability: require at least this degree of posterior certainty to declare significance
        :type probability: float
        :param less_than: True, to test parameter is less than cut-off, a-posteriori. False to test greater than
        :type less_than: bool
        :return: JSON-able dict object reporting test output
        :rtype: dict

        """

        event_time = np.array(self._times_to_event)
        recruit_time = np.array(self._recruitment_times)

        # Filter to just patients who are registered by time
        registered_patients = recruit_time <= time
        has_failed = time - recruit_time[registered_patients] > event_time[registered_patients]
        survival_time = np.array([min(x, y) for (x, y) in
                                 zip(time - recruit_time[registered_patients], event_time[registered_patients])
                                      ])
        # Update posterior beliefs for mu_E
        alpha_post = self.alpha_prior + sum(has_failed)
        beta_post = self.beta_prior + np.log(2) * sum(survival_time)
        mu_post = beta_post / (alpha_post-1)

        # Run test:
        test_probability = invgamma.cdf(cutoff, a=alpha_post, scale=beta_post) if less_than \
            else 1 - invgamma.cdf(cutoff, a=alpha_post, scale=beta_post)
        stop_trial = test_probability > probability if less_than else test_probability < probability

        test_report = OrderedDict()
        test_report['Time'] = time
        test_report['Patients'] = sum(registered_patients)
        test_report['Events'] = sum(has_failed)
        test_report['TotalEventTime'] = sum(survival_time)
        test_report['AlphaPosterior'] = alpha_post
        test_report['BetaPosterior'] = beta_post
        test_report['MeanEventTimePosterior'] = mu_post
        test_report['MedianEventTimePosterior'] = mu_post * np.log(2)
        test_report['Cutoff'] = cutoff
        test_report['Certainty'] = probability
        test_report['Probability'] = test_probability
        test_report['LessThan'] = atomic_to_json(less_than)
        test_report['Stop'] = atomic_to_json(stop_trial)
        return test_report


def matrix_cohort_analysis(n_simulations, n_patients, true_median, alpha_prior, beta_prior,
                           lower_cutoff, upper_cutoff, interim_certainty, final_certainty,
                           interim_analysis_after_patients, interim_analysis_time_delta,
                           final_analysis_time_delta, recruitment_stream):
    """ Simulate TTE outcomes in the National Lung Matrix trial.

    .. note:: See Thall, P.F., Wooten, L.H., & Tannir, N.M. (2005) - *Monitoring Event Times in Early Phase Clinical
            Trials: Some Practical Issues* for full information.

    """

    reports = []
    for i in range(n_simulations):
        trial = BayesianTimeToEvent(alpha_prior, beta_prior)
        recruitment_stream.reset()
        # recruitment_times = np.arange(1, n_patients+1) / recruitment
        recruitment_times = np.array([recruitment_stream.next() for i in range(n_patients)])
        true_mean = true_median/np.log(2)
        event_times = expon(scale=true_mean).rvs(n_patients)  # Exponential survival times
        cases = [(x, y) for (x, y) in zip(event_times, recruitment_times)]
        trial.update(cases)
        interim_analysis_times = list(set([recruitment_times[x-1] + interim_analysis_time_delta
                                           for x in interim_analysis_after_patients if x < n_patients]))

        trial_report = OrderedDict()
        # Call parameters
        trial_report['MaxPatients'] = n_patients
        trial_report['TrueMedianEventTime'] = true_median
        trial_report['PriorAlpha'] = alpha_prior
        trial_report['PriorBeta'] = beta_prior
        trial_report['LowerCutoff'] = lower_cutoff
        trial_report['UpperCutoff'] = upper_cutoff
        trial_report['InterimCertainty'] = interim_certainty
        trial_report['FinalCertainty'] = final_certainty
        trial_report['InterimAnalysisAfterPatients'] = interim_analysis_after_patients
        trial_report['InterimAnalysisTimeDelta'] = interim_analysis_time_delta
        trial_report['FinalAnalysisTimeDelta'] = final_analysis_time_delta
        # trial_report['Recruitment'] = recruitment
        # Simulated patient outcomes
        trial_report['RecruitmentTimes'] = iterable_to_json(recruitment_times)
        trial_report['EventTimes'] = iterable_to_json(event_times)
        trial_report['InterimAnalyses'] = []
        # Interim analyses
        for time in interim_analysis_times:
            interim_outcome = trial.test(time, lower_cutoff, interim_certainty, less_than=True)
            trial_report['InterimAnalyses'].append(interim_outcome)
            stop_trial = interim_outcome['Stop']
            if stop_trial:
                trial_report['Decision'] = 'StopAtInterim'
                trial_report['FinalAnalysis'] = interim_outcome
                trial_report['FinalPatients'] = interim_outcome['Patients']
                trial_report['FinalEvents'] = interim_outcome['Events']
                trial_report['FinalTotalEventTime'] = interim_outcome['TotalEventTime']
                return trial_report
        # Final analysis
        final_analysis_time = max(recruitment_times) + final_analysis_time_delta
        final_outcome = trial.test(final_analysis_time, upper_cutoff, final_certainty, less_than=False)
        trial_report['FinalAnalysis'] = final_outcome
        stop_trial = final_outcome['Stop']
        decision = 'StopAtFinal' if stop_trial else 'GoAtFinal'
        trial_report['Decision'] = decision
        trial_report['FinalPatients'] = final_outcome['Patients']
        trial_report['FinalEvents'] = final_outcome['Events']
        trial_report['FinalTotalEventTime'] = final_outcome['TotalEventTime']
        reports.append(trial_report)

    if n_simulations == 1:
        return reports[0]
    else:
        return reports
