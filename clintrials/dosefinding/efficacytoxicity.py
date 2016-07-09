__author__ = 'brockk'


import abc
from collections import OrderedDict
from itertools import product, combinations_with_replacement
import numpy as np
import logging

from clintrials.util import (atomic_to_json, iterable_to_json,
                             correlated_binary_outcomes_from_uniforms, to_1d_list)
# from clintrials.simulation import filter_sims


# Joint Phase I/II, Assessing efficacy and toxicity
class EfficacyToxicityDoseFindingTrial(object):
    """ This is the base class for a dose-finding trial that jointly monitors toxicity and efficacy.

    The interface for such a class is:
    status()
    reset()
    number_of_doses()
    dose_levels()
    first_dose()
    size()
    max_size()
    doses()
    toxicities()
    efficacies()
    treated_at_dose(dose)
    toxicities_at_dose(dose)
    efficacies_at_dose(dose)
    maximum_dose_given()
    minimum_dose_given()
    tabulate()
    set_next_dose(dose)
    next_dose()
    update(cases)
    has_more()
    admissable_set()
    observed_toxicity_rates()
    observed_efficacy_rates()
    optimal_decision(prob_tox, prob_eff)

    Further internal interface is provided by:
    __reset()
    __calculate_next_dose() # Subclasses should override, set _status & _admissable_set, and return _next_dose.

    Class uses the internal variable _status to signify the current status of the trial. At the start of each
    trial, the status is 0, signifying that the trial has not started. It is proposed that trial statuses
    greater than 0 be used to signify that the trial is progressing in a positive way, and that trial statuses
    less than 0 be used to signify states where the trial has arrived at some negative scenario that dictates
    the trial to stop, e.g. all doses being considered too toxic.
    Suggested values for _status are:
    0, trial not started
    1, trial in progress
    100, trial is finished and reached a proper conclusion
    -1, all doses are too toxic
    -2: no doses are admissable
    -3: lowest dose is probably too toxic
    -4: optimal dose in probably not efficacious enough
    -10: design is in some inconsistent or errorsome state

    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, first_dose, num_doses, max_size):
        if first_dose > num_doses:
            raise ValueError('First dose must be no greater than number of doses.')

        self._first_dose = first_dose
        self.num_doses = num_doses
        self._max_size = max_size

        # Reset
        self._doses = []
        self._toxicities = []
        self._efficacies = []
        self._next_dose = self._first_dose
        self._status = 0
        self._admissable_set = []

    def status(self):
        return self._status

    def reset(self):
        self._doses = []
        self._toxicities = []
        self._efficacies = []
        self._next_dose = self._first_dose
        self._status = 0
        self.__reset()

    def number_of_doses(self):
        """ How many dose-levels are under investigation?"""
        return self.num_doses

    def dose_levels(self):
        """ Get list of dose levels, aka dose indices
        :return: list of dose indices
        """
        return range(1, self.num_doses+1)

    def first_dose(self):
        """
        Get the first dose
        :return: first dose
        """
        return self._first_dose

    def size(self):
        """ How many patients have been treated? """
        return len(self._doses)

    def max_size(self):
        """ Maximum number of trial patients. """
        return self._max_size

    def doses(self):
        return self._doses

    def toxicities(self):
        return self._toxicities

    def efficacies(self):
        return self._efficacies

    def treated_at_dose(self, dose):
        """ Number of patients treated at a dose level. """
        return sum(np.array(self._doses) == dose)

    def toxicities_at_dose(self, dose):
        """ Number of toxicities at (1-based) dose level. """
        return sum([t for d, t in zip(self.doses(), self.toxicities()) if d == dose])

    def efficacies_at_dose(self, dose):
        """ Number of toxicities at (1-based) dose level. """
        return sum([e for d, e in zip(self.doses(), self.efficacies()) if d == dose])

    def maximum_dose_given(self):
        if len(self._doses) > 0:
            return max(self._doses)
        else:
            return None

    def minimum_dose_given(self):
        if len(self._doses) > 0:
            return min(self._doses)
        else:
            return None

    def tabulate(self):
        import pandas as pd
        tab_data = OrderedDict()
        treated_at_dose = [self.treated_at_dose(d) for d in self.dose_levels()]
        eff_at_dose = [self.efficacies_at_dose(d) for d in self.dose_levels()]
        tox_at_dose = [self.toxicities_at_dose(d) for d in self.dose_levels()]
        tab_data['Dose'] = self.dose_levels()
        tab_data['N'] = treated_at_dose
        tab_data['Efficacies'] = eff_at_dose
        tab_data['Toxicities'] = tox_at_dose
        df = pd.DataFrame(tab_data)
        df['EffRate'] = np.where(df.N > 0, df.Efficacies / df.N, np.nan)
        df['ToxRate'] = np.where(df.N > 0, df.Toxicities / df.N, np.nan)
        return df

    def set_next_dose(self, dose):
        """ Set the next dose that should be given. """
        self._next_dose = dose

    def next_dose(self):
        """ Get the next dose that should be given. """
        return self._next_dose

    def update(self, cases, **kwargs):
        """ Update the trial with a list of cases.

        Params:
        cases, list of 3-tuples, (dose, toxicity, efficacy), where dose is the given (1-based) dose level,
                    toxicity = 1 for a toxicity event; 0 for a tolerance event,
                    efficacy = 1 for an efficacy event; 0 for a non-efficacy event.

        Returns: next dose

        """

        if len(cases) > 0:
            for (dose, tox, eff) in cases:
                self._doses.append(dose)
                self._toxicities.append(tox)
                self._efficacies.append(eff)

            self._next_dose = self.__calculate_next_dose(**kwargs)
        else:
            logging.warn('Cannot update design with no cases')

        return self._next_dose

    def admissable_set(self):
        """ Get the admissable set of doses. """
        return self._admissable_set

    def dose_admissability(self):
        return np.array([(x in self._admissable_set) for x in self.dose_levels()])

    def observed_toxicity_rates(self):
        """ Get the observed rate of toxicity at all doses. """
        tox_rates = []
        for d in range(1, self.num_doses+1):
            num_treated = self.treated_at_dose(d)
            if num_treated:
                num_toxes = self.toxicities_at_dose(d)
                tox_rates.append(1. * num_toxes / num_treated)
            else:
                tox_rates.append(np.nan)
        return np.array(tox_rates)

    def observed_efficacy_rates(self):
        """ Get the observed rate of efficacy at all doses. """
        eff_rates = []
        for d in range(1, self.num_doses+1):
            num_treated = self.treated_at_dose(d)
            if num_treated:
                num_responses = self.efficacies_at_dose(d)
                eff_rates.append(1. * num_responses / num_treated)
            else:
                eff_rates.append(np.nan)
        return np.array(eff_rates)

    def optimal_decision(self, prob_tox, prob_eff):
        """ Get the optimal dose choice for a given dose-toxicity curve.

        .. note:: Ken Cheung (2014) presented the idea that the optimal behaviour of a dose-finding
        design can be calculated for a given set of patients with their own specific tolerances by
        invoking the dose decicion on the complete (and unknowable) toxicity and efficacy curves.

        :param prob_tox: collection of toxicity probabilities
        :type prob_tox: list
        :param prob_tox: collection of efficacy probabilities
        :type prob_tox: list
        :return: the optimal (1-based) dose decision
        :rtype: int

        """

        raise NotImplementedError()

    @abc.abstractmethod
    def __reset(self):
        """ Opportunity to run implementation-specific reset operations. """
        return

    @abc.abstractmethod
    def has_more(self):
        """ Is the trial ongoing? """
        return (self.size() < self.max_size()) and (self._status >= 0)

    @abc.abstractmethod
    def __calculate_next_dose(self, **kwargs):
        """ Subclasses should override this method and return the desired next dose. """
        return -1  # Default implementation


def _efftox_patient_outcome_to_label(po):
    """ Converts (0,0) to Neither; (1,0) to Toxicity, (0,1) to Efficacy, (1,1) to Both """
    if po == (0,0):
        return 'Neither'
    elif po == (1,0):
        return 'Toxicity'
    elif po == (0,1):
        return 'Efficacy'
    elif po == (1,1):
        return 'Both'
    else:
        return 'Error'


def _simulate_trial(design, true_toxicities, true_efficacies, tox_eff_odds_ratio=1.0, tolerances=None,
                            cohort_size=1, conduct_trial=1, calculate_optimal_decision=1):
    """ Simulate a dose finding trial based on efficacy and toxicity, like EffTox, etc.

    :param design: the design with which to simulate a dose-finding trial.
    :type design: clintrials.dosefinding.EfficacyToxicityDoseFindingTrial
    :param true_toxicities: list of the true toxicity rates at the dose levels under investigation.
                            In real life, these are unknown but we use them in simulations to test the algorithm.
                            Should be same length as prior.
    :type true_toxicities: list
    :param true_efficacies: list of the true efficacy rates at the dose levels under investigation.
                            In real life, these are unknown but we use them in simulations to test the algorithm.
                            Should be same length as prior.
    :type true_efficacies: list
    :param tox_eff_odds_ratio: odds ratio of toxicity and efficacy events. Use 1. for no association
    :type tox_eff_odds_ratio: float
    :param tolerances: optional n_patients*3 array of uniforms used to infer correlated toxicity and efficacy events
                        for patients. This array is passed to function that calculates correlated binary events from
                        uniform variables and marginal probabilities.
                        Leave None to get randomly sampled data.
                        This parameter is specifiable so that dose-finding methods can be compared on same 'patients'.
    :type tolerances: numpy.array
    :param cohort_size: to add several patients at a dose at once
    :type cohort_size: int
    :param conduct_trial: True to conduct cohort-by-cohort dosing using the trial design; False to suppress
    :type conduct_trial: bool
    :param calculate_optimal_decision: True to calculate the optimal dose; False to suppress
    :type calculate_optimal_decision: bool

    :return: report of the simulation outcome as a JSON-able dict
    :rtype: dict

    """

    correlated_outcomes = tox_eff_odds_ratio < 1.0 or tox_eff_odds_ratio > 1.0

    # Simulate trial
    if conduct_trial:
        i = 0
        design.reset()
        dose_level = design.next_dose()
        while i <= design.max_size() and design.has_more():
            u = (true_toxicities[dose_level-1], true_efficacies[dose_level-1])
            if correlated_outcomes:
                # Where outcomes are associated, simulated outcomes must reflect the association.
                # There is a special method for that:
                events = correlated_binary_outcomes_from_uniforms(tolerances[i:i+cohort_size, ], u,
                                                                  psi=tox_eff_odds_ratio).astype(int)
            else:
                # Outcomes are not associated. Simply use first two columns of tolerances as
                # uniformally-distributed thresholds for tox and eff. The third col is ignored.
                events = (tolerances[i:i+cohort_size, 0:2] < u).astype(int)
            cases = np.column_stack(([dose_level] * cohort_size, events))
            dose_level = design.update(cases)
            i += cohort_size

    # Report findings
    report = OrderedDict()
    # report['TrueToxicities'] = iterable_to_json(true_toxicities)
    # report['TrueEfficacies'] = iterable_to_json(true_efficacies)
    # Do not parrot back parameters

    if conduct_trial:
        report['RecommendedDose'] = atomic_to_json(design.next_dose())
        report['TrialStatus'] = atomic_to_json(design.status())
        report['Doses'] = iterable_to_json(design.doses())
        report['Toxicities'] = iterable_to_json(design.toxicities())
        report['Efficacies'] = iterable_to_json(design.efficacies())
    # Optimal decision, given these specific patient tolerances
    if calculate_optimal_decision:
        try:
            if correlated_outcomes:
                tox_eff_hat = np.array([
                    correlated_binary_outcomes_from_uniforms(tolerances, v, psi=tox_eff_odds_ratio).mean(axis=0)
                    for v in zip(true_toxicities, true_efficacies)])
                tox_hat, eff_hat = tox_eff_hat[:, 0], tox_eff_hat[:, 1]
            else:
                had_tox = lambda x: x < np.array(true_toxicities)
                tox_horizons = np.array([had_tox(x) for x in tolerances[:, 0]])
                tox_hat = tox_horizons.mean(axis=0)
                had_eff = lambda x: x < np.array(true_efficacies)
                eff_horizons = np.array([had_eff(x) for x in tolerances[:, 1]])
                eff_hat = eff_horizons.mean(axis=0)

            optimal_allocation = design.optimal_decision(tox_hat, eff_hat)
            report['FullyInformedToxicityCurve'] = iterable_to_json(np.round(tox_hat, 4))
            report['FullyInformedEfficacyCurve'] = iterable_to_json(np.round(eff_hat, 4))
            report['OptimalAllocation'] = atomic_to_json(optimal_allocation)
        except NotImplementedError:
            pass

    return report
# Alias
_simulate_eff_tox_trial = _simulate_trial

def simulate_trial(design, true_toxicities, true_efficacies,
                   tox_eff_odds_ratio=1.0, tolerances=None, cohort_size=1,
                   conduct_trial=1, calculate_optimal_decision=1):
    """ Simulate a dose finding trial based on efficacy and toxicity, like EffTox, etc.

    :param design: the design with which to simulate a dose-finding trial.
    :type design: clintrials.dosefinding.EfficacyToxicityDoseFindingTrial
    :param true_toxicities: list of the true toxicity rates at the dose levels under investigation.
                            In real life, these are unknown but we use them in simulations to test the algorithm.
                            Should be same length as prior.
    :type true_toxicities: list
    :param true_efficacies: list of the true efficacy rates at the dose levels under investigation.
                            In real life, these are unknown but we use them in simulations to test the algorithm.
                            Should be same length as prior.
    :type true_efficacies: list
    :param tox_eff_odds_ratio: odds ratio of toxicity and efficacy events. Use 1. for no association
    :type tox_eff_odds_ratio: float
    :param tolerances: optional n_patients*3 array of uniforms used to infer correlated toxicity and efficacy events
                        for patients. This array is passed to function that calculates correlated binary events from
                        uniform variables and marginal probabilities.
                        Leave None to get randomly sampled data.
                        This parameter is specifiable so that dose-finding methods can be compared on same 'patients'.
    :type tolerances: numpy.array
    :param cohort_size: to add several patients at a dose at once
    :type cohort_size: int
    :param conduct_trial: True to conduct cohort-by-cohort dosing using the trial design; False to suppress
    :type conduct_trial: bool
    :param calculate_optimal_decision: True to calculate the optimal dose; False to suppress
    :type calculate_optimal_decision: bool

    :return: report of the simulation outcome as a JSON-able dict
    :rtype: dict

    """

    # Validation and derivation of the inputs
    if len(true_efficacies) != len(true_toxicities):
        raise ValueError('true_efficacies and true_toxicities should be same length.')
    if len(true_toxicities) != design.number_of_doses():
        raise ValueError('Length of true_toxicities and number of doses should be the same.')
    n_patients = design.max_size()
    if tolerances is not None:
        if tolerances.ndim != 2 or tolerances.shape[0] < n_patients:
            raise ValueError('tolerances should be an n_patients*3 array')
    else:
        tolerances = np.random.uniform(size=3*n_patients).reshape(n_patients, 3)

    if tox_eff_odds_ratio != 1.0 and calculate_optimal_decision:
        logging.warn('Patient outcomes are not sequential when toxicity and efficacy events are correlated. ' +
                     'E.g. toxicity at d_1 dose not necessarily imply toxicity at d_2. It is important ' +
                     'to appreciate this when calculating optimal decisions.')

    return _simulate_trial(design, true_toxicities, true_efficacies, tox_eff_odds_ratio, tolerances,
                                   cohort_size, conduct_trial, calculate_optimal_decision)
# Alias
simulate_efficacy_toxicity_dose_finding_trial = simulate_trial

def simulate_efficacy_toxicity_dose_finding_trials(design_map, true_toxicities, true_efficacies,
                                                   tox_eff_odds_ratio=1.0, tolerances=None, cohort_size=1,
                                                   conduct_trial=1, calculate_optimal_decision=1):
    """ Simulate multiple dose finding trials based on efficacy and toxicity, like EffTox, etc.

    This method lets you see how different designs handle a single common set of patient outcomes.

    :param design_map: dict, label -> instance of EfficacyToxicityDoseFindingTrial
    :type design_map: dict
    :param true_toxicities: list of the true toxicity rates at the dose levels under investigation.
                            In real life, these are unknown but we use them in simulations to test the algorithm.
                            Should be same length as prior.
    :type true_toxicities: list
    :param true_efficacies: list of the true efficacy rates at the dose levels under investigation.
                            In real life, these are unknown but we use them in simulations to test the algorithm.
                            Should be same length as prior.
    :type true_efficacies: list
    :param tox_eff_odds_ratio: odds ratio of toxicity and efficacy events. Use 1. for no association
    :type tox_eff_odds_ratio: float
    :param tolerances: optional n_patients*3 array of uniforms used to infer correlated toxicity and efficacy events
                        for patients. This array is passed to function that calculates correlated binary events from
                        uniform variables and marginal probabilities.
                        Leave None to get randomly sampled data.
                        This parameter is specifiable so that dose-finding methods can be compared on same 'patients'.
    :type tolerances: numpy.array
    :param cohort_size: to add several patients at a dose at once
    :type cohort_size: int
    :param conduct_trial: True to conduct cohort-by-cohort dosing using the trial design; False to suppress
    :type conduct_trial: bool
    :param calculate_optimal_decision: True to calculate the optimal dose; False to suppress
    :type calculate_optimal_decision: bool

    :return: report of the simulation outcomes as a JSON-able dict. The outcome for each design is encased in its own
                map, keyed by the name (i.e. the key) in design_map.
    :rtype: dict

    """

    max_size = max([design.max_size() for design in design_map.values()])
    if tolerances is not None:
        if tolerances.ndim != 2 or tolerances.shape[0] < max_size:
            raise ValueError('tolerances should be an max_size*3 array')
    else:
        tolerances = np.random.uniform(size=3*max_size).reshape(max_size, 3)

    if tox_eff_odds_ratio != 1.0 and calculate_optimal_decision:
        logging.warn('Patient outcomes are not sequential when toxicity and efficacy events are correlated. ' +
                     'E.g. toxicity at d_1 dose not necessarily imply toxicity at d_2. It is important ' +
                     'to appreciate this when calculating optimal decisions.')

    report = OrderedDict()
    # report['TrueToxicities'] = iterable_to_json(true_toxicities)
    # report['TrueEfficacies'] = iterable_to_json(true_efficacies)
    # Do not parrot back parameters

    for label, design in design_map.iteritems():
        this_sim = _simulate_trial(design, true_toxicities, true_efficacies, tox_eff_odds_ratio, tolerances,
                                           cohort_size, conduct_trial, calculate_optimal_decision)
        report[label] = this_sim

    return report
# Alias
simulate_trials = simulate_efficacy_toxicity_dose_finding_trials

def dose_transition_pathways(trial, next_dose, cohort_sizes, cohort_number=1, cases_already_observed=[],
                                    custom_output_func=None, verbose=False, **kwargs):
    """ Calculate dose-transition pathways for an efficacy-toxicity design.

    :param trial: subclass of EfficacyToxicityDoseFindingTrial that will determine the dose path
    :type trial: clintrials.dosefinding.EfficacyToxicityDoseFindingTrial
    :param next_dose: the dose that will be given to patients in the very next cohort to get things going.
    :type next_dose: int
    :param cohort_sizes: list of ints, sizes of future cohorts that we want to calculate DTPs for.
                            E.g. use [3,2] to calculate DTPs for two subsequent cohorts, the first of
                            three patients followed by another cohort of two.
    :type cohort_size: list
    :param cohort_number: The decorative cohort number label for the first cohort
    :type cohort_number: int
    :param cases_already_observed: list of (dose, tox=0/1, eff=0/1) cases that have already been observed
    :type cases_already_observed: list
    :param custom_output_func: func that takes trial as sole argument and returns dict of extra output.
                                Called at end of each cohort, i.e. at each dose decision.
    :type custom_output_func: func
    :param verbose: True to print extra information to monitor progress
    :type verbose: bool
    :param kwargs: extra keyword args to send to trial.update method
    :type kwargs: dict

    :return: DTPs as JSON-able dict object. Paths are nested.
    :rtype: dict

    """

    if len(cohort_sizes) <= 0:
        return None
    else:
        cohort_size = cohort_sizes[0]
        patient_outcomes = [(0, 0), (0, 1), (1, 0), (1, 1)]
        cohort_outcomes = list(combinations_with_replacement(patient_outcomes, cohort_size))
        path_outputs = []
        for i, path in enumerate(cohort_outcomes):
            # Invoke dose-decision
            cohort_cases = [(next_dose, x[0], x[1]) for x in path]
            cases = cases_already_observed + cohort_cases
            if verbose:
                print('Running %s' % cases)
            trial.reset()
            obd = trial.update(cases, **kwargs)
            # Collect output
            bag_o_tricks = OrderedDict([('Pat{}.{}'.format(cohort_number, j+1), _efftox_patient_outcome_to_label(po))
                                        for (j, po) in enumerate(path)])
            bag_o_tricks.update(OrderedDict([
                        ('DoseGiven', atomic_to_json(next_dose)),
                        ('RecommendedDose', atomic_to_json(obd)),
                        ('CohortSize', cohort_size),
                        ('NumEff', sum([x[1] for x in path])),
                        ('NumTox', sum([x[0] for x in path])),
                    ]))
            if custom_output_func:
                bag_o_tricks.update(custom_output_func(trial))

            # Recurse subsequent cohorts
            further_paths = dose_transition_pathways(trial, next_dose=obd, cohort_sizes=cohort_sizes[1:],
                                                     cohort_number=cohort_number+1, cases_already_observed=cases,
                                                     custom_output_func=custom_output_func, verbose=verbose,
                                                     **kwargs)
            if further_paths:
                bag_o_tricks['Next'] = further_paths

            path_outputs.append(bag_o_tricks)

        return path_outputs
# Aliases
efftox_dose_transition_pathways = dose_transition_pathways
efficacy_toxicity_dose_transition_pathways = dose_transition_pathways


def get_path(x, dose_label_func=None):
    if dose_label_func is None:
        dose_label_func = lambda x: str(x)
    path = [x[z] for z in sorted([z for z in x.keys() if 'Pat' in z])]
    path = [z[0] for z in path]
    path = ''.join(path)
    path = dose_label_func(x['DoseGiven']) + path
    return path


def print_dtps(dtps, indent=0, dose_label_func=None):
    if dose_label_func is None:
        dose_label_func = lambda x: str(x)
    for x in dtps:
        path = get_path(x, dose_label_func=dose_label_func)
        obd = x['RecommendedDose']
        prob_sup = x['MinProbSuperiority']

        if prob_sup < 0.6:
            template_txt = '\t' * indent + '{} -> Dose {}, Superiority={} * tentative *'
        else:
            template_txt = '\t' * indent + '{} -> Dose {}, Superiority={}'
        print(template_txt.format(path, dose_label_func(obd), np.round(prob_sup, 2)))

        if 'Next' in x:
            print_dtps(x['Next'], indent=indent+1, dose_label_func=dose_label_func)


def print_dtps_verbose(dtps, indent=0, dose_label_func=None):
    if dose_label_func is None:
        dose_label_func = lambda x: str(x)
    for x in dtps:
        path = get_path(x, dose_label_func=dose_label_func)
        obd = x['RecommendedDose']
        prob_sup = x['MinProbSuperiority']
        util = [x['Utility1'], x['Utility2'], x['Utility3'], x['Utility4']]
        prob_acc_eff = [x['ProbAccEff1'], x['ProbAccEff2'], x['ProbAccEff3'], x['ProbAccEff4']]
        prob_acc_tox = [x['ProbAccTox1'], x['ProbAccTox2'], x['ProbAccTox3'], x['ProbAccTox4']]
        template_txt = '\t' * indent + '{} -> Dose {}, Sup={}, Util={}, Pr(Acc Eff)={}, Pr(Acc Tox)={}'
        print(template_txt.format(path, dose_label_func(obd), np.round(prob_sup, 2), np.round(util, 2),
                                 np.round(prob_acc_eff, 2), np.round(prob_acc_tox,2)))

        if 'Next' in x:
            print_dtps_verbose(x['Next'], indent=indent+1, dose_label_func=dose_label_func)
