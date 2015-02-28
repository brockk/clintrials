__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'


__all__ = ["crm", "efftox", "novel", "wagestait"]


import abc
from collections import OrderedDict
from itertools import product
import logging
import numpy as np
import pandas as pd
from scipy.stats import uniform

from clintrials.util import atomic_to_json, iterable_to_json
from clintrials.util import correlated_binary_outcomes_from_uniforms, to_1d_list


class DoseFindingTrial(object):
    """ This is the base class for a dose-finding trial.

    The interface for such a class is:
    status()
    reset()
    number_of_doses()
    first_dose()
    size()
    max_size()
    doses()
    toxicities()
    treated_at_dose(dose)
    toxicities_at_dose(dose)
    maximum_dose_given()
    set_next_dose(dose)
    next_dose()
    update(cases)
    has_more()

    Further internal interface is provided by:
    __reset()
    __process_cases(cases)
    __calculate_next_dose() # Subclasses should override this method, set _status and return value for _next_dose.

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
        """

        Params:
        first_dose, starting dose level, 1-based. I.e. first_dose=3 means the middle dose of 5.
        num_doses, number of doses being tested. Must be at least equal to first_dose!
        max_size, maximum number of patients to use in trial

        """

        if first_dose > num_doses:
            raise ValueError('First dose must be no greater than number of doses.')

        self._first_dose = first_dose
        self.num_doses = num_doses
        self._max_size = max_size
        # Reset
        self._doses = []
        self._toxicities = []
        self._next_dose = self._first_dose
        self._status = 0

    def status(self):
        return self._status

    def reset(self):
        self._doses = []
        self._toxicities = []
        self._next_dose = self._first_dose
        self._status = 0
        self.__reset()

    def number_of_doses(self):
        """ How many dose-levels are under investigation?"""
        return self.num_doses

    def first_dose(self):
        """ Get first dose
        :return: First dose
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

    def treated_at_dose(self, dose):
        """ Number of patients treated at a dose level. """
        return sum(np.array(self._doses) == dose)

    def toxicities_at_dose(self, dose):
        """ Number of toxicities at (1-based) dose level. """
        return sum([t for d, t in zip(self.doses(), self.toxicities()) if d == dose])

    def maximum_dose_given(self):
        if len(self._doses) > 0:
            return max(self._doses)
        else:
            return None

    def set_next_dose(self, dose):
        """ Set the next dose that should be given. """
        self._next_dose = dose

    def next_dose(self):
        """ Get the next dose that should be given. """
        return self._next_dose

    def update(self, cases):
        """ Update the trial with a list of cases.

        Params:
        cases, list of 2-tuples, (dose, toxicity), where dose is the given (1-based) dose level
                    and toxicity = 1 for a toxicity event; 0 for a tolerance event.

        Returns: next dose

        """

        for (dose, tox) in cases:
            self._doses.append(dose)
            self._toxicities.append(tox)

        self.__process_cases(cases)
        self._next_dose = self.__calculate_next_dose()
        return self._next_dose

    @abc.abstractmethod
    def __reset(self):
        """ Opportunity to run implementation-specific reset operations. """
        return

    @abc.abstractmethod
    def has_more(self):
        """ Is the trial ongoing? """
        return (self.size() < self.max_size()) and (self._status >= 0)

    @abc.abstractmethod
    def __process_cases(self, cases):
        """ Subclasses should override this method to perform an cases-specific processing. """
        return  # Default implementation

    @abc.abstractmethod
    def __calculate_next_dose(self):
        """ Subclasses should override this method and return the desired next dose. """
        return -1  # Default implementation


class SimpleToxicityCountingDoseEscalationTrial(DoseFindingTrial):
    """ Simple design to monotonically increase dose until a certain number of toxicities are observed in aggregate.

    Recommends the highest dose given.

    e.g. general usage
    >>> trial = SimpleToxicityCountingDoseEscalationTrial(first_dose=1, num_doses=5, max_size=10, max_toxicities=3)
    >>> trial.update([(1,0)])
    2
    >>> trial.has_more()
    True
    >>> trial.update([(2,0), (2,0), (2,1), (2,1), (2,0)])
    3
    >>> trial.has_more()
    True
    >>> trial.update([(3,0), (3,1), (3,1)])
    3
    >>> trial.has_more()
    False
    >>> trial.toxicities_at_dose(3)
    2
    >>> trial.next_dose()
    3

    """

    def __init__(self, first_dose, num_doses, max_size, max_toxicities=1):

        DoseFindingTrial.__init__(self, first_dose=first_dose, num_doses=num_doses, max_size=max_size)

        self.max_toxicities = max_toxicities
        # Reset
        self.max_dose_given = -1

    def _DoseFindingTrial__reset(self):
        self.max_dose_given = -1

    def _DoseFindingTrial__process_cases(self, cases):
        return

    def _DoseFindingTrial__calculate_next_dose(self):
        if self.has_more():
            self._status = 1
            return min(max(self.doses()) + 1, self.number_of_doses())
        else:
            self._status = 100
            return max(self.doses())

    def has_more(self):
        return DoseFindingTrial.has_more(self) and (sum(self.toxicities()) < self.max_toxicities) \
               and self.maximum_dose_given() < self.number_of_doses()


class MultiStageDoseFindingTrial(DoseFindingTrial):
    """ Sequentially conduct dose-finding trials, with the first feeding into the second if desired.

    e.g. trivial example that hopefully illustrates usage of transition_mode='informs'
    >>> stage_1 = SimpleToxicityCountingDoseEscalationTrial(first_dose=1, num_doses=5, max_size=10, max_toxicities=1)
    >>> stage_2 = SimpleToxicityCountingDoseEscalationTrial(first_dose=1, num_doses=5, max_size=10, max_toxicities=2)
    >>> trial = MultiStageDoseFindingTrial(first_dose=1, num_doses=5, max_size=6+18, first_design=stage_1,
    ...                                    second_design=stage_2, transition_mode='informs')
    >>> trial.current_trial_cursor
    0
    >>> trial.update([(1,0), (2,0), (3,0), (3,1)])
    3
    >>> trial.has_more()
    True
    >>> trial.dose_finding_trials[0].size()
    4
    >>> trial.dose_finding_trials[1].size()
    0
    >>> trial.current_trial_cursor
    1
    >>> trial.next_dose()
    3
    >>> trial.update([(3,0), (3,0), (3,1)])
    4
    >>> trial.update([(3,0), (4,0), (4,1), (4,1)])
    4
    >>> trial.has_more()
    False
    >>> trial.next_dose()
    4

    and now an example to illustrate
    >>> stage_1 = SimpleToxicityCountingDoseEscalationTrial(first_dose=1, num_doses=5, max_size=6, max_toxicities=1)
    >>> stage_2 = SimpleToxicityCountingDoseEscalationTrial(first_dose=1, num_doses=5, max_size=18, max_toxicities=2)
    >>> trial2 = MultiStageDoseFindingTrial(first_dose=1, num_doses=5, max_size=6+18, first_design=stage_1,
    ...                                     second_design=stage_2, transition_mode='feeds')
    >>> trial2.current_trial_cursor
    0
    >>> trial2.update([(1,0), (2,1)])
    3
    >>> trial2.current_trial_cursor
    1
    >>> trial2.dose_finding_trials[0].size()
    2
    >>> trial2.dose_finding_trials[1].size()
    2
    >>> trial2.has_more()
    True
    >>> trial2.update([(3,1)])
    3
    >>> trial2.has_more()
    False
    >>> trial2.next_dose()
    3

    """

    def __init__(self, first_dose, num_doses, max_size, first_design, second_design, transition_mode='feeds'):
        """
        Params:
        transition_mode, one of:
            feeds, for all observed dose and toxicity data from first design to be fed into second design at transition
            informs, for second design to start at the MTD determined by the first design
        """

        DoseFindingTrial.__init__(self, first_dose=first_dose, num_doses=num_doses, max_size=max_size)

        self.dose_finding_trials = [first_design, second_design]
        self.transition_mode = transition_mode
        # Reset
        self.current_trial_cursor = 0  # Index of trial design currently being used to provide doses

    def _DoseFindingTrial__reset(self):
        self.current_trial_cursor = 0
        for design in self.dose_finding_trials:
            design.reset()

    def _DoseFindingTrial__process_cases(self, cases):
        if self.current_trial_cursor < len(self.dose_finding_trials):
            current_design = self.dose_finding_trials[self.current_trial_cursor]
            _next_dose = current_design.update(cases)
            if not current_design.has_more():
                self.current_trial_cursor += 1
                if self.current_trial_cursor < len(self.dose_finding_trials):
                    previous_design = current_design
                    current_design = self.dose_finding_trials[self.current_trial_cursor]
                    if self.transition_mode == 'feeds':
                        observed_cases = [(x, y) for (x, y) in zip(previous_design.doses(), previous_design.toxicities())]
                        _next_dose = current_design.update(observed_cases)
                    elif self.transition_mode == 'informs':
                        current_design.set_next_dose(_next_dose)
                    else:
                        raise ValueError('Invalid transition mode: ', self.transition_mode)
            return
        else:
            if len(self.dose_finding_trials) > 0:
                logging.error('There is no dose-finding design to update. All designs are exhausted.')
            return

    def _DoseFindingTrial__calculate_next_dose(self):
        if self.current_trial_cursor < len(self.dose_finding_trials):
            current_design = self.dose_finding_trials[self.current_trial_cursor]
            self._status = current_design.status()
            return current_design.next_dose()
        elif len(self.dose_finding_trials) > 0:
            last_design = self.dose_finding_trials[-1]
            self._status = last_design.status()
            return last_design.next_dose()
        else:
            self._status = -10
            return -1

    def has_more(self):
        if self.current_trial_cursor < len(self.dose_finding_trials):
            current_design = self.dose_finding_trials[self.current_trial_cursor]
            if current_design.has_more():
                return DoseFindingTrial.has_more(self)
            elif self.current_trial_cursor+1 < len(self.dose_finding_trials):
                next_design = self.dose_finding_trials[self.current_trial_cursor+1]
                if next_design.has_more():
                    return DoseFindingTrial.has_more(self)

        # Got here? There is no more
        return False


class ThreePlusThree(DoseFindingTrial):
    """ This is an object-oriented attempt at the 3+3 trial design.

    e.g. general usage
    >>> trial = ThreePlusThree(5)
    >>> trial.next_dose()
    1
    >>> trial.update([(1,0), (1,0), (1,0)])
    2
    >>> trial.has_more()
    True
    >>> trial.update([(2,1), (2,0), (2,0)])
    2
    >>> trial.has_more()
    True
    >>> trial.update([(2,0), (2,0), (2,0)])
    3
    >>> trial.has_more()
    True
    >>> trial.update([(3,1), (3,0), (3,0)])
    3
    >>> trial.has_more()
    True
    >>> trial.update([(3,1), (3,0), (3,0)])
    2
    >>> trial.has_more()
    False
    >>> trial.size()
    15
    >>> trial.toxicities_at_dose(3)
    2
    >>> trial.next_dose()
    2

    And some obvious mistakes
    >>> trial2 = ThreePlusThree(5)
    >>> trial2.update([(1,0)])
    Traceback (most recent call last):
     ...
    Exception: Doses in the 3+3 trial must be given in common batches of three.
    >>> trial3 = ThreePlusThree(5)
    >>> trial3.update([(1,0), (1,0), (2,0)])
    Traceback (most recent call last):
     ...
    Exception: Doses in the 3+3 trial must be given in common batches of three.

    """

    def __init__(self, num_doses):

        DoseFindingTrial.__init__(self, first_dose=1, num_doses=num_doses, max_size=6*num_doses)

        self.num_doses = num_doses
        self.cohort_size = 3
        # Reset
        self._continue = True

    def _DoseFindingTrial__reset(self):
        self._continue = True

    def _DoseFindingTrial__process_cases(self, cases):
        return

    def _DoseFindingTrial__calculate_next_dose(self):
        dose_indices = np.array(self._doses) == self._next_dose
        toxes_at_dose = sum(np.array(self._toxicities)[dose_indices])
        if sum(dose_indices) == 3:
            if toxes_at_dose == 0:
                if self._next_dose < self.num_doses:
                    # escalate
                    self._status = 1
                    self._next_dose += 1
                else:
                    # end trial
                    self._status = 100
                    self._continue = False
            elif toxes_at_dose == 1:
                # Do not escalate but continue trial
                self._status = 1
                pass
            else:
                # too many toxicities at this dose so de-escalate and end trial
                self._next_dose -= 1
                if self._next_dose > 0:
                    self._status = 100
                else:
                    self._status = -1
                self._continue = False
        elif sum(dose_indices) == 6:
            if toxes_at_dose <= 1:
                if self._next_dose < self.num_doses:
                    # escalate
                    self._status = 1
                    self._next_dose += 1
                else:
                    # end trial
                    self._status = 100
                    self._continue = False
            else:
                # too many toxicities at this dose so de-escalate and end trial
                self._next_dose -= 1
                if self._next_dose > 0:
                    self._status = 100
                else:
                    self._status = -1
                self._continue = False
        else:
            msg = 'Doses in the 3+3 trial must be given in common batches of three.'
            raise Exception(msg)

        return self._next_dose

    def has_more(self):
        """ Is the trial ongoing? 3+3 stops when the MTD has been found. """
        return DoseFindingTrial.has_more(self) and self._continue


def simulate_dose_finding_trial(design, true_toxicities, tolerances=None, cohort_size=1, to_json=0):
    """ Simulate a dose finding trial based on toxicity, like CRM, 3+3, etc.

    Params:
    design, an instance of DoseFindingTrial
    true_toxicities, list of the true toxicity rates at the dose levels under investigation.
                    Obviously these are unknown in real-life but we use them in simulations to test the algorithm.
                    Should be same length as prior.
    tolerances, list of tolerances. Patient experiences toxicity if their tolerance is less than the probability
                    of toxicity at the dose they received. Leave None to get random uniform tolerances.
    cohort_size, to add several patients at once
    to_json, True to get result returned as JSON-friendly object

    Returns, if to_json, a dict-like
             else, a 2-tuple: (dose selected, dictionary of Tox, Dose and Tol data)

    """

    if tolerances is None:
        tolerances = uniform().rvs(design.max_size())
    else:
        if len(tolerances) < design.max_size():
            logging.warn('You have provided fewer tolerances than maximum number of patients on trial. Beware errors!')

    i = 0
    design.reset()
    dose_level = design.next_dose()
    while i <= design.max_size() and design.has_more():
        tox = [1 if x < true_toxicities[dose_level-1] else 0 for x in tolerances[i:i+cohort_size]]
        cases = zip([dose_level] * cohort_size, tox)
        dose_level = design.update(cases)
        i += cohort_size

    if to_json:
        report = OrderedDict()
        report['TrueToxicities'] = iterable_to_json(true_toxicities)
        report['RecommendedDose'] = atomic_to_json(design.next_dose())
        report['TrialStatus'] = atomic_to_json(design.status())
        report['Doses'] = iterable_to_json(design.doses())
        report['Toxicities'] = iterable_to_json(design.toxicities())
        return report
    else:
        sim = {'Tox': design.toxicities(), 'Dose': design.doses(), 'Tol': tolerances}
        return design.next_dose(), sim


def simulate_dose_finding_trials(design_map, true_toxicities, tolerances=None, cohort_size=1):
    """ Simulate multiple toxicity-driven dose finding trials (like CRM, 3+3, etc) from the same patient data.

    This method lets you see how different designs handle a single common set of patient outcomes.

    Params:
    design_map, dict, label -> instance of DoseFindingTrial
    true_toxicities, list of the true toxicity rates. Obviously these are unknown in real-life but
                    we use them in simulations to test the algorithm. Should be same length as prior.
    tolerances, list of tolerances. Leave None to get random tolerances.
    cohort_size, to add several patients at once
    to_json, True to get result returned as JSON-friendly object

    Returns, if to_json, a dict-like
             else, a 2-tuple: (dose selected, dictionary of Tox, Dose and Tol data)

    """

    max_size = max([design.max_size() for design in design_map.values()])
    if tolerances is None:
        tolerances = uniform().rvs(max_size)
    else:
        if len(tolerances) < max_size:
            logging.warn('You have provided fewer tolerances than maximum number of patients on trial. Beware errors!')

    report = OrderedDict()
    report['TrueToxicities'] = true_toxicities
    for label, design in design_map.iteritems():
        design_sim = simulate_dose_finding_trial(design, true_toxicities, tolerances=tolerances,
                                                 cohort_size=cohort_size, to_json=1)
        report[label] = design_sim
    return report


class EfficacyToxicityDoseFindingTrial(object):
    """ This is the base class for a dose-finding trial that jointly monitors toxicity and efficacy.

    The interface for such a class is:
    status()
    reset()
    number_of_doses()
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
    set_next_dose(dose)
    next_dose()
    update(cases)
    has_more()
    admissable_set()

    Further internal interface is provided by:
    __reset()
    __process_cases(cases)
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

    # def __init__(self):
    #     # TODO
    #     # Reset
    #     self._status = 0

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

        for (dose, tox, eff) in cases:
            self._doses.append(dose)
            self._toxicities.append(tox)
            self._efficacies.append(eff)

        self.__process_cases(cases)
        self._next_dose = self.__calculate_next_dose(**kwargs)
        return self._next_dose

    def admissable_set(self):
        """ Get the admissable set of doses. """
        return self._admissable_set

    @abc.abstractmethod
    def __reset(self):
        """ Opportunity to run implementation-specific reset operations. """
        return

    @abc.abstractmethod
    def has_more(self):
        """ Is the trial ongoing? """
        return (self.size() < self.max_size()) and (self._status >= 0)

    @abc.abstractmethod
    def __process_cases(self, cases):
        """ Subclasses should override this method to perform an cases-specific processing. """
        return  # Default implementation

    @abc.abstractmethod
    def __calculate_next_dose(self, **kwargs):
        """ Subclasses should override this method and return the desired next dose. """
        return -1  # Default implementation


def simulate_efficacy_toxicity_dose_finding_trial(design, true_toxicities, true_efficacies,
                                                  tox_eff_odds_ratio=1.0, tolerances=None, cohort_size=1,
                                                  to_json=0):
    """ Simulate a dose finding trial based on efficacy and toxicity, like EffTox, etc.

    Params:
    design, an instance of EfficacyToxicityDoseFindingTrial
    true_toxicities, list of the true toxicity rates at the dose levels under investigation.
                     Obviously these are unknown in real-life but we use them in simulations to test the algorithm.
                     Should be same length as prior.
    true_efficacies, list of the true efficacy rates at the dose levels under investigation.
                     Obviously these are unknown in real-life but we use them in simulations to test the algorithm.
                     Should be same length as prior.
    tox_eff_odds_ratio, odds ratio of toxicity and efficacy events. Use 1. for no association
    tolerances, optional n_patients*3 array of uniforms used to infer correlated toxicity and efficacy events
                        for patients. This array is passed to function that calculates correlated binary events from
                        uniform variables and marginal probabilities.
                        Leave None to get randomly sampled data.
                        This parameter is specifiable so that dose-finding methods can be compared on same 'patients'.
    cohort_size, to add several patients at once
    to_json, True to get result returned as JSON-friendly object

    Returns, if to_json, a dict-like
             else, a 2-tuple: (dose selected, dictionary of Tox, Dose and Tol data)

    """

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

    i = 0
    design.reset()
    dose_level = design.next_dose()
    while i <= design.max_size() and design.has_more():
        u = (true_toxicities[dose_level-1], true_efficacies[dose_level-1])
        events = correlated_binary_outcomes_from_uniforms(tolerances[i:i+cohort_size, ], u,
                                                          psi=tox_eff_odds_ratio).astype(int)
        cases = np.column_stack(([dose_level] * cohort_size, events))
        dose_level = design.update(cases)
        i += cohort_size

    if to_json:
        report = OrderedDict()
        report['TrueToxicities'] = iterable_to_json(true_toxicities)
        report['TrueEfficacies'] = iterable_to_json(true_efficacies)
        report['RecommendedDose'] = atomic_to_json(design.next_dose())
        report['TrialStatus'] = atomic_to_json(design.status())
        report['Doses'] = iterable_to_json(design.doses())
        report['Toxicities'] = iterable_to_json(design.toxicities())
        report['Efficacies'] = iterable_to_json(design.efficacies())
        return report
    else:
        sim = {'Dose': design.doses(), 'Tox': design.toxicities(), 'Eff': design.efficacies(), 'Tol': tolerances}
        return design.next_dose(), sim


def simulate_efficacy_toxicity_dose_finding_trials(design_map, true_toxicities, true_efficacies,
                                                   tox_eff_odds_ratio=1.0, tolerances=None, cohort_size=1):
    """ Simulate multiple dose finding trials based on efficacy and toxicity, like EffTox, etc.

    This method lets you see how different designs handle a single common set of patient outcomes.

    Params:
    design_map, dict, label -> instance of EfficacyToxicityDoseFindingTrial
    true_toxicities, list of the true toxicity rates at the dose levels under investigation.
                     Obviously these are unknown in real-life but we use them in simulations to test the algorithm.
                     Should be same length as prior.
    true_efficacies, list of the true efficacy rates at the dose levels under investigation.
                     Obviously these are unknown in real-life but we use them in simulations to test the algorithm.
                     Should be same length as prior.
    tox_eff_odds_ratio, odds ratio of toxicity and efficacy events. Use 1. for no association
    tolerances, optional n_patients*3 array of uniforms used to infer correlated toxicity and efficacy events
                        for patients. This array is passed to function that calculates correlated binary events from
                        uniform variables and marginal probabilities.
                        Leave None to get randomly sampled data.
                        This parameter is specifiable so that dose-finding methods can be compared on same 'patients'.
    cohort_size, to add several patients at once
    to_json, True to get result returned as JSON-friendly object

    Returns, if to_json, a dict-like
             else, a 2-tuple: (dose selected, dictionary of Tox, Dose and Tol data)

    """

    max_size = max([design.max_size() for design in design_map.values()])
    if tolerances is not None:
        if tolerances.ndim != 2 or tolerances.shape[0] < max_size:
            raise ValueError('tolerances should be an max_size*3 array')
    else:
        tolerances = np.random.uniform(size=3*max_size).reshape(max_size, 3)

    report = OrderedDict()
    report['TrueToxicities'] = true_toxicities
    report['TrueEfficacies'] = true_efficacies
    for label, design in design_map.iteritems():
        design_sim = simulate_efficacy_toxicity_dose_finding_trial(design, true_toxicities, true_efficacies,
                                                                   tox_eff_odds_ratio, tolerances=tolerances,
                                                                   cohort_size=cohort_size, to_json=1)
        report[label] = design_sim
    return report


def find_mtd(toxicity_target, scenario, strictly_lte=False, verbose=False):
    """ Find the MTD in a list of toxicity probabilities and a target toxicity rate.

    :param toxicity_target: target probability of toxicity
    :type toxicity_target: float
    :param scenario: list of probabilities of toxicity at each dose
    :type scenario: list
    :param strictly_lte: True to demand that Prob(toxicity) at MD is <= toxicity_target;
                         False to allow it to be over if it is near.
    :type strictly_lte: bool
    :param verbose: True to print output
    :type verbose: bool
    :return: 1-based location of MTD
    :rtype: int

    For example,

    >>> find_mtd(0.25, [0.15, 0.25, 0.35], strictly_lte=0)
    2
    >>> find_mtd(0.25, [0.15, 0.25, 0.35], strictly_lte=1)
    2
    >>> find_mtd(0.25, [0.3, 0.4, 0.5], strictly_lte=0)
    1
    >>> find_mtd(0.25, [0.3, 0.4, 0.5], strictly_lte=1)
    0
    >>> find_mtd(0.25, [0.20, 0.22, 0.26], strictly_lte=0)
    3
    >>> find_mtd(0.25, [0.20, 0.22, 0.26], strictly_lte=1)
    2

    """

    if toxicity_target in scenario:
        # Return exact match
        loc = scenario.index(toxicity_target) + 1
        if verbose:
            print 'MTD is', loc
        return loc
    else:
        if strictly_lte:
            if sum(np.array(scenario) <= toxicity_target) == 0:
                # Infeasible scenario
                if verbose:
                    print 'All doses are too toxic'
                return 0
            else:
                # Return highest tox no greater than target
                objective = np.where(np.array(scenario)<=toxicity_target, toxicity_target-np.array(scenario), np.inf)
                loc = np.argmin(objective) + 1
                if verbose:
                    print 'Highest dose below MTD is', loc
                return loc
        else:
            # Return nearest
            loc = np.argmin(np.abs(np.array(scenario) - toxicity_target)) + 1
            if verbose:
                print 'Dose nearest to MTD is', loc
            return loc


def summarise_dose_finding_sims(sims, label, num_doses, filter={}):
    """ Summarise a list of dose-finding simulations for doses recommended, doses given and trial outcome.

    :param sims: list of JSON reps of dose-finding trial outcomes
    :type sims: list
    :param label: name of simulation at first level in each JSON object
    :type label: str
    :param num_doses: number of dose levels under study
    :type num_doses: int
    :param filter: map of item to item-value to filter list of simulations
    :type filter: dict
    :return: 5-tuple, (doses DataFrame, outcomes DataFrame, doses chosen, doses given to patients,
                        trial end statuses), the first two as pandas DataFrames and the latter three as numpy arrays.
    :rtype: tuple

    .. note::
        This function is a bit of a mess but it is useful. Use methods in clintrials.simulation instead.

    """

    # Quick and dirty filter
    for key, val in filter.iteritems():
        sims = [x for x in sims if x[key] == val]

    # Recommended Doses
    doses = [x[label]['RecommendedDose'] for x in sims]
    df_doses = pd.DataFrame({'RecN': pd.Series(doses).value_counts()}, index=range(0,num_doses+1))
    df_doses['Rec%'] = 1.0 * df_doses['RecN'] / df_doses['RecN'].sum()
    # Given Doses
    doses_given = to_1d_list([x[label]['Doses'] for x in sims])
    df_doses = df_doses.join(pd.DataFrame({'PatN': pd.Series(doses_given).value_counts()}))
    df_doses['Pat%'] = 1.0 * df_doses['PatN'] / df_doses['PatN'].sum()
    # Order
    df_doses = df_doses.loc[range(0, num_doses+1)]

    # Trial Outcomes
    statuses = [x[label]['TrialStatus'] for x in sims]
    df_statuses = pd.DataFrame({'N': pd.Series(statuses).value_counts()})
    df_statuses['%'] = 1.0 * df_statuses['N'] / df_statuses['N'].sum()

    return df_doses, df_statuses, np.array(doses), np.array(doses_given), np.array(statuses)


def batch_summarise_dose_finding_sims(sims, label, num_doses, dimensions=None, func1=None):
    """ Batch summarise a list of dose-finding simulations.

    The dimensions along which to group and simmarise the simulations are determined via dimensions (see below)

    :param sims: list of JSON reps of dose-finding trial outcomes
    :type sims: list
    :param label: name of simulation at first level in each JSON object
    :type label: str
    :param num_doses: number of dose levels under study
    :type num_doses: int
    :param dimensions: 2-tuple, (dict of JSON variable name -> arg name in ParameterSpace, instance of ParameterSpace)
    :type dimensions: tuple
    :param func1: Function that takes pandas.DataFrame as first arg and a dict of variable name -> values as second arg
                    and returns summary output.
                    Use func1=None to just see the pandas.DataFrames printed as summary.
    :type func1: func

    .. note::
        This function is a bit of a mess but it is useful. Use methods in clintrials.simulation instead.

    """
    if dimensions is not None:
        var_map, params = dimensions
        z = [(k, params[v]) for k,v in var_map.iteritems()]
        labels, val_arrays = zip(*z)
        param_combinations = list(product(*val_arrays))
        for param_combo in param_combinations:
            for lab, vals in zip(labels, param_combo):
                print '{}: {}'.format(lab, vals)
            these_params = dict(zip(labels, param_combo))
            abc = summarise_dose_finding_sims(sims, label, num_doses, filter=these_params)
            if func1:
                print func1(abc[0], these_params)
                print
                print
            else:
                print
                print abc[0]
                print
                print abc[1]
                print
    else:
        abc = summarise_dose_finding_sims(sims, label, num_doses)
        print abc[0]
        print
        print abc[1]
        print
