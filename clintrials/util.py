__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'

""" This module provides a home for all those useful bits-and-bobs that do not warrant their own module. """


from collections import OrderedDict, Iterable
from copy import copy
from datetime import datetime
from itertools import product
import numpy as np
import json
# import pandas as pd
# from statsmodels.stats.proportion import proportion_confint

from clintrials.coll import to_1d_list_gen, to_1d_list


def fullname(o):
    """ Get the fully-qualified class name of an object

    :param o: object of any kind
    :type o: object
    :return: fully-qualified class name
    :rtype: string

    """

    return o.__module__ + "." + o.__class__.__name__


def atomic_to_json(obj):
    """ Wrapper to ensure the smallest object is in a JSON-friendly form.

    .. note:: This function exists because numpy types raise errors when you try to JSON save them. Don't believe me?
                Try ``json.dumps(np.int(1))`` and get something about 1 not being serializable.

    :param obj: Object to convert to JSON-able form
    :type obj: object
    :return: obj, or its scalar equivalent if obj is a numpy generic type
    :rtype: object

    """

    if isinstance(obj, np.generic):
        return np.asscalar(obj)
    else:
        return obj


def iterable_to_json(obj):
    """ Returns a list of JSON-friendly representations of the objects in a collection.

    :param obj: JSON-ise each element of this collection
    :type obj: iterable
    :return: congruent list of JSON-able objects
    :rtype: list

    """
    if isinstance(obj, Iterable):
        return [atomic_to_json(x) for x in obj]
    else:
        return atomic_to_json(obj)


def row_to_json(row, **kwargs):
    """ Turn a pandas.Series to a JSON object

    :param row: a row to turn to JSON-able dict
    :type row: pandas.Series
    :param kwargs: map of keyword args to pass to json.loads
    :type kwargs: dict
    :return: a JSON-friendly dict representation of row
    :rtype: dict

    """

    try:
        doc = json.loads(row.to_json(), **kwargs)
    except UnicodeDecodeError:
        # iso-8859-1 has solved this before; but might not solve all ills.
        return row_to_json(row, encoding='iso-8859-1')
    # to_json turns all dates to long as 'ticks after epoch',
    # regardless of params passed (bug?) so cast dates to isoformat manually:
    # n.b. only actual dates can be cast to an isoformat string so screen null dates.
    import pandas as pd
    for x in row.index:
        if isinstance(row[x], datetime) and not pd.isnull(row[x]):
            doc[x] = pd.to_datetime(row[x]).date().isoformat()
    return doc


def df_to_json(df, do_value_counts=True, definitely_do_value_counts=False,
               do_column_summaries=True, do_row_summaries=True):
    """ Serialize a pandas.DataFrame to an object that may be written as JSON.

    .. note:: pandas.DataFrame provides its own JSON serialization method but I don't like it.

    :param df: DataFrame to serialise to JSON-able form.
    :type df: pandas.DataFrame
    :param do_value_counts: True to calculate value counts for each column
    :type do_value_counts: bool
    :param definitely_do_value_counts: If there is no aggregation possible (i.e. all elements are unique), aggregation
                                        will be suppressed. Use True to override this suppression.
    :type definitely_do_value_counts: bool
    :param do_column_summaries: True to calculate summary statistics for each column
    :type do_column_summaries: bool
    :param do_row_summaries: True to calculate summary statistics for each row
    :type do_row_summaries: bool
    :return: A JSON-able representation of df
    :rtype: dict

    """

    doc = OrderedDict()
    doc['Format'] = 'Table'

    rows = []
    for i, row_name in enumerate(df.index):
        rows.append(OrderedDict([('ID', str(i)), ('Position', i+1), ('Label', atomic_to_json(row_name))]))
    doc['Rows'] = rows
    doc['NumRows'] = len(df.index)

    cols = []
    for i, col_name in enumerate(df):
        cols.append(OrderedDict([('ID', str(i)), ('Position', i+1), ('Label', atomic_to_json(col_name))]))
    doc['Cols'] = cols
    doc['NumCols'] = len(df.columns)

    table_data = OrderedDict()
    for j, col_name in enumerate(df):
        col_data = OrderedDict()
        for i, o in enumerate(df[col_name]):
            col_data[i] = atomic_to_json(o)
        table_data[str(j)] = col_data
    doc['Data'] = table_data

    if do_value_counts:
        freqs = OrderedDict()
        for col_name in df:
            vc = df[col_name].value_counts()
            if len(vc) < len(df) or definitely_do_value_counts:
                freqs[atomic_to_json(col_name)] = dict([(atomic_to_json(k), atomic_to_json(v)) for k, v in vc.iteritems()])
        doc['Frequencies'] = freqs

    if do_column_summaries:
        col_summaries = OrderedDict()
        for i, col_name in enumerate(df):
            col_summary = OrderedDict()
            try:
                col_summary['Mean'] = df[col_name].mean()
            except:
                pass
            try:
                col_summary['Sum'] = df[col_name].sum()
            except:
                pass
            col_summaries[str(i)] = col_summary
        doc['ColumnSummary'] = col_summaries

    if do_row_summaries:
        row_summaries = OrderedDict()
        for i, row_name in enumerate(df.index):
            row_summary = OrderedDict()
            try:
                row_summary['Sum'] = df.loc[row_name].sum()
            except:
                pass
            row_summaries[str(i)] = row_summary
        doc['RowSummary'] = row_summaries

    return doc


def levenshtein(s1, s2):
    """ How 'far' is string s1 from s2? Calculate the Levenshtein distance between two strings.

    See http://en.wikipedia.org/wiki/Levenshtein_distance

    :param s1: first string
    :type s1: string
    :param s2: second string
    :type s2: string
    :return: the Levenshtein distance
    :rtype: int

    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = xrange(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1  # j+1 instead of j since previous_row and current_row
                                                  # are one character longer
            deletions = current_row[j] + 1        # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def levenshtein_index(s1, s2):
    """ Returns similarity score for strings s1 and s2 between 0 and 1 by dividing levenshtein score by greatest length.

    This method uses :func:`clintrials.util.levenshtein`.

    :param s1: first string
    :type s1: string
    :param s2: second string
    :type s2: string
    :return: Similarity index between 0 and 1
    :rtype: float

    """

    l = levenshtein(s1, s2)
    max_length = max(len(s1), len(s2))
    if max_length:
        return 1 - 1. * l / max_length
    else:
        return 0.0


def support_match(a, b):
    """ Percentage score showing % of elements of a in b and b in a.

    :param a: collection 1
    :type a: iterable
    :param b: collection 2
    :type b: iterable
    :return: Match score from 0.0 to 1.0
    :rtype: float

    """

    try:
        a_set = set(a)
        b_set = set(b)
        a_in_b = [x in b_set for x in a_set]
        b_in_a = [x in a_set for x in b_set]
        return 1.0 * (sum(a_in_b) + sum(b_in_a)) / (len(a_set) + len(b_set))
    except:
        return 0.0


def _correlated_binary_outcomes_mardia(a, b, c):
    """ Helper function to correlated_binary_outcomes """
    if a == 0:
        return -c / b

    if b > 0:
        k = 1
    elif b < 0:
        k = -1
    else:
        k = 0
    p = -0.5 * (b + k * np.sqrt(b**2 - 4*a*c))
    r1 = 1.* p / a
    r2 = 1.* c / p
    r = r2 if r2 > 0 else r1
    return r


def _correlated_binary_outcomes_solve2(mui, muj, psi):
    if psi == 1:
        return mui*muj
    else:
        a = 1 - psi
        b = 1 - a * (mui + muj)
        c = -psi * (mui * muj)
        muij = _correlated_binary_outcomes_mardia(a,b,c)
    return muij


def correlated_binary_outcomes(num_pairs, u, psi, seed=None):
    """ Randomly sample correlated binary digits, copying the method from R-package ranBin2.

    :param num_pairs: number of pairs
    :type num_pairs: int
    :param u: 2-item list/tuple of event probabilities
    :type u: list or tuple
    :param psi: odds ratio of the binary outcomes
    :type psi: float
    :param seed: optional seed for reproducible randomness
    :type seed: int
    :return: ndarray of paired binary digits, 2 columns and num_pairs rows
    :rtype: numpy.ndarray

    .. note:: The Bonett article at http://psych.colorado.edu/~willcutt/pdfs/Bonett_2007.pdf
                details Yule's method (1912) for estimating correlation from odds ratio and vice-versa.
                If the two proportions in u are close, r = (sqrt(OR) - 1) / (sqrt(OR) + 1), and
                OR = ((1+r) / (1-r))**2
                provide decent approximations.

    """
    if seed:
        np.random.seed(seed)

    u12 = _correlated_binary_outcomes_solve2(u[0], u[1], psi)
    y = -1 * np.ones(shape=(num_pairs, 2))
    y[:, 0] = (np.random.uniform(size=num_pairs) < u[0]).astype(int)
    y[:, 1] = y[:, 0] * (np.random.uniform(size=num_pairs) <= u12/u[0]) + (1-y[:, 0]) * \
                        (np.random.uniform(size=num_pairs) <= (u[1]-u12)/(1-u[0]))
    return y


def correlated_binary_outcomes_from_uniforms(unifs, u, psi):
    """ Create correlated binary outcomes from observed n*3 array of uniforms, tweaking method from R-package ranBin2.

    :param unifs: array of shape (n, 3) of uniforms between 0 and 1
    :type unifs: numpy.ndarray
    :param u: 2-item list/tuple of event probabilities
    :type u: list or tuple
    :param psi: odds ratio of the binary outcomes
    :type psi: float
    :return: ndarray of paired binary digits, 2 columns and num_pairs rows
    :rtype: numpy.ndarray

    .. note:: The Bonett article at http://psych.colorado.edu/~willcutt/pdfs/Bonett_2007.pdf
                details Yule's method (1912) for estimating correlation from odds ratio and vice-versa.
                If the two proportions in u are close, r = (sqrt(OR) - 1) / (sqrt(OR) + 1), and
                OR = ((1+r) / (1-r))**2
                provide decent approximations.

    """

    if unifs.ndim == 2 and unifs.shape[1] == 3:
        u12 = _correlated_binary_outcomes_solve2(u[0], u[1], psi)
        n = unifs.shape[0]
        y = -1 * np.ones(shape=(n, 2))
        y[:, 0] = (unifs[:, 0] < u[0]).astype(int)
        y[:, 1] = y[:, 0] * (unifs[:, 1] <= u12/u[0]) + (1-y[:, 0]) * (unifs[:, 2] <= (u[1]-u12)/(1-u[0]))
        return y
    else:
        raise ValueError('unifs must be an n*3 array')


def get_proportion_confint_report(num_successes, num_trials, alpha=0.05, do_normal=True, do_agresti_coull=True,
                                  do_beta=False, do_wilson=True, do_jeffrey=False, do_binom_test=False):
    """ Get confidence intervals of proportion num_successes / num_trials using different methods in JSON-friendly form.

    :param num_successes: number of successes
    :type num_successes: int
    :param num_trials: number of trials or attempts
    :type num_trials: int
    :param alpha: significance used in statistical inferences
    :type alpha: float
    :param do_normal: True to get a confidence interval using the normal approximation method
    :type do_normal: bool
    :param do_agresti_coull: True to get a confidence interval using the Agresti-Coull method
    :type do_agresti_coull: bool
    :param do_beta: True to get a confidence interval using the beta method
    :type do_beta: bool
    :param do_wilson: True to get a confidence interval using the Wilson method
    :type do_wilson: bool
    :param do_jeffrey: True to get a confidence interval using the Jeffrey method
    :type do_jeffrey: bool
    :param do_binom_test: True to get a confidence interval using the binomial test method
    :type do_binom_test: bool
    :return: JSON-able dict report
    :rtype: dict

    Why do I use normal, agresti_coull and wilson by default?

    1) Normal, because it is the widely-used but flawed option.
    2) AgrestiCoull & Wilson, because Lawrence D. Brown, T. Tony Cai and Anirban DasGupta in their paper
        `Interval Estimation for a Binomial Proportion` say 'we recommend the Wilson interval for small n and the
        interval suggested in Agresti and Coull for larger n'

    Call to proportion_confint allows methods:

    - `normal` : asymptotic normal approximation
    - `agresti_coull` : Agresti-Coull interval
    - `beta` : Clopper-Pearson interval based on Beta distribution
    - `wilson` : Wilson Score interval
    - `jeffrey` : Jeffrey's Bayesian Interval
    - `binom_test`

    """

    from statsmodels.stats.proportion import proportion_confint

    conf_int_reports = OrderedDict()

    if do_normal:
        conf_int = proportion_confint(num_successes, num_trials, alpha=alpha, method='normal')
        conf_int_report = OrderedDict()
        conf_int_report['Lower'] = conf_int[0]
        conf_int_report['Upper'] = conf_int[1]
        conf_int_report['Alpha'] = alpha
        conf_int_report['Method'] = 'Normal'
        conf_int_reports['Normal'] = conf_int_report

    if do_agresti_coull:
        conf_int = proportion_confint(num_successes, num_trials, alpha=alpha, method='agresti_coull')
        conf_int_report = OrderedDict()
        conf_int_report['Lower'] = conf_int[0]
        conf_int_report['Upper'] = conf_int[1]
        conf_int_report['Alpha'] = alpha
        conf_int_report['Method'] = 'AgrestiCoull'
        conf_int_reports['AgrestiCoull'] = conf_int_report

    if do_beta:
        conf_int = proportion_confint(num_successes, num_trials, alpha=alpha, method='beta')
        conf_int_report = OrderedDict()
        conf_int_report['Lower'] = conf_int[0]
        conf_int_report['Upper'] = conf_int[1]
        conf_int_report['Alpha'] = alpha
        conf_int_report['Method'] = 'Beta'
        conf_int_reports['Beta'] = conf_int_report

    if do_wilson:
        conf_int = proportion_confint(num_successes, num_trials, alpha=alpha, method='wilson')
        conf_int_report = OrderedDict()
        conf_int_report['Lower'] = conf_int[0]
        conf_int_report['Upper'] = conf_int[1]
        conf_int_report['Alpha'] = alpha
        conf_int_report['Method'] = 'Wilson'
        conf_int_reports['Wilson'] = conf_int_report

    if do_jeffrey:
        conf_int = proportion_confint(num_successes, num_trials, alpha=alpha, method='jeffrey')
        conf_int_report = OrderedDict()
        conf_int_report['Lower'] = conf_int[0]
        conf_int_report['Upper'] = conf_int[1]
        conf_int_report['Alpha'] = alpha
        conf_int_report['Method'] = 'Jeffrey'
        conf_int_reports['Jeffrey'] = conf_int_report

    if do_binom_test:
        conf_int = proportion_confint(num_successes, num_trials, alpha=alpha, method='binom_test')
        conf_int_report = OrderedDict()
        conf_int_report['Lower'] = conf_int[0]
        conf_int_report['Upper'] = conf_int[1]
        conf_int_report['Alpha'] = alpha
        conf_int_report['Method'] = 'BinomTest'
        conf_int_reports['BinomTest'] = conf_int_report

    return conf_int_reports


def cross_tab(col_row_pairs, cols=None, rows=None, to_json=False, do_value_counts=False):
    """ Cross-tabulate counts of data pairs.

    :param col_row_pairs: list of 2-tuples, (col item, row item), e.g.
                            [('1', 'Related'), ('2', 'Unrelated'), ('2', 'Related')]
    :type col_row_pairs: list
    :param cols: list of col-headers. Distinct items will be used if omitted.
    :type cols: list
    :param rows: list of row-headers. Distinct items will be used if omitted and rows will be sorted by row-wise totals.
    :type rows: list
    :param to_json: True to return JSON-able object; False to get a pandas.DataFrame
    :type to_json: bool
    :param do_value_counts: True to return value counts
    :type do_value_counts: bool
    :return: pivottable-style cross tabulation
    :rtype: dict of pandas.DataFrame

    """

    col_data, row_data = zip(*col_row_pairs)
    row_h = rows if rows else list(set(row_data))
    col_h = cols if cols else list(set(col_data))
    counts = np.zeros((len(row_h), len(col_h)))
    for i, r in enumerate(row_h):
        for j, c in enumerate(col_h):
            n = sum(np.array([x == c for x in col_data]) & np.array([x == r for x in row_data]))
            counts[i, j] = n
    import pandas as pd
    df_n = pd.DataFrame(counts, index=row_h, columns=col_h)

    if not rows:
        row_order = np.argsort(-df_n.sum(axis=1).values)
        df_n = df_n.iloc[row_order]

    if to_json:
        return df_to_json(df_n, do_value_counts=do_value_counts)
    else:
        return df_n


class Memoize:
    """ Class to transparently cache function results by their runtime args

    E.g.

    >>> f = lambda x: x**3
    >>> f = Memoize(f)
    >>> f(2.0) # Result is calculated and cached
    8.0
    >>> f(2.0) # Result is fetched from cache
    8.0

    """

    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]


class ParameterSpace:
    """ Class to handle combinations of parameters (i.e. a parameter space) in simulations. """

    def __init__(self):
        self.vals_map = OrderedDict()

    def add(self, label, values):
        """ Add a variable and a list of all values the variable may take.

        :param label: variable label or name
        :type label: str
        :param values: list of values that variable may take
        :type values: list

        """

        self.vals_map[label] = values

    def sample(self, label):
        """ Randomly fetch a value for variable with label.

        :param label: variable label or name
        :type label: str
        :return: randomly-sampled value
        :rtype: object

        """

        if label in self.vals_map:
            vals = self.vals_map[label]
            return vals[np.random.choice(range(len(vals)))]
        else:
            return None

    def sample_all(self):
        """ Randomly sample a value for each variable, returned as a map from label to value.

        :return: a randomly sampled set of parameter values
        :rtype: dict

        """

        sampled = {}
        for label in self.vals_map:
            sampled[label] = self.sample(label)
        return sampled

    def get_cyclical_iterator(self, limit=-1):
        """ Get iterator to **deterministically** cycle the possible parameter perumtations, optionally forever.

        :param limit: -1 to iterate cyclically forever, else maximum number of elements to iterate through.
        :type limit: int
        :return: an iterable object
        :rtype: iterable

        """

        return _ParameterSpaceIter(self, limit)

    def keys(self):
        """ Get parameter space keys, i.e. the variable names

        :return: Collection of keys
        :rtype: iterable

        """

        return self.vals_map.keys()

    def dimensions(self):
        """ Get the numbers of values per dimension.

        E.g. a param-space of two values for A and three values for B would return [2, 3]

        :return: Array of number of values per dimension
        :rtype: numpy.array

        """

        return np.array([len(y) for x,y in self.vals_map.iteritems()])

    def size(self):
        """ Get the size of this parameter space, i.e. the product of the dimension sizes.

        :return: Size of parameter space.
        :rtype: int

        """

        return np.prod(self.dimensions())

    def __getitem__(self, key):
        return self.vals_map[key]


class _ParameterSpaceIter():

        def __init__(self, parameter_space, limit):
            self.limit = limit
            self.cursor = 0
            self.vals_map = copy(parameter_space.vals_map)
            self.labels = self.vals_map.keys()
            num_options = []
            for label in self.labels:
                num_options.append(len(parameter_space[label]))
            self.paths = list(product(*[range(x) for x in num_options]))
            #print zip(labels, num_options)

        def __iter__(self):
            return self

        def next(self):
            if 0 < self.limit <= self.cursor:
                raise StopIteration()
            i = self.cursor % len(self.paths)
            path = self.paths[i]
            param_map = {}
            assert(len(path) == len(self.labels))
            for j, label in enumerate(self.labels):
                param_map[label] = self.vals_map[label][path[j]]
            self.cursor += 1
            return param_map





if __name__ == "__main__":
    import doctest
    doctest.testmod()

