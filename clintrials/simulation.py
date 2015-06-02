__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'


from datetime import datetime
import glob
import itertools
import json


def run_sims(sim_func, n1=1, n2=1, out_file=None, **kwargs):
    """ Run simulations using a delegate function.

    :param sim_func: Delegate function to be called to yield single simulation.
    :type sim_func: func
    :param n1: Number of batches
    :type n1: int
    :param n2: Number of iterations per batch
    :type n2: int
    :param out_file: Location of file for incremental saving after completion of each batch.
    :type out_file: str
    :param kwargs: key-word args for sim_func
    :type kwargs: dict

    .. note::

        - n1 * n2 simualtions are performed, in all.
        - sim_func is expected to return a JSON-able object
        - file is saved after each of n1 iterations, where applicable.

    """

    sims = []
    for j in range(n1):
        sims1 = [sim_func(**kwargs) for i in range(n2)]
        sims += sims1
        if out_file:
            try:
                with open(out_file, 'w') as outfile:
                    json.dump(sims, outfile)
            except Exception as e:
                print 'Error writing:', e
        print j, datetime.now(), len(sims)
    return sims


def sim_parameter_space(sim_func, ps, n1=1, n2=None, out_file=None):
    """ Run simulations using a function and a ParameterSpace.

    :param sim_func: function to be called to yield single simulation. Parameters are provided via ps as unpacked kwargs
    :type sim_func: func
    :param ps: Parameter space to explore via simulation
    :type ps: clintrials.util.ParameterSpace
    :param n1: Number of batches
    :type n1: int
    :param n2: Number of iterations per batch
    :type n2: int
    :param out_file: Location of file for incremental saving after completion of each batch.
    :type out_file: str

    .. note::

        - n1 * n2 simualtions are performed, in all.
        - sim_func is expected to return a JSON-able object
        - file is saved after each of n1 iterations, where applicable.

    """

    if not n2 or n2 <= 0:
        n2 = ps.size()

    sims = []
    params_iterator = ps.get_cyclical_iterator()
    for j in range(n1):
        sims1 = [sim_func(**params_iterator.next()) for i in range(n2)]
        sims += sims1
        if out_file:
            try:
                with open(out_file, 'w') as outfile:
                    json.dump(sims, outfile)
            except Exception as e:
                print 'Error writing:', e
        print j, datetime.now(), len(sims)
    return sims


def _open_json_local(file_loc):
    return json.load(open(file_loc, 'r'))


def _open_json_url(url):
    return json.load(urllib2.urlopen(url))


def go_fetch_json_sims(file_pattern):
    files = glob.glob(file_pattern)
    sims = []
    for f in files:
        sub_sims = _open_json_local(f)
        print f, len(sub_sims)
        sims += sub_sims
    print 'Fetched', len(sims), 'sims'
    return sims

def filter_sims(sims, filter):
    """ Filter a list of simulations.

    :param sims: list of simulations (probably in JSON format)
    :type sims: list
    :param filter: map of item -> value pairs that forms the filter. Exact matches are retained.
    :type filter: dict

    """

    for key, val in filter.iteritems():
        # In JSON, tuples are masked as lists. In this filter, we treat them as equivalent:
        if isinstance(val, (tuple)):
            sims = [x for x in sims if x[key] == val or x[key] == list(val)]
        else:
            sims = [x for x in sims if x[key] == val]
    return sims


def summarise_sims(sims, ps, func_map, var_map=None, to_pandas=True):
    """ Summarise a list of simulations.

    Method partitions simulations into subsets that used the same set of parameters, and then invokes
    a collection of summary functions on each subset; outputs a pandas DataFrame with a multi-index.

    :param sims: list of simulations (probably in JSON format)
    :type sims: list
    :param ps: ParameterSpace that will explain how to filter simulations
    :type ps: ParameterSpace
    :param var_map: map from variable name in simulation JSON to arg name in ParameterSpace
    :type var_map: dict
    :param func_map: map from item name to function that takes list of sims and parameter map as args and returns
                        a summary statistic or object.
    :type func_map: dict
    :param to_pandas: True, to get a pandas.DataFrame; False, to get several lists
    :type to_pandas: bool

    """

    if var_map is None:
        var_names = ps.keys()
        var_map = {}
        for var_name in var_names:
            var_map[var_name] = var_name
    else:
        var_names = var_map.keys()

    z = [(var_name, ps[var_map[var_name]]) for var_name in var_names]
    labels, val_arrays = zip(*z)
    param_combinations = list(itertools.product(*val_arrays))
    index_tuples = []
    row_tuples = []
    for param_combo in param_combinations:
        these_params = dict(zip(labels, param_combo))
        these_sims = filter_sims(sims, these_params)
        if len(these_sims):
            these_metrics = dict([(label, func(these_sims, these_params)) for label, func in func_map.iteritems()])
            index_tuples.append(param_combo)
            row_tuples.append(these_metrics)
    if len(row_tuples):
        if to_pandas:
            import pandas as pd
            return pd.DataFrame(row_tuples, pd.MultiIndex.from_tuples(index_tuples, names=var_names))
        else:
            # TODO
            return row_tuples, index_tuples
    else:
        if to_pandas:
            import pandas as pd
            return pd.DataFrame(columns=func_map.keys())
        else:
            # TODO
            return [], []