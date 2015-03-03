__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'


from datetime import datetime
from itertools import product
import json
import pandas as pd


def sim_parameter_space(sim_func, ps, n1=10, n2=10, out_file=None):
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
        n1 * n2 simualtions are performed, in all.
        sim_func is expected to return a JSON-able object

    """

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


def filter_sims(sims, filter):
    """ Filter a list of simulations.

    :param sims: list of simulations (probably in JSON format)
    :type sims: list
    :param filter: map of item -> value pairs that forms the filter. Exact matches are retained.
    :type filter: dict

    """

    for key, val in filter.iteritems():
        if isinstance(val, tuple):
            # Mask tuples as lists because tuples are not JSON but lists are
            sims = [x for x in sims if x[key] == list(val)]
        else:
            sims = [x for x in sims if x[key] == val]
    return sims


def summarise_sims(sims, ps, var_map, func_map, to_pandas=True):
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

    var_names = var_map.keys()
    z = [(var_name, ps[var_map[var_name]]) for var_name in var_names]
    labels, val_arrays = zip(*z)
    param_combinations = list(product(*val_arrays))
    index_tuples = []
    row_tuples = []
    for param_combo in param_combinations:
        these_params = dict(zip(labels, param_combo))
        these_sims = filter_sims(sims, these_params)
        if len(these_sims):
            these_metrics = dict([(label, func(these_sims, these_params)) for label, func in func_map.iteritems()])
            index_tuples.append(param_combo)
            row_tuples.append(these_metrics)
    if to_pandas:
        return pd.DataFrame(row_tuples, pd.MultiIndex.from_tuples(index_tuples, names=var_names))
    else:
        # TODO
        return row_tuples, index_tuples