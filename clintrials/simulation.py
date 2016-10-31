__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'


from collections import OrderedDict
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
                print('Error writing: %s' % e)
        print('{} {} {}'.format(j, datetime.now(), len(sims)))
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
                print('Error writing: %s' % e)
        print('{} {} {}'.format(j, datetime.now(), len(sims)))
    return sims


def _open_json_local(file_loc):
    return json.load(open(file_loc, 'r'))


def _open_json_url(url):
    try:
        from urllib2 import urlopen
    except:
        from urllib import urlopen
    return json.load(urlopen(url))


def go_fetch_json_sims(file_pattern):
    files = glob.glob(file_pattern)
    sims = []
    for f in files:
        sub_sims = _open_json_local(f)
        print('{} {}'.format(f, len(sub_sims)))
        sims += sub_sims
    print('Fetched %s sims' % len(sims))
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


# Map-Reduce methods for summarising sims in memory-efficient ways
def map_reduce_files(files, map_func, reduce_func):
    """
    Invoke map_func on each file in sim_files and reduce results using reduce_func.

    :param files: list of files that contain simulations
    :type files: list
    :param map_func:function to create summary content for object x
    :type map_func: function
    :param reduce_func: function to reduce summary content of objects x & y
    :type reduce_func: function

    :returns: ?
    :rtype: ?

    """
    if len(files):
        x = map(map_func, files)
        return reduce(reduce_func, x)
    else:
        raise TypeError('No files')


def invoke_map_reduce_function_map(sims, function_map):
    """ Invokes map/reduce pattern for many items on a list of simulations.
    Functions are specified as "item name" -> (map_func, reduce_func) pairs in function_map.
    In each iteration, map_func is invoked on sims, and then reduce_func is invoked on result.
    As usual, map_func takes iterable as single argument and reduce_func takes x and y as args.

    Returns a dict with keys function_map.keys() and values the result of reduce_func
    """

    response = OrderedDict()
    for item, function_tuple in function_map.iteritems():
        map_func, reduce_func = function_tuple
        x = reduce(reduce_func, map(map_func, sims))
        response[item] = x

    return response


def reduce_maps_by_summing(x, y):
    """ Reduces maps x and y by adding the value of every item in x to matching value in y.

    :param x: first map
    :type x: dict
    :param y: second map
    :type y: dict
    :returns: map of summed values
    :rtype: dict

    """

    response = OrderedDict()
    for k in x.keys():
        response[k] = x[k] + y[k]
    return response


# I wrote the functions below during a specific analysis.
# TODO: Do they make sense in a general package?
def partition_and_aggregate(sims, ps, function_map):
    """ Function partitions simulations into subsets that used the same set of parameters,
    and then invokes a collection of map/reduce function pairs on each subset.

    :param sims: list of simulations (probably in JSON format)
    :type sims: list
    :param ps: ParameterSpace that will explain how to filter simulations
    :type ps: ParameterSpace
    :param function_map: map of item -> (map_func, reduce_func) pairs
    :type function_map: dict

    :returns: map of parameter combination to reduced object
    :rtype: dict

    """

    var_names = ps.keys()
    z = [(var_name, ps[var_name]) for var_name in var_names]
    labels, val_arrays = zip(*z)
    param_combinations = list(itertools.product(*val_arrays))
    out = OrderedDict()
    for param_combo in param_combinations:

        these_params = dict(zip(labels, param_combo))
        these_sims = filter_sims(sims, these_params)

        out[param_combo] =  invoke_map_reduce_function_map(these_sims, function_map)

    return out


def fetch_partition_and_aggregate(f, ps, function_map, verbose=False):
    """ Function loads JSON sims in file f and then hands off to partition_and_aggregate.

    :param f: file location
    :type f: str
    :param ps: ParameterSpace that will explain how to filter simulations
    :type ps: ParameterSpace
    :param function_map: map of item -> (map_func, reduce_func) pairs
    :type function_map: dict

    :returns: map of parameter combination to reduced object
    :rtype: dict

    """

    sims = _open_json_local(f)
    if verbose:
        print('Fetched {} sims from {}'.format(len(sims), f))
    return partition_and_aggregate(sims, ps, function_map)


def reduce_product_of_two_files_by_summing(x, y):
    """ Reduce the summaries of two files by summing. """
    response = OrderedDict()
    for k in x.keys():
        response[k] = reduce_maps_by_summing(x[k], y[k])
    return response


def multiindex_dataframe_from_tuple_map(x, labels):
    """ Create pandas.DataFrame from map of param-tuple -> value

    :param x: map of parameter-tuple -> value pairs
    :type x: dict
    :param labels: list of item labels
    :type labels: list
    :returns: DataFrame object
    :rtype: pandas.DataFrame

    """
    import pandas as pd
    k, v = zip(*[(k, v) for (k, v) in x.iteritems()])
    i = pd.MultiIndex.from_tuples(k, names=labels)
    return pd.DataFrame(list(v), index=i)
