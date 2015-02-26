__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'


from datetime import datetime
import json


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