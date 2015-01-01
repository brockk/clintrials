__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'

""" Functions and classes for manipulating collections. """


def to_1d_list_gen(x):
    """ Generator function to reduce lists of lists of arbitrary depth (and scalars) to single depth-1 list.

    .. note:: this function is recursive.

    """

    if isinstance(x, list):
        for y in x:
            for z in to_1d_list_gen(y):
                yield z
    else:
        yield x


def to_1d_list(x):
    """ Reshape scalars, lists and lists of lists of arbitrary depth as a single flat list, i.e. list of depth 1.

    .. note:: this function basically offloads all its work to a generator function because **we like yield**!

    E.g.

    >>> to_1d_list(0)
    [0]
    >>> to_1d_list([1])
    [1]
    >>> to_1d_list([[1,2],3,[4,5]])
    [1, 2, 3, 4, 5]
    >>> to_1d_list([[1,2],3,[4,5,[6,[7,8,[9]]]]])
    [1, 2, 3, 4, 5, 6, 7, 8, 9]

    """
    return list(to_1d_list_gen(x))