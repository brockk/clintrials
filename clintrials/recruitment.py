__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'


import abc
import copy
import numpy as np


""" Classes and functions for modelling recruitment to clinical trials. """


class RecruitmentStream(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def reset(self):
        """ Reset the recruitment stream to start anew.

        :return: None
        :rtype: None

        """
        pass

    @abc.abstractmethod
    def next(self):
        """ Get the time that the next patient is recruited.

        :return: The time that the next patient is recruited.
        :rtype: float

        """
        pass


class ConstantRecruitmentStream(RecruitmentStream):
    """ Recruitment stream where the intrapatient wait is constant.

    This is the simplest recruitment stream case. A patient arrives every delta units of time.

    E.g.

    >>> s = ConstantRecruitmentStream(2.5)
    >>> s.next()
    2.5
    >>> s.next()
    5.0
    >>> s.next()
    7.5
    >>> s.reset()
    >>> s.next()
    2.5


    """

    def __init__(self, intrapatient_gap):
        """ Create instance

        :param intrapatient_gap: the gap between recruitment times, aka delta.
        :type intrapatient_gap: float

        """

        self.delta = intrapatient_gap
        self.cursor = 0

    def reset(self):
        """ Reset the recruitment stream to start anew.

        :return: None
        :rtype: None

        """

        self.cursor = 0

    def next(self):
        """ Get the time that the next patient is recruited.

        :return: The time that the next patient is recruited.
        :rtype: float

        """
        self.cursor += self.delta
        return self.cursor


class QuadrilateralRecruitmentStream(RecruitmentStream):
    """ Recruitment stream that allows recruitment potential to vary as a function of time using vertices.
    Between two vertices, recruitment potential is represented by areas of quadrilaterals. Recruitment potential
    may change linearly using interpolation, or instantananeously using steps. In the former case, the quadrilaterals
    are trapeziums; in the latter, rectangles.

    I started by calling this class DampenedRecruitmentStream because recruitment typically opens at something
    like 50% potency where half recruitment centres are open and then increases linearly to 100% after about a year.
    However, I settled on the name QuadrilateralRecruitmentStream because of the important role quadrilaterals play in
    calculating the cumulative recruitment mass between two times.

    Let's do an example. Imagine a hypothetical trial that will recruit using several recruitment centres. When all
    recruitment centres are open, the trial expects to recruit a patient every four days, thus the intrapatient gap
    is 4.0. The trial will open with initial recruitment potential of 50% (i.e. half of the recruiting sites are open).
    Recruitment potential is expected to reach 100% after 20 days, linearly increasing from 50% to 100% over the first
    20 days, i.e. recruitment centres will be continually opened at a constant rate. The first patient will be recruited
    at time t where t satisfies the integral equation

    :math:`\\int_0^t 0.5 + \\frac{1.0 - 0.5}{20 - 0}s  ds = \\int_0^t 0.5 + \\frac{s}{40} ds
    = \\frac{t}{2} + \\frac{t^2}{80} = 4`

    i.e. solving the quadratic

    :math:`t = \\frac{-\\frac{1}{2} + \\sqrt{\\frac{1}{2}^2 - 4 \\times \\frac{1}{80} \\times -4}}{\\frac{2}{80}}
    = 6.83282`

    , and so on. The root of the quadratic yielded by :math:`-b - \\sqrt{b^2-4ac}` is ignored because it makes no sense.

    E.g.

    >>> s1 = QuadrilateralRecruitmentStream(4.0, 0.5, [(20, 1.0)], interpolate=True)
    >>> s1.next()
    6.8328157299974768
    >>> s1.next()
    12.2490309931942
    >>> s1.next()
    16.878177829171548
    >>> s1.next()
    21.0
    >>> s1.next()
    25.0

    Now, let's consider the same scenario again, with stepped transition rather than interpolated transition. In this
    scenario, a patient is recruited after each 4 / 0.5 = 8 days for times from 0 to 20 when recruitment potential is
    at 50%. After time=20, a patient is recruited after every 4 days because recruitment potential is at 100%. For the
    patient that straddles the time t=20, the time to recruit is 4 days at 50% potential plus 2 days at 100% = 4 days,
    as required.

    E.g.

    >>> s2 = QuadrilateralRecruitmentStream(4.0, 0.5, [(20, 1.0)], interpolate=False)
    >>> s2.next()
    8.0
    >>> s2.next()
    16.0
    >>> s2.next()
    22.0
    >>> s2.next()
    26.0

    """

    def __init__(self, intrapatient_gap, initial_intensity, vertices, interpolate=True):
        """ Create instance

        :param intrapatient_gap: time to recruit one patient at 100% recruitment intensity, i.e. the gap between
                                    recruitment times when recruitment is at 100% intensity.
        :type intrapatient_gap: float
        :param initial_intensity: recruitment commences at this % of total power.
                                    E.g. if it takes 2 days to recruit a patient at full recruitment power,
                                            at intensity 0.1 it will take 20 days to recruit a patient.
                                    TODO: zero? negative?
        :type initial_intensity: float
        :param vertices: list of additional vertices as (time t, intensity r) tuples, where recruitment power is r% at t
                        Recruitment intensity is linearly extrapolated between vertex times, including the origin, t=0.
                        .. note::
                        - intensity can dampen (e.g. intensity=50%) or amplify (e.g. intensity=150%) average recruitment;
                        - intensity should not be negative. Any negative values will yield a TypeError
        :type vertices: list of (float, float) tuples
        :param interpolate: True to linearly interpolate between vertices; False to use steps.
        :type interpolate: bool

        """

        self.delta = intrapatient_gap
        self.initial_intensity = initial_intensity
        self.interpolate = interpolate

        v = vertices
        v.sort(key=lambda x: x[0])
        self.shapes = {}  # t1 -> t0, t1, y0, y1 vertex parameters
        self.recruiment_mass = {}  # t1 -> recruitment mass available (i.e. area of quadrilateral) to left of t1
        if len(v) > 0:
            t0 = 0
            y0 = initial_intensity
            for x in v:
                t1, y1 = x
                if interpolate:
                    mass = 0.5 * (t1-t0) * (y0+y1)  # Area of trapezium
                else:
                    mass = (t1-t0) * y0  # Are of rectangle
                self.recruiment_mass[t1] = mass
                self.shapes[t1] = (t0, t1, y0, y1)
                t0, y0 = t1, y1
            self.available_mass = copy.copy(self.recruiment_mass)
        else:
            self.available_mass = {}
        self.vertices = v
        self.cursor = 0

    def reset(self):
        """ Reset the recruitment stream to start anew.

        :return: None
        :rtype: None

        """

        self.cursor = 0
        self.available_mass = copy.copy(self.recruiment_mass)

    def next(self):
        """ Get the time that the next patient is recruited.

        :return: The time that the next patient is recruited.
        :rtype: float

        """

        sought_mass = self.delta
        t = sorted(self.available_mass.keys())
        for t1 in t:
            avail_mass = self.available_mass[t1]
            t0, _, y0, y1 = self.shapes[t1]
            if avail_mass >= sought_mass:
                if self.interpolate:
                    y_at_cursor = self._linearly_interpolate_y(self.cursor, t0, t1, y0, y1)
                    new_cursor = self._invert(self.cursor, t1, y_at_cursor, y1, sought_mass)
                    self.cursor = new_cursor
                else:
                    y_at_cursor = y0
                    new_cursor = self._invert(self.cursor, t1, y_at_cursor, y1, sought_mass, as_rectangle=True)
                    self.cursor = new_cursor

                self.available_mass[t1] -= sought_mass
                return self.cursor
            else:
                sought_mass -= avail_mass
                self.available_mass[t1] = 0.0
                if t1 > self.cursor:
                    self.cursor = t1

        # Got here? Satisfy outstanding sought mass using terminal recruitment intensity
        terminal_rate = y1 if len(self.vertices) else self.initial_intensity
        if terminal_rate > 0:
            self.cursor += sought_mass / terminal_rate
            return self.cursor
        else:
            return np.nan

    def _linearly_interpolate_y(self, t, t0, t1, y0, y1):
        """ Linearly interpolate y-value at t using line through (t0, y0) and (t1, y1) """
        if t1 == t0:
            # The line either has infiniite gradient or is not a line at all, but a point. No logical response
            return np.nan
        else:
            m = (y1-y0) / (t1-t0)
            return y0 + m * (t-t0)

    def _invert(self, t0, t1, y0, y1, mass, as_rectangle=False):
        """ Returns time t at which the area of quadrilateral with vertices at t0, t, f(t), f(t0) equals mass. """
        if t1 == t0:
            # The quadrilateral has no area
            return np.nan
        elif y0 == y1 and y0 <= 0:
            # The quadrilateral has no area or is badly defined
            return np.nan
        elif (y0 == y1 and y0 > 0) or as_rectangle:
            # We require area of a rectangle; easy!
            return t0 + 1.0 * mass / y0
        else:
            # We require area of a trapezium. That requires solving a quadratic.
            m = (y1-y0) / (t1-t0)
            discriminant = y0**2 + 2 * m * mass
            if discriminant < 0:
                raise TypeError('Discriminant is negative')
            z = np.sqrt(discriminant)
            tau0 = (-y0 + z) / m
            tau1 = (-y0 - z) / m
            if tau0 + t0 > 0:
                return t0 + tau0
            else:
                assert(t0 + tau1 > 0)
                return t0 + tau1