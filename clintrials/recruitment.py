__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'


import copy
import numpy as np


class ConstantRecruitmentStream:
    """ Recruitment stream where intrapatient wait is constant. Simplest recruitment stream case. """

    def __init__(self, intrapatient_gap):
        """
        Params:
        intrapatient_gap, the gap between recruitment times.

        """

        self.delta = intrapatient_gap
        self.cursor = 0

    def reset(self):
        self.cursor = 0

    def next(self):
        self.cursor += self.delta
        return self.cursor


class QuadrilateralRecruitmentStream:
    """ Recruitment stream that allows recruitment potential to vary as a function of time using vertices.
    Between two vertices, recruitment potential is represented by areas of quadrilaterals. Recruitment potential
    may change linearly using interpolation, or instantananeously using steps. In the former case, the quadrilaterals
    are trapeziums; in the latter, rectangles.

    I started by calling this class DampenedRecruitmentStream because recruitment typically opens at something
    like 50% potency where half recruitment centres are open and then increases linearly to 100% after about a year.
    However, I settled on the name QuadrilateralRecruitmentStream because of the important role quadrilaterals play in
    calculating the cumulative recruitment mass between two times.

    """

    def __init__(self, intrapatient_gap, initial_intensity, vertices, interpolate=True):
        """
        Params:
        intrapatient_gap, time to recruit one patient at 100% recruitment intensity,
                            i.e. the gap between recruitment times when recruitment is at 100% intensity.
        initial_intensity, recruitment commences at this % of total power.
                            E.g. if it takes 2 days to recruit a patient at full recruitment power,
                            at intensity 0.1 it will take 20 days to recruit a patient.
                            TODO: zero? negative?
        vertices, list of additional vertices as (time t, intensity r) tuples, where recruitment power is r% at t.
                    Recruitment intensity is linearly extrapolated between vertex times, including the origin, t=0.
                    Note:
                    - intensity can dampen (e.g. intensity=50%) or amplify (e.g. intensity=150%) average recruitment;
                    - intensity should not be negative. Any negative values will yield a TypeError
        interpolate, True to linearly interpolate between vertices; False to use steps

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
        self.cursor = 0
        self.available_mass = copy.copy(self.recruiment_mass)

    def next(self):
        sought_mass = self.delta
        t = sorted(self.available_mass.keys())
        for t1 in t:
            avail_mass = self.available_mass[t1]
            t0, _, y0, y1 = self.shapes[t1]
            # print 'Using shape between', (t0,t1), 'and seeking', sought_mass, ',', avail_mass, 'is avail'
            if avail_mass >= sought_mass:
                if self.interpolate:
                    y_at_cursor = self._linearly_interpolate_y(self.cursor, t0, t1, y0, y1)
                    new_cursor = self._invert(self.cursor, t1, y_at_cursor, y1, sought_mass)
                    # print 'Set cursor to', new_cursor, 'after interpolation'
                    self.cursor = new_cursor
                else:
                    y_at_cursor = y0
                    new_cursor = self._invert(self.cursor, t1, y_at_cursor, y1, sought_mass, as_rectangle=True)
                    # print 'Set cursor to', new_cursor, 'after rectangulation'
                    self.cursor = new_cursor

                self.available_mass[t1] -= sought_mass
                # print 'After shuffling along, cursor is at', self.cursor
                return self.cursor
            else: #if avail_mass > 0:
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
            # print 'Seeking', mass, 'at', y0, 'yields', t0, 'plus', 1.0 * mass / y0, 'equals', t0 + 1.0 * mass / y0
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