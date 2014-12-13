__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'

from nose.tools import assert_almost_equal

from clintrials.recruitment import ConstantRecruitmentStream, QuadrilateralRecruitmentStream


def test_constant_recruitment_stream():

    s = ConstantRecruitmentStream(2)

    assert s.next() == 2
    assert s.next() == 4
    assert s.next() == 6
    s.reset()
    assert s.next() == 2


def test_quadrilateral_recruitment_stream_1():

    initial = 1.0
    vertices = [(90, 1.0)]
    s = QuadrilateralRecruitmentStream(15.0, initial, vertices)

    assert s.next() == 15.0
    assert s.next() == 30.0
    assert s.next() == 45.0
    assert s.next() == 60.0
    assert s.next() == 75.0
    assert s.next() == 90.0
    s.reset()
    assert s.next() == 15.0


def test_quadrilateral_recruitment_stream_2():

    initial = 1.0
    vertices = [(90, 1.0)]
    s = QuadrilateralRecruitmentStream(15.0, initial, vertices, interpolate=False)

    assert s.next() == 15.0
    assert s.next() == 30.0
    assert s.next() == 45.0
    assert s.next() == 60.0
    assert s.next() == 75.0
    assert s.next() == 90.0
    s.reset()
    assert s.next() == 15.0


def test_quadrilateral_recruitment_stream_3():

    initial = 0.5
    vertices = []
    s = QuadrilateralRecruitmentStream(10, initial, vertices)

    assert s.next() == 20.0
    assert s.next() == 40.0
    assert s.next() == 60.0
    s.reset()
    assert s.next() == 20.0


def test_quadrilateral_recruitment_stream_4():

    initial = 0.5
    vertices = []
    s = QuadrilateralRecruitmentStream(10, initial, vertices, interpolate=False)

    assert s.next() == 20.0
    assert s.next() == 40.0
    assert s.next() == 60.0
    s.reset()
    assert s.next() == 20.0


def test_quadrilateral_recruitment_stream_5():

    initial = 0.1
    vertices = [(90, 0.25), (180, 1), (150, 0.75)]
    s = QuadrilateralRecruitmentStream(5, initial, vertices)

    assert_almost_equal(s.next(), 37.979589711327129)
    assert_almost_equal(s.next(), 64.899959967967959)
    assert_almost_equal(s.next(), 86.969384566990684)
    assert_almost_equal(s.next(), 103.8178046004133)
    assert_almost_equal(s.next(), 115.85696017507577)
    assert_almost_equal(s.next(), 125.72670690061994)
    assert_almost_equal(s.next(), 134.29670248402687)
    assert_almost_equal(s.next(), 141.9756061276768)
    assert_almost_equal(s.next(), 148.99438184514796)
    assert_almost_equal(s.next(), 155.49869109050658)
    assert_almost_equal(s.next(), 161.58740079360237)
    assert_almost_equal(s.next(), 167.33126291998994)
    assert_almost_equal(s.next(), 172.78297743897352)
    assert_almost_equal(s.next(), 177.98304963002104)
    assert_almost_equal(s.next(), 183.0)
    assert_almost_equal(s.next(), 188.0)
    s.reset()
    assert_almost_equal(s.next(), 37.979589711327129)


def test_quadrilateral_recruitment_stream_6():

    initial = 0.1
    vertices = [(90, 0.25), (180, 1), (150, 0.75)]
    s = QuadrilateralRecruitmentStream(5, initial, vertices, interpolate=False)

    assert_almost_equal(s.next(), 50.0)
    assert_almost_equal(s.next(), 94.0)
    assert_almost_equal(s.next(), 114.0)
    assert_almost_equal(s.next(), 134.0)
    assert_almost_equal(s.next(), 151.33333333333334)
    assert_almost_equal(s.next(), 158.0)
    assert_almost_equal(s.next(), 164.66666666666666)
    assert_almost_equal(s.next(), 171.33333333333331)
    assert_almost_equal(s.next(), 177.99999999999997)
    assert_almost_equal(s.next(), 183.5)
    assert_almost_equal(s.next(), 188.5)
    s.reset()
    assert_almost_equal(s.next(), 50.0)


def test_quadrilateral_recruitment_stream_7():

    initial = 0.0
    vertices = [(100, 1.0)]
    s = QuadrilateralRecruitmentStream(10.0, initial, vertices)

    assert_almost_equal(s.next(), 44.721359549995789)
    assert_almost_equal(s.next(), 63.245553203367578)
    assert_almost_equal(s.next(), 77.459666924148337)
    assert_almost_equal(s.next(), 89.442719099991578)
    assert_almost_equal(s.next(), 99.999999999999986)
    assert_almost_equal(s.next(), 110.0)
    assert_almost_equal(s.next(), 120.)
    s.reset()
    assert_almost_equal(s.next(), 44.721359549995789)


def test_quadrilateral_recruitment_stream_8():

    initial = 0.0
    vertices=[(100, 1.0), (130, 0.0), (150, 0.5)]
    s = QuadrilateralRecruitmentStream(10.0, initial, vertices, interpolate=False)

    assert_almost_equal(s.next(), 110.0)
    assert_almost_equal(s.next(), 120.0)
    assert_almost_equal(s.next(), 130.0)
    assert_almost_equal(s.next(), 170.0)
    assert_almost_equal(s.next(), 190.0)
    s.reset()
    assert_almost_equal(s.next(), 110.0)


def test_quadrilateral_recruitment_stream_9():

    initial = 0.0
    vertices = [(100, 0.0), (200, 1.0), (250, 0.5)]
    s = QuadrilateralRecruitmentStream(10.0, initial, vertices)

    assert_almost_equal(s.next(), 144.72135954999578)
    assert_almost_equal(s.next(), 163.24555320336759)
    assert_almost_equal(s.next(), 177.45966692414834)
    assert_almost_equal(s.next(), 189.44271909999159)
    assert_almost_equal(s.next(), 200.0)
    assert_almost_equal(s.next(), 210.55728090000841)
    assert_almost_equal(s.next(), 222.54033307585166)
    assert_almost_equal(s.next(), 236.75444679663241)
    assert_almost_equal(s.next(), 255.0)
    assert_almost_equal(s.next(), 275.0)
    s.reset()
    assert_almost_equal(s.next(), 144.72135954999578)
