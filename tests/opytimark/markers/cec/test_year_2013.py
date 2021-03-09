import opytimark.utils.loader as l
from opytimark.markers.cec import year_2013


def test_F1():
    f = year_2013.F1()

    x = l.load_cec_auxiliary('F1_o', '2013')

    y = f(x)

    assert y == 0


def test_F2():
    f = year_2013.F2()

    x = l.load_cec_auxiliary('F2_o', '2013')

    y = f(x)

    assert y == 0


def test_F3():
    f = year_2013.F3()

    x = l.load_cec_auxiliary('F3_o', '2013')

    y = f(x)

    assert y == 0


def test_F4():
    f = year_2013.F4()

    x = l.load_cec_auxiliary('F4_o', '2013')

    y = f(x)

    assert y == 0
