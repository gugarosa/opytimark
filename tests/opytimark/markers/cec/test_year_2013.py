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


def test_F5():
    f = year_2013.F5()

    x = l.load_cec_auxiliary('F5_o', '2013')

    y = f(x)

    assert y == 0


def test_F6():
    f = year_2013.F6()

    x = l.load_cec_auxiliary('F6_o', '2013')

    y = f(x)

    assert y == 0


def test_F7():
    f = year_2013.F7()

    x = l.load_cec_auxiliary('F7_o', '2013')

    y = f(x)

    assert y == 0


def test_F8():
    f = year_2013.F8()

    x = l.load_cec_auxiliary('F8_o', '2013')

    y = f(x)

    assert y == 0


def test_F12():
    f = year_2013.F12()

    x = l.load_cec_auxiliary('F12_o', '2013')
    x += 1

    y = f(x)

    assert round(y) == 0
