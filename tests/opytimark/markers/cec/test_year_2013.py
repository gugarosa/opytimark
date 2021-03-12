import numpy as np

import opytimark.utils.loader as l
from opytimark.markers.cec import year_2013


def test_T_irregularity():
    x = np.asarray([1, 2, 3, 4, 5])

    x_t = year_2013.T_irregularity(x)

    assert np.sum(x_t) == 14.878553613491857


def test_T_asymmetry():
    x = np.asarray([1, 2, 3, 4, 5])

    x_t = year_2013.T_asymmetry(x, 0.2)

    assert np.sum(x_t) == 23.061844625640674


def test_T_diagonal():
    M = year_2013.T_diagonal(5, 10)

    assert np.sum(M) == 7.905694150420949


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


def test_F9():
    f = year_2013.F9()

    x = l.load_cec_auxiliary('F9_o', '2013')

    y = f(x)

    assert y == 0


def test_F10():
    f = year_2013.F10()

    x = l.load_cec_auxiliary('F10_o', '2013')

    y = f(x)

    assert y == 0


def test_F11():
    f = year_2013.F11()

    x = l.load_cec_auxiliary('F11_o', '2013')

    y = f(x)

    assert y == 0


def test_F12():
    f = year_2013.F12()

    x = l.load_cec_auxiliary('F12_o', '2013')
    x += 1

    y = f(x)

    assert round(y) == 0


def test_F13():
    f = year_2013.F13()

    x = l.load_cec_auxiliary('F13_o', '2013')

    y = f(x)

    assert y == 0


def test_F14():
    f = year_2013.F14()

    x = l.load_cec_auxiliary('F14_o', '2013')

    y = f(x)

    assert round(y) == 274036796965288214528


def test_F15():
    f = year_2013.F15()

    x = l.load_cec_auxiliary('F15_o', '2013')

    y = f(x)

    assert y == 0
