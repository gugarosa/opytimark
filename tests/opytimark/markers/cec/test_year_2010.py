import numpy as np

import opytimark.utils.loader as l
from opytimark.markers.cec import year_2010

np.random.seed(0)


def test_F1():
    f = year_2010.F1()

    x = l.load_cec_auxiliary('F1_o', '2010')

    y = f(x)

    assert y == 0


def test_F2():
    f = year_2010.F2()

    x = l.load_cec_auxiliary('F2_o', '2010')

    y = f(x)

    assert y == 0


def test_F3():
    f = year_2010.F3()

    x = l.load_cec_auxiliary('F3_o', '2010')

    y = f(x)

    assert y == 0


def test_F4():
    f = year_2010.F4(group_size=50)

    x = l.load_cec_auxiliary('F4_o', '2010')

    y = f(x)

    assert y == 0


def test_F5():
    f = year_2010.F5(group_size=50)

    x = l.load_cec_auxiliary('F5_o', '2010')

    y = f(x)

    assert y == 0


def test_F6():
    f = year_2010.F6(group_size=50)

    x = l.load_cec_auxiliary('F6_o', '2010')

    y = f(x)

    assert y == 0


def test_F7():
    f = year_2010.F7(group_size=50)

    x = l.load_cec_auxiliary('F7_o', '2010')

    y = f(x)

    assert y == 0


def test_F8():
    f = year_2010.F8(group_size=50)

    x = l.load_cec_auxiliary('F8_o', '2010')
    x += 1

    y = f(x)

    assert y == 950


def test_F9():
    f = year_2010.F9(group_size=50)

    x = l.load_cec_auxiliary('F9_o', '2010')

    y = f(x)

    assert y == 0


def test_F10():
    f = year_2010.F10(group_size=50)

    x = l.load_cec_auxiliary('F10_o', '2010')

    y = f(x)

    assert y == 0


def test_F11():
    f = year_2010.F11(group_size=50)

    x = l.load_cec_auxiliary('F11_o', '2010')

    y = f(x)

    assert y == 0


def test_F12():
    f = year_2010.F12(group_size=50)

    x = l.load_cec_auxiliary('F12_o', '2010')

    y = f(x)

    assert y == 0


def test_F13():
    f = year_2010.F13(group_size=50)

    x = l.load_cec_auxiliary('F13_o', '2010')
    x += 1

    y = f(x)

    assert y == 500


def test_F14():
    f = year_2010.F14(group_size=50)

    x = l.load_cec_auxiliary('F14_o', '2010')

    y = f(x)

    assert y == 0


def test_F15():
    f = year_2010.F15(group_size=50)

    x = l.load_cec_auxiliary('F15_o', '2010')

    y = f(x)

    assert y == 0


def test_F16():
    f = year_2010.F16(group_size=50)

    x = l.load_cec_auxiliary('F16_o', '2010')

    y = f(x)

    assert y == 0


def test_F17():
    f = year_2010.F17(group_size=50)

    x = l.load_cec_auxiliary('F17_o', '2010')

    y = f(x)

    assert y == 0


def test_F18():
    f = year_2010.F18(group_size=50)

    x = l.load_cec_auxiliary('F18_o', '2010')
    x += 1

    y = f(x)

    assert np.round(y) == 0


def test_F19():
    f = year_2010.F19()

    x = l.load_cec_auxiliary('F19_o', '2010')

    y = f(x)

    assert y == 0


def test_F20():
    f = year_2010.F20()

    x = l.load_cec_auxiliary('F20_o', '2010')
    x += 1

    y = f(x)

    assert np.round(y) == 0
