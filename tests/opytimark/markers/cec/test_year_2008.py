import numpy as np

import opytimark.utils.loader as l
from opytimark.markers.cec import year_2008


def test_F1():
    f = year_2008.F1()

    x = l.load_cec_auxiliary("F1_o", "2008")

    y = f(x)

    assert y == -450


def test_F2():
    f = year_2008.F2()

    x = l.load_cec_auxiliary("F2_o", "2008")

    y = f(x)

    assert y == -450


def test_F3():
    f = year_2008.F3()

    x = l.load_cec_auxiliary("F3_o", "2008") + 1

    y = f(x)

    assert y == 390


def test_F4():
    f = year_2008.F4()

    x = l.load_cec_auxiliary("F4_o", "2008")

    y = f(x)

    assert y == -330


def test_F5():
    f = year_2008.F5()

    x = l.load_cec_auxiliary("F5_o", "2008")

    y = f(x)

    assert y == -180


def test_F6():
    f = year_2008.F6()

    x = l.load_cec_auxiliary("F6_o", "2008")

    y = f(x)

    assert y == -140


def test_F7():
    f = year_2008.F7()

    x = np.ones(50)

    y = f(x)

    assert y == 0
