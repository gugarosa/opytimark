import numpy as np

from opytimark.markers.cec import year_2005


def test_F1():
    f = year_2005.F1()

    x = np.array([-39.311900, 58.899900, -46.322400, -74.651500, -16.799700])

    y = f(x)

    assert y == -450


def test_F2():
    f = year_2005.F2()

    x = np.array([35.626700, -82.912300, -10.642300, -83.581500, 83.155200])

    y = f(x)

    assert y == -450


def test_F3():
    f = year_2005.F3()

    x = np.array([-32.201300, 64.977600])

    y = f(x)

    assert y == -450


def test_F4():
    f = year_2005.F4()

    x = np.array([35.626700, -82.912300, -10.642300, -83.581500, 83.155200])

    y = f(x)

    assert y == -450


def test_F5():
    f = year_2005.F5()

    x = np.array([-5.5559000, 7.9470000, -1.5380000, 8.3897000, 7.7182000])

    y = f(x)

    assert y == -310


def test_F6():
    f = year_2005.F6()

    x = np.array([81.023200, -48.395000, 19.231600, -2.5231000, 70.433800])

    y = f(x)

    assert y == 390


def test_F7():
    f = year_2005.F7()

    x = np.array([-276.26840, -11.911000])

    y = f(x)

    assert y == -180
