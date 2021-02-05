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
