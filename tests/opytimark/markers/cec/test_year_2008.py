import numpy as np

from opytimark.markers.cec import year_2008


def test_F1():
    f = year_2008.F1()

    x = np.array([97.249936, 77.060985, -19.031149, 25.428698, -22.908803])

    y = f(x)

    assert y == -450


def test_F2():
    f = year_2008.F2()

    x = np.array([-26.887899, -4.9090304, -56.826025, -95.043670, -4.3397757])

    y = f(x)

    assert y == -450


def test_F3():
    f = year_2008.F3()

    x = np.array([-75.427528, -35.731702, -57.595644, 38.909846, 52.247682])

    y = f(x)

    assert y == 390


def test_F4():
    f = year_2008.F4()

    x = np.array([3.8465944, 4.3236221, -2.8216294, 0.64653818, 4.3382019])

    y = f(x)

    assert y == -330
