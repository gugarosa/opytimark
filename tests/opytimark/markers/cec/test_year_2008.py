import numpy as np

from opytimark.markers.cec import year_2008


def test_F1():
    f = year_2008.F1()

    x = np.array([97.249936, 77.060985, -19.031149, 25.428698, -22.908803])

    y = f(x)

    assert y == -450
