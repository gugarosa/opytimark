import numpy as np

from opytimark.markers.cec import year_2013


def test_F4():
    f = year_2013.F4()

    x = np.loadtxt('./data/2013/F4_o.txt')

    x = x[:300]

    # x = np.array([3.8465944, 4.3236221, -2.8216294, 0.64653818, 4.3382019])

    y = f(x)

    assert y == 0