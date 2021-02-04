import numpy as np

from opytimark.markers.cec import year_2005


def test_F1():
    f = year_2005.F1()

    x = np.array([-39.311900, 58.899900, -46.322400, -74.651500, -16.799700])

    y = f(x)

    assert y == -450
