import numpy as np
import pytest

from opytimark.markers import three_dimensional


def test_biggs_exponential3():
    f = three_dimensional.BiggsExponential3()

    x = np.array([1, 10, 5])

    y = f(x)

    assert y == 0


def test_wolfe():
    f = three_dimensional.Wolfe()

    x = np.array([1, 1, 1])

    y = f(x)

    assert np.round(y, 2) == 2.33
