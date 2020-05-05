import numpy as np
import pytest

from opytimark.markers import three_dimensional


def test_wolfe():
    f = three_dimensional.Wolfe()

    x = np.array([1, 1, 1])

    y = f(x)

    assert np.round(y, 2) == 2.33
