import numpy as np
import pytest

from opytimark.markers import classic


def test_sphere():
    f = classic.Sphere()

    x = np.array([0.5, 0.5, 1, 1])

    y = f(x)

    assert np.round(y, 2) == 2.5
