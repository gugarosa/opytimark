import numpy as np
import pytest

from opytimark.markers import one_dimensional


def test_gramacy_lee():
    f = one_dimensional.GramacyLee()

    x = np.array([1])

    y = f(x)

    assert np.round(y, 2) == 0
