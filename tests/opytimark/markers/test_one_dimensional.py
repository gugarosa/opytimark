import numpy as np

from opytimark.markers import one_dimensional


def test_forrester():
    f = one_dimensional.Forrester()

    x = np.array([0.75])

    y = f(x)

    assert y == -5.9932767166446155


def test_gramacy_lee():
    f = one_dimensional.GramacyLee()

    x = np.array([1])

    y = f(x)

    assert np.round(y, 2) == 0
