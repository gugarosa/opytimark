import numpy as np
import pytest

from opytimark.markers import many_dimensional


def test_biggs_exponential3():
    f = many_dimensional.BiggsExponential3()

    x = np.array([1, 10, 5])

    y = f(x)

    assert y == 0


def test_biggs_exponential4():
    f = many_dimensional.BiggsExponential4()

    x = np.array([1, 10, 1, 5])

    y = f(x)

    assert y == 0


def test_biggs_exponential5():
    f = many_dimensional.BiggsExponential5()

    x = np.array([1, 10, 1, 5, 4])

    y = f(x)

    assert y == 0


def test_biggs_exponential6():
    f = many_dimensional.BiggsExponential6()

    x = np.array([1, 10, 1, 5, 4, 3])

    y = f(x)

    assert y == 0


def test_box_betts():
    f = many_dimensional.BoxBetts()

    x = np.array([1, 10, 1])

    y = f(x)

    assert y == 0


def test_colville():
    f = many_dimensional.Colville()

    x = np.array([1, 1, 1, 1])

    y = f(x)

    assert y == 0


def test_gulf_research():
    f = many_dimensional.GulfResearch()

    x = np.array([50, 25, 1.5])

    y = f(x)

    assert np.round(y) == 0


def test_helical_valley():
    f = many_dimensional.HelicalValley()

    x = np.array([1, 0, 0])

    y = f(x)

    assert y == 0


def test_miele_cantrell():
    f = many_dimensional.MieleCantrell()

    x = np.array([0, 1, 1, 1])

    y = f(x)

    assert y == 0


def test_mishra9():
    f = many_dimensional.Mishra9()

    x = np.array([1, 2, 3])

    y = f(x)

    assert y == 0


def test_wolfe():
    f = many_dimensional.Wolfe()

    x = np.array([1, 1, 1])

    y = f(x)

    assert np.round(y, 2) == 2.33
