import numpy as np

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

    x = np.array([-1, 0, 0])

    y = f(x)

    assert y == 98696.04401089359


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


def test_paviani():
    f = many_dimensional.Paviani()

    x = np.full(10, 9.351)

    y = f(x)

    assert y == -45.778452053828865


def test_schmidt_vetters():
    f = many_dimensional.SchmidtVetters()

    x = np.array([0.78547, 0.78547, 0.78547])

    y = f(x)

    assert np.round(y) == 3


def test_watson():
    f = many_dimensional.Watson()

    x = np.array([-0.0158, 1.0129, -0.23299, 1.2598, -1.5129, 0.9928])

    y = f(x)

    assert np.round(y, 5) == 0.00229


def test_wolfe():
    f = many_dimensional.Wolfe()

    x = np.array([1, 1, 1])

    y = f(x)

    assert np.round(y, 2) == 2.33
