import numpy as np
import pytest

from opytimark.markers import n_dimensional

def test_ackley1():
    f = n_dimensional.Ackley1()

    x = np.zeros(1)

    y = f(x)

    assert y == 0

def test_alpine1():
    f = n_dimensional.Alpine1()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_alpine2():
    f = n_dimensional.Alpine2()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_brown():
    f = n_dimensional.Brown()

    x = np.zeros(50)

    y = f(x)

    assert y == 0

def test_chung_reynolds():
    f = n_dimensional.ChungReynolds()

    x = np.zeros(50)

    y = f(x)

    assert y == 0

def test_cosine_mixture():
    f = n_dimensional.CosineMixture()

    x = np.zeros(50)

    y = f(x)

    assert y == -5.0

def test_csendes():
    f = n_dimensional.Csendes()

    x = np.zeros(50)

    y = f(x)

    assert y == 0

def test_deb1():
    f = n_dimensional.Deb1()

    x = np.array([-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9])

    y = f(x)

    assert y == -1.0

def test_deb3():
    f = n_dimensional.Deb3()

    x = np.zeros(50)

    y = f(x)

    assert np.round(y, 3) == -0.125


def test_dixon_price():
    f = n_dimensional.DixonPrice()

    x = np.array([1, np.sqrt(0.5)])

    y = f(x)

    assert np.round(y) == 0


def test_exponential():
    f = n_dimensional.Exponential()

    x = np.zeros(50)

    y = f(x)

    assert y == 1

def test_griewank():
    f = n_dimensional.Griewank()

    x = np.zeros(50)

    y = f(x)

    assert y == 0

def test_happy_cat():
    f = n_dimensional.HappyCat()

    x = np.full(50, -1)

    y = f(x)

    assert y == 0

def test_levy():
    f = n_dimensional.Levy()

    x = np.ones(50)

    y = f(x)

    assert np.round(y) == 0

def test_michalewicz():
    f = n_dimensional.Michalewicz()

    x = np.array([2.20, 1.57])

    y = f(x)

    assert np.round(y, 4) == -1.8011