import numpy as np
import pytest

from opytimark.markers import n_dimensional
from opytimark.markers.cec import year_2005


def test_exact_dimension_validation():
    benchmark = n_dimensional.Sphere()

    with pytest.raises(ValueError):
        benchmark(np.array([]))

    benchmark.dims = 1
    with pytest.raises(ValueError):
        benchmark(np.array([1, 2]))

    assert benchmark(np.array([[1]])) == 1


def test_cec_dimension_validation_selects_matrix():
    benchmark = year_2005.F3()

    with pytest.raises(ValueError):
        benchmark(np.zeros(51))

    for dimension in (2, 10, 30, 50):
        benchmark(np.zeros(dimension))
        assert benchmark.M.shape == (dimension, dimension)


def test_maximum_dimension_validation():
    benchmark = year_2005.F1()

    with pytest.raises(ValueError):
        benchmark(np.zeros(101))

    benchmark(np.zeros(100))
