import numpy as np
import pytest

from opytimark.markers import n_dimensional
from opytimark.markers.cec import year_2005
from opytimark.utils import decorator, exception


def test_exact_dimension_validation():
    @decorator.check_exact_dimension
    def call(benchmark, x):
        return x

    benchmark = n_dimensional.Sphere()

    with pytest.raises(exception.SizeError):
        call(benchmark, np.array([]))

    benchmark.dims = 2
    with pytest.raises(exception.SizeError):
        call(benchmark, np.array([1]))

    value = call(benchmark, np.ones((2, 1, 3)), "ignored")
    assert value.shape == (2, 3)


def test_cec_dimension_validation_selects_matrix():
    benchmark = year_2005.F3()

    with pytest.raises(exception.SizeError):
        benchmark(np.zeros(51))

    for dimension in (2, 10, 30, 50):
        benchmark(np.zeros(dimension))
        assert benchmark.M.shape == (dimension, dimension)


def test_maximum_dimension_validation():
    benchmark = year_2005.F1()

    with pytest.raises(exception.SizeError):
        benchmark(np.zeros(101))

    benchmark(np.zeros(100))
