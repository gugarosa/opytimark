# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import pytest

from opytimark.core import Benchmark
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


@pytest.mark.parametrize(
    "decorate",
    [
        decorator.check_exact_dimension,
        decorator.check_exact_dimension_and_auxiliary_matrix,
        decorator.check_less_equal_dimension,
    ],
)
@pytest.mark.parametrize("shape", [(2,), (2, 1), (2, 1, 3), (2, 3)])
@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
def test_decorators_preserve_keyword_positional_shape_and_dtype(decorate, shape, dtype):
    def call(benchmark, x):
        return x

    wrapped = decorate(call)
    benchmark = Benchmark(dims=2)
    benchmark.M2 = np.eye(2)
    point = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    original = point.copy()
    expected = point.squeeze(axis=1) if len(shape) > 1 and shape[1] == 1 else point

    positional = wrapped(benchmark, point, "ignored", "also ignored")
    keyword = wrapped(benchmark, x=point)

    for result in (positional, keyword):
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == dtype
        assert np.shares_memory(result, point)
    np.testing.assert_array_equal(point, original)
    assert wrapped.__name__ == call.__name__
    assert wrapped.__qualname__ == call.__qualname__
    assert wrapped.__wrapped__ is call


@pytest.mark.parametrize(
    "decorate",
    [
        decorator.check_exact_dimension,
        decorator.check_exact_dimension_and_auxiliary_matrix,
        decorator.check_less_equal_dimension,
    ],
)
@pytest.mark.parametrize("point", [[1, 2], (1, 2), [[1], [2]]])
def test_decorators_convert_keyword_arraylikes_with_numpy_dtype_inference(decorate, point):
    @decorate
    def call(benchmark, x):
        return x

    benchmark = Benchmark(dims=2)
    benchmark.M2 = np.eye(2)
    result = call(benchmark, x=point)

    np.testing.assert_array_equal(result, [1, 2])
    assert result.dtype == np.asarray(point).dtype


@pytest.mark.parametrize(
    ("decorate", "dims", "size", "message"),
    [
        (decorator.check_exact_dimension, -1, 0, "`Benchmark` requires a nonempty input, but got shape (0,)."),
        (decorator.check_exact_dimension, 2, 1, "`Benchmark` requires 2 input dimensions, but got shape (1,)."),
        (
            decorator.check_exact_dimension_and_auxiliary_matrix,
            2,
            3,
            "`Benchmark` requires 2, 10, 30, or 50 input dimensions, but got shape (3,).",
        ),
        (
            decorator.check_less_equal_dimension,
            2,
            3,
            "`Benchmark` requires at most 2 input dimensions, but got shape (3,).",
        ),
    ],
)
def test_keyword_dimensions_are_validated_before_evaluation(decorate, dims, size, message):
    @decorate
    def call(benchmark, x):
        pytest.fail("an invalid input must not reach evaluation")

    benchmark = Benchmark(dims=dims)
    benchmark.M = "unchanged"

    with pytest.raises(exception.SizeError) as error:
        call(benchmark, x=np.zeros(size))

    assert str(error.value) == message
    assert benchmark.M == "unchanged"


@pytest.mark.parametrize(
    "benchmark_type",
    [n_dimensional.Sphere, year_2005.F1, year_2005.F3, year_2005.F15],
)
def test_benchmark_keyword_calls_match_positional_calls(benchmark_type):
    benchmark = benchmark_type()
    point = np.array([0.25, 0.75], dtype=np.float32)
    original = point.copy()
    expected = benchmark(point)

    for result in (benchmark(x=point), benchmark(x=point[:, None])):
        np.testing.assert_equal(result, expected)
        assert type(result) is type(expected)
    np.testing.assert_array_equal(point, original)
