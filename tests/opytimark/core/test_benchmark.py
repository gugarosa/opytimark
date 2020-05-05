import numpy as np
import pytest

from opytimark.core import benchmark


def test_benchmark_name():
    new_benchmark = benchmark.Benchmark()

    assert new_benchmark.name == 'Benchmark'


def test_benchmark_name_setter():
    new_benchmark = benchmark.Benchmark()

    try:
        new_benchmark.name = 1
    except:
        new_benchmark.name = 'name'

    assert new_benchmark.name == 'name'


def test_benchmark_dims():
    new_benchmark = benchmark.Benchmark()

    assert new_benchmark.dims == 1


def test_benchmark_dims_setter():
    new_benchmark = benchmark.Benchmark()

    try:
        new_benchmark.dims = 'a'
    except:
        new_benchmark.dims = 1

    assert new_benchmark.dims == 1

    try:
        new_benchmark.dims = 0
    except:
        new_benchmark.dims = 1

    assert new_benchmark.dims == 1


def test_benchmark_continuous():
    new_benchmark = benchmark.Benchmark()

    assert new_benchmark.continuous == False


def test_benchmark_continuous_setter():
    new_benchmark = benchmark.Benchmark()

    try:
        new_benchmark.continuous = 1
    except:
        new_benchmark.continuous = True

    assert new_benchmark.continuous == True


def test_benchmark_convex():
    new_benchmark = benchmark.Benchmark()

    assert new_benchmark.convex == False


def test_benchmark_convex_setter():
    new_benchmark = benchmark.Benchmark()

    try:
        new_benchmark.convex = 1
    except:
        new_benchmark.convex = True

    assert new_benchmark.convex == True


def test_benchmark_differentiable():
    new_benchmark = benchmark.Benchmark()

    assert new_benchmark.differentiable == False


def test_benchmark_differentiable_setter():
    new_benchmark = benchmark.Benchmark()

    try:
        new_benchmark.differentiable = 1
    except:
        new_benchmark.differentiable = True

    assert new_benchmark.differentiable == True


def test_benchmark_multimodal():
    new_benchmark = benchmark.Benchmark()

    assert new_benchmark.multimodal == False


def test_benchmark_multimodal_setter():
    new_benchmark = benchmark.Benchmark()

    try:
        new_benchmark.multimodal = 1
    except:
        new_benchmark.multimodal = True

    assert new_benchmark.multimodal == True


def test_benchmark_separable():
    new_benchmark = benchmark.Benchmark()

    assert new_benchmark.separable == False


def test_benchmark_separable_setter():
    new_benchmark = benchmark.Benchmark()

    try:
        new_benchmark.separable = 1
    except:
        new_benchmark.separable = True

    assert new_benchmark.separable == True


def test_benchmark_call():
    new_benchmark = benchmark.Benchmark()

    with pytest.raises(NotImplementedError):
        new_benchmark(None)
