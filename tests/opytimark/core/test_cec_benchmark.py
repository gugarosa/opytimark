import numpy as np
import pytest

from opytimark.core import cec_benchmark


def test_cec_benchmark_name():
    new_cec_benchmark = cec_benchmark.CECBenchmark('F1', '2005')

    assert new_cec_benchmark.name == 'F1'


def test_cec_benchmark_name_setter():
    new_cec_benchmark = cec_benchmark.CECBenchmark('F1', '2005')

    try:
        new_cec_benchmark.name = 1
    except:
        new_cec_benchmark.name = 'F1'

    assert new_cec_benchmark.name == 'F1'


def test_cec_benchmark_year():
    new_cec_benchmark = cec_benchmark.CECBenchmark('F1', '2005')

    assert new_cec_benchmark.year == '2005'


def test_cec_benchmark_year_setter():
    new_cec_benchmark = cec_benchmark.CECBenchmark('F1', '2005')

    try:
        new_cec_benchmark.year = 1
    except:
        new_cec_benchmark.year = '2005'

    assert new_cec_benchmark.year == '2005'


def test_cec_benchmark_dims():
    new_cec_benchmark = cec_benchmark.CECBenchmark('F1', '2005')

    assert new_cec_benchmark.dims == 1


def test_cec_benchmark_dims_setter():
    new_cec_benchmark = cec_benchmark.CECBenchmark('F1', '2005')

    try:
        new_cec_benchmark.dims = 'a'
    except:
        new_cec_benchmark.dims = 1

    assert new_cec_benchmark.dims == 1

    try:
        new_cec_benchmark.dims = 0
    except:
        new_cec_benchmark.dims = 1

    assert new_cec_benchmark.dims == 1


def test_cec_benchmark_continuous():
    new_cec_benchmark = cec_benchmark.CECBenchmark('F1', '2005')

    assert new_cec_benchmark.continuous == False


def test_cec_benchmark_continuous_setter():
    new_cec_benchmark = cec_benchmark.CECBenchmark('F1', '2005')

    try:
        new_cec_benchmark.continuous = 1
    except:
        new_cec_benchmark.continuous = True

    assert new_cec_benchmark.continuous == True


def test_cec_benchmark_convex():
    new_cec_benchmark = cec_benchmark.CECBenchmark('F1', '2005')

    assert new_cec_benchmark.convex == False


def test_cec_benchmark_convex_setter():
    new_cec_benchmark = cec_benchmark.CECBenchmark('F1', '2005')

    try:
        new_cec_benchmark.convex = 1
    except:
        new_cec_benchmark.convex = True

    assert new_cec_benchmark.convex == True


def test_cec_benchmark_differentiable():
    new_cec_benchmark = cec_benchmark.CECBenchmark('F1', '2005')

    assert new_cec_benchmark.differentiable == False


def test_cec_benchmark_differentiable_setter():
    new_cec_benchmark = cec_benchmark.CECBenchmark('F1', '2005')

    try:
        new_cec_benchmark.differentiable = 1
    except:
        new_cec_benchmark.differentiable = True

    assert new_cec_benchmark.differentiable == True


def test_cec_benchmark_multimodal():
    new_cec_benchmark = cec_benchmark.CECBenchmark('F1', '2005')

    assert new_cec_benchmark.multimodal == False


def test_cec_benchmark_multimodal_setter():
    new_cec_benchmark = cec_benchmark.CECBenchmark('F1', '2005')

    try:
        new_cec_benchmark.multimodal = 1
    except:
        new_cec_benchmark.multimodal = True

    assert new_cec_benchmark.multimodal == True


def test_cec_benchmark_separable():
    new_cec_benchmark = cec_benchmark.CECBenchmark('F1', '2005')

    assert new_cec_benchmark.separable == False


def test_cec_benchmark_separable_setter():
    new_cec_benchmark = cec_benchmark.CECBenchmark('F1', '2005')

    try:
        new_cec_benchmark.separable = 1
    except:
        new_cec_benchmark.separable = True

    assert new_cec_benchmark.separable == True


def test_cec_benchmark_call():
    new_cec_benchmark = cec_benchmark.CECBenchmark('F1', '2005')

    with pytest.raises(NotImplementedError):
        new_cec_benchmark(None)


def test_cec_benchmark_load_auxiliary_data():
    new_cec_benchmark = cec_benchmark.CECBenchmark('F1', '2005')

    new_cec_benchmark._load_auxiliary_data('F1', '2005', 'o')

    assert new_cec_benchmark.o.shape == (100,)
