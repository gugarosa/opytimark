import numpy as np
import pytest

from opytimark.core import benchmark


def test_benchmark_dims():
    new_benchmark = benchmark.Benchmark()

    assert new_benchmark.dims == 1


def test_benchmark_call():
    new_benchmark = benchmark.Benchmark()

    with pytest.raises(NotImplementedError):
        new_benchmark(None)

