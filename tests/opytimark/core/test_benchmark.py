import pytest

from opytimark.core import Benchmark


def test_benchmark_defaults():
    benchmark = Benchmark()

    assert benchmark.dims == 1

    with pytest.raises(NotImplementedError):
        benchmark(None)
