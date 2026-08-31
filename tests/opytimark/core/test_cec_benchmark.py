import pytest

from opytimark.core import CECBenchmark


class F1(CECBenchmark):
    year = "2005"
    auxiliary_data = ("o",)


def test_cec_benchmark_loads_bundled_data():
    benchmark = F1()

    assert benchmark.o.shape == (100,)

    with pytest.raises(NotImplementedError):
        benchmark(None)
