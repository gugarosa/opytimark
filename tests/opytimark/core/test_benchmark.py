import pytest

from opytimark.core import Benchmark
from opytimark.markers.two_dimensional import Ackley2
from opytimark.utils import exception


def test_benchmark_defaults_and_call():
    benchmark = Benchmark()

    assert (
        benchmark.name,
        benchmark.dims,
        benchmark.continuous,
        benchmark.convex,
        benchmark.differentiable,
        benchmark.multimodal,
        benchmark.separable,
    ) == ("Benchmark", 1, False, False, False, False, False)

    with pytest.raises(NotImplementedError):
        benchmark(None)


def test_concrete_benchmark_preserves_constructor_arguments():
    benchmark = Ackley2("custom", 3, False, False, False, True, True)

    assert (
        benchmark.name,
        benchmark.dims,
        benchmark.continuous,
        benchmark.convex,
        benchmark.differentiable,
        benchmark.multimodal,
        benchmark.separable,
    ) == ("custom", 3, False, False, False, True, True)


@pytest.mark.parametrize(
    ("attribute", "value", "error"),
    [
        ("name", 1, exception.TypeError),
        ("dims", "1", exception.TypeError),
        ("dims", 0, exception.ValueError),
        ("continuous", 1, exception.TypeError),
        ("convex", 1, exception.TypeError),
        ("differentiable", 1, exception.TypeError),
        ("multimodal", 1, exception.TypeError),
        ("separable", 1, exception.TypeError),
    ],
)
def test_benchmark_metadata_validation(attribute, value, error):
    benchmark = Benchmark()

    with pytest.raises(error):
        setattr(benchmark, attribute, value)


def test_explicit_none_is_validated():
    with pytest.raises(exception.TypeError):
        Ackley2(dims=None)
