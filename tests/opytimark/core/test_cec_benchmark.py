import math

import numpy as np
import pytest

from opytimark.core import CECBenchmark, CECCompositeBenchmark
from opytimark.markers.cec import year_2005, year_2010
from opytimark.utils import exception


class F1(CECBenchmark):
    _defaults = ("F1", 100, True, True, True, False, True)
    _year = "2005"
    _auxiliary_data = ("o",)


def test_cec_benchmark_loads_bundled_data():
    benchmark = F1()

    assert benchmark.o.shape == (100,)

    with pytest.raises(NotImplementedError):
        benchmark(None)


def test_cec_constructor_and_subclass_compatibility():
    benchmark = year_2005.F1(
        "F1",
        "2005",
        ("o",),
        50,
        False,
        False,
        False,
        True,
        False,
    )

    assert benchmark.dims == 50
    assert benchmark.multimodal is True

    class DerivedF1(year_2005.F1):
        pass

    derived = DerivedF1()
    assert derived.name == "F1"
    assert derived.o.shape == (100,)


def test_special_cec_constructor_positions():
    grouped = year_2010.F4(
        "F4",
        "2010",
        ("o", "M"),
        1000,
        25,
        True,
        True,
        True,
        False,
        False,
    )
    composite = year_2005.F15(
        "F15",
        "2005",
        ("o", "M2", "M10", "M30", "M50"),
        321,
        100,
        True,
        True,
        True,
        True,
        True,
    )

    assert grouped.m == 25
    assert composite.bias == 321


def test_cec_year_validation_and_manual_loading():
    benchmark = CECBenchmark("F1", "2005")

    benchmark._load_auxiliary_data("F1", "2005", "o")
    assert benchmark.o.shape == (100,)

    with pytest.raises(exception.TypeError):
        benchmark.year = 2005

    with pytest.raises(exception.TypeError):
        year_2005.F1(year=None)


@pytest.mark.parametrize(
    "benchmark_type",
    [CECCompositeBenchmark] + [getattr(year_2005, f"F{i}") for i in range(15, 26)],
)
@pytest.mark.parametrize("point", [[0.25, 0.75], [1000.0, -1000.0]])
def test_composition_preserves_constant_fitness(benchmark_type, point, monkeypatch):
    benchmark = benchmark_type(auxiliary_data=())
    benchmark.sigma = benchmark.l = (1,) * 10
    benchmark.f = (lambda x: 1.0,) * 10
    benchmark.o = np.arange(10.0)[:, None] * np.ones((10, 2))
    benchmark.M2 = np.tile(np.eye(2), (10, 1))
    benchmark.f_bias = np.full(10, 37.0)
    monkeypatch.setattr(np.random, "normal", lambda: -2.0)

    expected = benchmark.C + 37
    if benchmark_type is year_2005.F17:
        expected *= 1.4

    with np.errstate(invalid="raise", divide="raise"):
        actual = benchmark(np.array(point))

    assert actual == pytest.approx(expected + benchmark.bias)


def test_composition_matches_hand_calculated_weighted_fitness():
    benchmark = CECCompositeBenchmark(
        sigma=(1, 2),
        l=(1, 2),
        functions=(lambda x: np.dot(x, x), lambda x: np.dot(x, x) + 1),
        bias=7,
    )
    benchmark.o = np.array([[0.0, 0.0], [1.0, 2.0]])
    benchmark.M2 = np.array([[0, -1], [1, 0], [2, 0], [0, 0.5]])
    benchmark.f_bias = (0, 100)

    first_weight = math.exp(-0.5)
    second_weight = math.exp(-9 / 16) * (1 - math.exp(-5))
    expected = (first_weight * 80 + second_weight * (2000 * 1.5625 / 27.5625 + 100)) / (
        first_weight + second_weight
    ) + 7

    assert benchmark(np.array([1.0, -1.0])) == pytest.approx(expected)


def test_noncontinuous_composition_applies_rounding():
    benchmark = year_2005.F23()
    reference = year_2005.F21(auxiliary_data=())
    reference.o = benchmark.o.copy()
    reference.M2 = benchmark.M2
    x = benchmark.o[0, :2] + np.array([0.25, 0.75])
    rounded = np.array([x[0], round(2 * x[1]) / 2])
    original = x.copy()

    assert benchmark(x) == pytest.approx(reference(rounded))
    np.testing.assert_array_equal(x, original)
