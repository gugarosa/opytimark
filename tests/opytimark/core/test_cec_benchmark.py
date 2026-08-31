import pytest

from opytimark.core import CECBenchmark
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
