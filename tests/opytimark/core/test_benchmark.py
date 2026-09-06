# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from typing import get_type_hints

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
    original = getattr(benchmark, attribute)

    with pytest.raises(error) as raised:
        setattr(benchmark, attribute, value)

    assert getattr(benchmark, attribute) == original
    assert str(raised.value).startswith(f"`{attribute}` must be ")
    assert repr(value) in str(raised.value)
    assert str(raised.value).endswith(".")


@pytest.mark.parametrize(
    "attribute",
    ["name", "dims", "continuous", "convex", "differentiable", "multimodal", "separable"],
)
def test_explicit_none_is_validated(attribute):
    with pytest.raises(exception.TypeError) as raised:
        Ackley2(**{attribute: None})

    assert str(raised.value).startswith(f"`{attribute}` must be ")
    assert "None (NoneType)" in str(raised.value)


def test_subclass_tuple_defaults_and_mutable_metadata_are_preserved():
    class CustomBenchmark(Benchmark):
        _defaults = ("custom", 3, True, False, True, False, True)

    benchmark = CustomBenchmark(convex=True)

    assert benchmark.name == "custom"
    assert benchmark.dims == 3
    assert benchmark.continuous is True
    assert benchmark.convex is True
    assert benchmark.differentiable is True
    assert benchmark.multimodal is False
    assert benchmark.separable is True

    benchmark.name = "renamed"
    benchmark.dims = -1
    for attribute in ("continuous", "convex", "differentiable", "multimodal", "separable"):
        setattr(benchmark, attribute, not getattr(benchmark, attribute))

    assert benchmark.name == "renamed"
    assert benchmark.dims == -1
    assert benchmark.continuous is False
    assert benchmark.convex is False
    assert benchmark.differentiable is False
    assert benchmark.multimodal is True
    assert benchmark.separable is False
    assert CustomBenchmark().name == "custom"
    assert CustomBenchmark().dims == 3


def test_integer_subclasses_remain_valid_dimensions():
    benchmark = Benchmark(dims=True)

    assert benchmark.dims is True


@pytest.mark.parametrize(
    ("attribute", "expected_type"),
    [
        ("name", str),
        ("dims", int),
        ("continuous", bool),
        ("convex", bool),
        ("differentiable", bool),
        ("multimodal", bool),
        ("separable", bool),
    ],
)
def test_metadata_properties_expose_public_types_and_documentation(attribute, expected_type):
    accessor = getattr(Benchmark, attribute)

    assert get_type_hints(accessor.fget)["return"] is expected_type
    assert get_type_hints(accessor.fset)[attribute] is expected_type
    assert get_type_hints(accessor.fset)["return"] is type(None)
    assert accessor.fget.__doc__
    assert accessor.fset.__doc__
