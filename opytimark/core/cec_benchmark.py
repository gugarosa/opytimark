# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

import opytimark.utils.decorator as d
import opytimark.utils.loader as ld
from opytimark.core.benchmark import _MISSING, Benchmark, _Missing, _validate_type
from opytimark.typing import BenchmarkValue


class CECBenchmark(Benchmark):
    """
    Base class for CEC benchmarks backed by auxiliary data.

    """

    _year: str = ""
    _auxiliary_data: tuple[str, ...] = ()

    def __init__(
        self,
        name: str | _Missing = _MISSING,
        year: str | _Missing = _MISSING,
        auxiliary_data: Iterable[str] | _Missing = _MISSING,
        dims: int | _Missing = _MISSING,
        continuous: bool | _Missing = _MISSING,
        convex: bool | _Missing = _MISSING,
        differentiable: bool | _Missing = _MISSING,
        multimodal: bool | _Missing = _MISSING,
        separable: bool | _Missing = _MISSING,
    ) -> None:
        """Initialize CEC metadata and load the requested auxiliary arrays.

        Args:
            name: Benchmark name used as the auxiliary-data filename prefix.
            year: CEC edition identifying the auxiliary-data archive.
            auxiliary_data: Iterable of auxiliary variable names to load as instance attributes.
            dims: Required or maximum dimension count for the benchmark's input validator.
            continuous: Whether the benchmark is continuous.
            convex: Whether the benchmark is convex.
            differentiable: Whether the benchmark is differentiable.
            multimodal: Whether the benchmark has multiple modes.
            separable: Whether coordinates can be optimized independently.

        Raises:
            TypeError: A metadata value has the wrong type or auxiliary_data is not iterable.
            ValueError: The dimension count or auxiliary data is invalid.
            OSError: Requested auxiliary data cannot be read or downloaded.

        Notes:
            Omitted arguments use the subclass's `_defaults`, `_year`, and `_auxiliary_data` values.
            Explicit None values are not treated as omitted.
            Local auxiliary data takes precedence over bundled data, with remote loading as the final fallback.
            Mutating metadata after initialization does not reload auxiliary data.

        """

        super().__init__(
            name,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )

        self.year = self._year if year is _MISSING else year

        data = self._auxiliary_data if auxiliary_data is _MISSING else auxiliary_data
        self._load_auxiliary_data(self.name, self.year, data)

    @property
    def year(self) -> str:
        """Return the declared CEC edition.

        Returns:
            Validated CEC edition string.

        """

        return self._year_value

    @year.setter
    def year(self, year: str) -> None:
        """Set the CEC edition without reloading auxiliary arrays.

        Args:
            year: CEC edition identifying the auxiliary-data archive.

        Raises:
            TypeError: The year is not a string.

        """

        _validate_type("year", year, str)

        self._year_value = year

    def _load_auxiliary_data(self, name: str, year: str, data: Iterable[str]) -> None:
        for variable in data:
            setattr(self, variable, ld.load_cec_auxiliary(f"{name}_{variable}", year))


class CECCompositeBenchmark(CECBenchmark):
    """
    Base class for shifted, scaled, and rotated CEC composite benchmarks.

    """

    _bias = 1

    def __init__(
        self,
        name: str | _Missing = _MISSING,
        year: str | _Missing = _MISSING,
        auxiliary_data: Iterable[str] | _Missing = _MISSING,
        sigma: Sequence[float] | NDArray[Any] = (),
        l: Sequence[float] | NDArray[Any] = (),
        functions: Sequence[Callable[[NDArray[Any]], BenchmarkValue]] = (),
        bias: BenchmarkValue | _Missing = _MISSING,
        dims: int | _Missing = _MISSING,
        continuous: bool | _Missing = _MISSING,
        convex: bool | _Missing = _MISSING,
        differentiable: bool | _Missing = _MISSING,
        multimodal: bool | _Missing = _MISSING,
        separable: bool | _Missing = _MISSING,
    ) -> None:
        """Initialize a CEC composition while retaining caller-supplied component objects.

        Args:
            name: Benchmark name used as the auxiliary-data filename prefix.
            year: CEC edition identifying the auxiliary-data archive.
            auxiliary_data: Iterable of auxiliary variable names to load as instance attributes.
            sigma: Component weight widths in evaluation order.
            l: Component coordinate scales in evaluation order.
            functions: Component callables evaluated in the supplied order.
            bias: Overall additive bias, using the subclass's _bias value when omitted.
            dims: Dimension metadata retained for compatibility with the shared benchmark constructor.
            continuous: Whether the benchmark is continuous.
            convex: Whether the benchmark is convex.
            differentiable: Whether the benchmark is differentiable.
            multimodal: Whether the benchmark has multiple modes.
            separable: Whether coordinates can be optimized independently.

        Raises:
            TypeError: A metadata value has the wrong type or auxiliary_data is not iterable.
            ValueError: The dimension count or auxiliary data is invalid.
            OSError: Requested auxiliary data cannot be read or downloaded.

        Notes:
            Metadata and auxiliary-data defaults follow CECBenchmark.
            Component sequences are retained without conversion or length validation.
            Initialization supplies C = 2000 and the ten historical component biases.

        """

        super().__init__(
            name,
            year,
            auxiliary_data,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )

        self._initialize_composition(
            sigma,
            l,
            functions,
            self._bias if bias is _MISSING else bias,
        )

    def _initialize_composition(
        self,
        sigma: Sequence[float] | NDArray[Any],
        scale: Sequence[float] | NDArray[Any],
        functions: Sequence[Callable[[NDArray[Any]], BenchmarkValue]],
        bias: BenchmarkValue,
    ) -> None:
        self.sigma = sigma
        self.l = scale
        self.f = functions
        self.bias = bias
        self.C = 2000
        self.f_bias = (0, 100, 200, 300, 400, 500, 600, 700, 800, 900)

    @d.check_exact_dimension_and_auxiliary_matrix
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the weighted CEC composition and add its overall bias.

        Args:
            x: Numeric coordinates with first-axis length 2, 10, 30, or 50.

        Returns:
            Weighted component fitness plus the overall bias, without converting the result type.

        Raises:
            SizeError: The input's first-axis length is unsupported.

        Notes:
            Positional and keyword inputs are normalized without forcing a floating-point dtype.
            A singleton second axis is removed, other axes are retained, and the input is not mutated.
            The matching M2, M10, M30, or M50 rotation array is selected as M.
            Each component is evaluated at its scaled, rotated reference point before its shifted input.
            Relative exponential weights are attenuated before normalization to avoid numerical underflow.

        """

        return self._evaluate_composition(x) + self.bias

    def _evaluate_composition(self, x: NDArray[Any]) -> BenchmarkValue:
        dimension = x.shape[0]
        log_weights = np.zeros(len(self.f))
        fitness = np.zeros(len(self.f))
        reference = 5 * np.ones(dimension)

        for index, function in enumerate(self.f):
            start = index * dimension
            end = start + dimension
            shifted = x - self.o[index][:dimension]
            log_weights[index] = -np.sum(shifted**2) / (2 * dimension * self.sigma[index] ** 2)
            normalizer = function(np.matmul(reference / self.l[index], self.M[start:end]))
            fitness[index] = self.C * function(np.matmul(shifted / self.l[index], self.M[start:end])) / normalizer

        # Relative exponentials avoid underflow before weight normalization
        maximum = np.max(log_weights)
        weights = np.exp(log_weights - maximum)
        weights[log_weights != maximum] *= -np.expm1(10 * maximum)
        weights /= np.sum(weights)

        return np.matmul(weights, fitness + self.f_bias)
