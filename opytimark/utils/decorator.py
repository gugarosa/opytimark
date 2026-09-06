# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

import opytimark.utils.exception as e

_Result = TypeVar("_Result")


def _vector(x: ArrayLike) -> NDArray[Any]:
    x = np.asarray(x)
    try:
        return np.squeeze(x, axis=1)
    except ValueError:
        return x


def check_exact_dimension(function: Callable[..., _Result]) -> Callable[..., _Result]:
    """Require the exact dimension declared by a benchmark.

    Args:
        function: Benchmark method receiving an instance and a normalized NumPy input.

    Returns:
        Wrapped method that validates positional or keyword input and preserves the result type.

    Raises:
        SizeError: The input is empty for dimension -1 or has the wrong first-axis length.

    Notes:
        Inputs use NumPy's asarray dtype inference, with only a singleton second axis removed.
        Other axes are retained, and extra positional arguments are ignored.

    """

    @wraps(function)
    def _validate(benchmark: Any, x: ArrayLike, *args: Any) -> _Result:
        x = _vector(x)

        if benchmark.dims == -1:
            if not x.shape[0]:
                raise e.SizeError(f"`{benchmark.name}` requires a nonempty input, but got shape {x.shape}.")
        elif x.shape[0] != benchmark.dims:
            raise e.SizeError(
                f"`{benchmark.name}` requires {benchmark.dims} input dimensions, but got shape {x.shape}."
            )

        return function(benchmark, x)

    return _validate


def check_exact_dimension_and_auxiliary_matrix(function: Callable[..., _Result]) -> Callable[..., _Result]:
    """Require a supported CEC dimension and select its rotation matrix.

    Args:
        function: Benchmark method receiving an instance and a normalized NumPy input.

    Returns:
        Wrapped method that selects M2, M10, M30, or M50 as M before evaluation.

    Raises:
        SizeError: The input's first-axis length is not 2, 10, 30, or 50.

    Notes:
        Positional and keyword inputs use NumPy's asarray dtype inference and lose only a singleton second axis.
        Other axes are retained, extra positional arguments are ignored, and results are not converted.

    """

    @wraps(function)
    def _validate(benchmark: Any, x: ArrayLike, *args: Any) -> _Result:
        x = _vector(x)
        dimension = x.shape[0]
        if dimension not in {2, 10, 30, 50}:
            raise e.SizeError(
                f"`{benchmark.name}` requires 2, 10, 30, or 50 input dimensions, but got shape {x.shape}."
            )

        benchmark.M = getattr(benchmark, f"M{dimension}")
        return function(benchmark, x)

    return _validate


def check_less_equal_dimension(function: Callable[..., _Result]) -> Callable[..., _Result]:
    """Require an input no larger than the benchmark's maximum dimension.

    Args:
        function: Benchmark method receiving an instance and a normalized NumPy input.

    Returns:
        Wrapped method that checks the first-axis length against the benchmark's maximum dimension.

    Raises:
        SizeError: The input has more coordinates along its first axis than the benchmark permits.

    Notes:
        Positional and keyword inputs use NumPy's asarray dtype inference and lose only a singleton second axis.
        Other axes are retained, extra positional arguments are ignored, and results are not converted.

    """

    @wraps(function)
    def _validate(benchmark: Any, x: ArrayLike, *args: Any) -> _Result:
        x = _vector(x)
        if x.shape[0] > benchmark.dims:
            raise e.SizeError(
                f"`{benchmark.name}` requires at most {benchmark.dims} input dimensions, but got shape {x.shape}."
            )

        return function(benchmark, x)

    return _validate
