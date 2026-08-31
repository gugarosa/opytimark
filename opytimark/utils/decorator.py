"""Input validation decorators."""

from functools import wraps

import numpy as np


def _vector(x):
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 1:
        return x[:, 0]
    return x


def check_exact_dimension(function):
    """Require the exact dimension declared by a benchmark."""

    @wraps(function)
    def validate(benchmark, x):
        x = _vector(x)
        name = type(benchmark).__name__

        if benchmark.dims == -1:
            if not x.shape[0]:
                raise ValueError(f"{name} input should be n-dimensional")
        elif x.shape[0] != benchmark.dims:
            raise ValueError(f"{name} input should be {benchmark.dims}-dimensional")

        return function(benchmark, x)

    return validate


def check_exact_dimension_and_auxiliary_matrix(function):
    """Require a supported CEC dimension and select its rotation matrix."""

    @wraps(function)
    def validate(benchmark, x):
        x = _vector(x)
        dimension = x.shape[0]
        if dimension not in {2, 10, 30, 50}:
            raise ValueError(
                f"{type(benchmark).__name__} input should be "
                "2-, 10-, 30- or 50-dimensional"
            )

        benchmark.M = getattr(benchmark, f"M{dimension}")
        return function(benchmark, x)

    return validate


def check_less_equal_dimension(function):
    """Require an input no larger than the benchmark's maximum dimension."""

    @wraps(function)
    def validate(benchmark, x):
        x = _vector(x)
        if x.shape[0] > benchmark.dims:
            raise ValueError(
                f"{type(benchmark).__name__} input should be less than or equal "
                f"to {benchmark.dims}-dimensional"
            )

        return function(benchmark, x)

    return validate
