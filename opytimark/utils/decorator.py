"""Input validation decorators."""

from functools import wraps

import numpy as np

import opytimark.utils.exception as e


def _vector(x):
    x = np.asarray(x)
    try:
        return np.squeeze(x, axis=1)
    except ValueError:
        return x


def check_exact_dimension(function):
    """Require the exact dimension declared by a benchmark."""

    @wraps(function)
    def validate(*args):
        benchmark, x = args[0], _vector(args[1])

        if benchmark.dims == -1:
            if not x.shape[0]:
                raise e.SizeError(f"{benchmark.name} input should be n-dimensional")
        elif x.shape[0] != benchmark.dims:
            raise e.SizeError(
                f"{benchmark.name} input should be {benchmark.dims}-dimensional"
            )

        return function(benchmark, x)

    return validate


def check_exact_dimension_and_auxiliary_matrix(function):
    """Require a supported CEC dimension and select its rotation matrix."""

    @wraps(function)
    def validate(*args):
        benchmark, x = args[0], _vector(args[1])
        dimension = x.shape[0]
        if dimension not in {2, 10, 30, 50}:
            raise e.SizeError(
                f"{benchmark.name} input should be 2-, 10-, 30- or 50-dimensional"
            )

        benchmark.M = getattr(benchmark, f"M{dimension}")
        return function(benchmark, x)

    return validate


def check_less_equal_dimension(function):
    """Require an input no larger than the benchmark's maximum dimension."""

    @wraps(function)
    def validate(*args):
        benchmark, x = args[0], _vector(args[1])
        if x.shape[0] > benchmark.dims:
            raise e.SizeError(
                f"{benchmark.name} input should be less or equal to "
                f"{benchmark.dims}-dimensional"
            )

        return function(benchmark, x)

    return validate
