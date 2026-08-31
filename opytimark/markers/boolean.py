"""Boolean benchmark functions."""

import itertools

import numpy as np

from opytimark.core import Benchmark
from opytimark.utils.constants import FLOAT_MAX
from opytimark.utils.decorator import check_exact_dimension


class Knapsack(Benchmark):
    r"""Boolean knapsack benchmark.

    .. math:: f(\mathbf{x}) = \min -{\sum_{i=1}^{n}v_i x_i}

    Subject to :math:`\sum_{i=1}^{n}w_i x_i \leq b`, where
    :math:`x_i \in \{0, 1\}`.
    """

    dims = -1

    def __init__(self, values=(0,), weights=(0,), max_capacity=0.0):
        if not isinstance(values, tuple):
            raise TypeError("values should be a tuple")
        if not isinstance(weights, tuple):
            raise TypeError("weights should be a tuple")
        if len(values) != len(weights):
            raise ValueError("values and weights should have the same size")
        if not isinstance(max_capacity, (float, int)):
            raise TypeError("max_capacity should be a number")
        if max_capacity < 0:
            raise ValueError("max_capacity should be non-negative")

        self.values = values
        self.weights = weights
        self.max_capacity = max_capacity
        self.dims = len(values)

    @check_exact_dimension
    def __call__(self, x):
        values = np.array(list(itertools.compress(self.values, x)))
        weights = np.array(list(itertools.compress(self.weights, x)))

        if np.sum(weights) > self.max_capacity:
            return FLOAT_MAX

        return -np.sum(values)
