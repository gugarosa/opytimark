"""Boolean benchmark functions."""

import itertools as it
from typing import Optional, Tuple, Union

import numpy as np

import opytimark.utils.constants as c
import opytimark.utils.decorator as d
import opytimark.utils.exception as e
from opytimark.core import Benchmark


class Knapsack(Benchmark):
    r"""Boolean knapsack benchmark.

    .. math:: f(\mathbf{x}) = \min -{\sum_{i=1}^{n}v_i x_i}

    Subject to :math:`\sum_{i=1}^{n}w_i x_i \leq b`, where
    :math:`x_i \in \{0, 1\}`.
    """

    def __init__(
        self,
        name: Optional[str] = "Knapsack",
        dims: Optional[int] = -1,
        continuous: Optional[bool] = False,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = False,
        multimodal: Optional[bool] = False,
        separable: Optional[bool] = False,
        values: Optional[Tuple[Union[float, int], ...]] = (0,),
        weights: Optional[Tuple[Union[float, int], ...]] = (0,),
        max_capacity: Optional[Union[float, int]] = 0.0,
    ):
        super().__init__(
            name,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )
        if len(values) != len(weights):
            raise e.SizeError("`values` and `weights` needs to have the same size")

        self.values = values
        self.weights = weights
        self.max_capacity = max_capacity
        self.dims = len(values)

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        if not isinstance(values, tuple):
            raise e.TypeError("`values` should be a tuple")
        self._values = values

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        if not isinstance(weights, tuple):
            raise e.TypeError("`weights` should be a tuple")
        self._weights = weights

    @property
    def max_capacity(self):
        return self._max_capacity

    @max_capacity.setter
    def max_capacity(self, max_capacity):
        if not isinstance(max_capacity, (float, int)):
            raise e.TypeError("`max_capacity` should be a float or integer")
        if max_capacity < 0:
            raise e.ValueError("`max_capacity` should be >= 0")
        self._max_capacity = max_capacity

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:
        v = np.array(list(it.compress(self.values, x)))
        w = np.array(list(it.compress(self.weights, x)))

        if np.sum(w) > self.max_capacity:
            return c.FLOAT_MAX

        return -np.sum(v)
