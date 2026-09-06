# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import itertools as it
from typing import Any

import numpy as np
from numpy.typing import NDArray

import opytimark.utils.constants as c
import opytimark.utils.decorator as d
import opytimark.utils.exception as e
from opytimark.core import Benchmark
from opytimark.typing import BenchmarkValue


class Knapsack(Benchmark):
    r"""Boolean knapsack benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = \min -{\sum_{i=1}^{n}v_i x_i}

        Subject to :math:`\sum_{i=1}^{n}w_i x_i \leq b`, where
        :math:`x_i \in \{0, 1\}`.

    """

    def __init__(
        self,
        name: str = "Knapsack",
        dims: int = -1,
        continuous: bool = False,
        convex: bool = False,
        differentiable: bool = False,
        multimodal: bool = False,
        separable: bool = False,
        values: tuple[float | int, ...] = (0,),
        weights: tuple[float | int, ...] = (0,),
        max_capacity: float | int = 0.0,
    ) -> None:
        """Initialize the knapsack benchmark.

        Args:
            name: Benchmark name.
            dims: Initial dimension metadata before the item count is applied.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function has multiple local optima.
            separable: Whether coordinates can be optimized independently.
            values: Numeric values of the available items.
            weights: Numeric weights of the available items.
            max_capacity: Maximum permitted total weight.

        Raises:
            SizeError: Values and weights have different lengths.
            TypeError: Metadata or item collections have unsupported types.
            ValueError: Dimensions or capacity violate their constraints.

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

        if len(values) != len(weights):
            raise e.SizeError("`values` and `weights` must have the same size.")

        self.values = values
        self.weights = weights
        self.max_capacity = max_capacity
        self.dims = len(values)

    @property
    def values(self) -> tuple[float | int, ...]:
        """Return the item values.

        Returns:
            Numeric values of the available items.

        """

        return self._values

    @values.setter
    def values(self, values: tuple[float | int, ...]) -> None:
        """Set the item values.

        Args:
            values: Numeric values of the available items.

        Raises:
            TypeError: Values are not supplied as a tuple.

        """

        if not isinstance(values, tuple):
            raise e.TypeError(f"`values` must be a tuple, but got {values!r}.")

        self._values = values

    @property
    def weights(self) -> tuple[float | int, ...]:
        """Return the item weights.

        Returns:
            Numeric weights of the available items.

        """

        return self._weights

    @weights.setter
    def weights(self, weights: tuple[float | int, ...]) -> None:
        """Set the item weights.

        Args:
            weights: Numeric weights of the available items.

        Raises:
            TypeError: Weights are not supplied as a tuple.

        """

        if not isinstance(weights, tuple):
            raise e.TypeError(f"`weights` must be a tuple, but got {weights!r}.")

        self._weights = weights

    @property
    def max_capacity(self) -> float | int:
        """Return the maximum total weight.

        Returns:
            Maximum permitted total item weight.

        """

        return self._max_capacity

    @max_capacity.setter
    def max_capacity(self, max_capacity: float | int) -> None:
        """Set the maximum total weight.

        Args:
            max_capacity: Nonnegative weight capacity.

        Raises:
            TypeError: Capacity is not a float or integer.
            ValueError: Capacity is negative.

        """

        if not isinstance(max_capacity, (float, int)):
            raise e.TypeError(f"`max_capacity` must be a float or integer, but got {max_capacity!r}.")
        if max_capacity < 0:
            raise e.ValueError(f"`max_capacity` must be nonnegative, but got {max_capacity!r}.")

        self._max_capacity = max_capacity

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Knapsack benchmark.

        Args:
            x: Item-selection indicators supplied as a vector or single-column array.

        Returns:
            Negative selected-item value or FLOAT_MAX when the capacity is exceeded.

        Raises:
            SizeError: The selection size differs from the item count.

        """

        v = np.array(list(it.compress(self.values, x)))
        w = np.array(list(it.compress(self.weights, x)))

        if np.sum(w) > self.max_capacity:
            return c.FLOAT_MAX

        return -np.sum(v)
