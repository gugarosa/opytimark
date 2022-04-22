"""Boolean-based benchmarking functions.
"""

import itertools as it
from typing import Optional, Tuple, Union

import numpy as np

import opytimark.utils.constants as c
import opytimark.utils.decorator as d
import opytimark.utils.exception as e
from opytimark.core import Benchmark


class Knapsack(Benchmark):
    """Knapsack class implements a boolean-based version of the Knapsack problem.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \min -{\sum_{i=1}^{n}v_i x_i}

    s.t.

    .. math:: \sum_{i=1}^{n}w_i x_i \leq b

    Domain:
        The function is evaluated using :math:`x_i \in \{0, 1\} \mid i = \{1, 2, \ldots, n\}`.

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
        """Initialization method.

        Args:
            name: Name of the function.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.
            values: Tuple of items values.
            weights: Tuple of items weights.
            max_capacity: Maximum capacity of the knapsack.

        """

        super(Knapsack, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable
        )

        if len(values) != len(weights):
            raise e.SizeError("`values` and `weights` needs to have the same size")

        # Items values
        self.values = values

        # Items weights
        self.weights = weights

        # Maximum capacity of the knapsack
        self.max_capacity = max_capacity

        # Re-writes the correct number of dimensions
        self.dims = len(values)

    @property
    def values(self) -> Tuple[Union[float, int], ...]:
        """Values of items in the knapsack."""

        return self._values

    @values.setter
    def values(self, values: Tuple[Union[float, int], ...]) -> None:
        if not isinstance(values, tuple):
            raise e.TypeError("`values` should be a tuple")

        self._values = values

    @property
    def weights(self) -> Tuple[Union[float, int], ...]:
        """Weights of items in the knapsack."""

        return self._weights

    @weights.setter
    def weights(self, weights: Tuple[Union[float, int], ...]) -> None:
        if not isinstance(weights, tuple):
            raise e.TypeError("`weights` should be a tuple")

        self._weights = weights

    @property
    def max_capacity(self) -> Union[float, int]:
        """Maximum capacity of the knapsack."""

        return self._max_capacity

    @max_capacity.setter
    def max_capacity(self, max_capacity: Union[float, int]) -> None:
        if not isinstance(max_capacity, (float, int)):
            raise e.TypeError("`max_capacity` should be a float or integer")
        if max_capacity < 0:
            raise e.ValueError("`max_capacity` should be >= 0")

        self._max_capacity = max_capacity

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Gathering an array of possible values
        v = np.array(list(it.compress(self.values, x)))

        # Gathering an array of possible weights
        w = np.array(list(it.compress(self.weights, x)))

        # If the sum of weights exceed the maximum capacity
        if np.sum(w) > self.max_capacity:
            # Returns the maximum number possible
            return c.FLOAT_MAX

        # Returns its negative sum as it is a minimization problem
        return -np.sum(v)
