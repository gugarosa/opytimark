"""Boolean-based benchmarking functions.
"""

import itertools as it

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

    def __init__(self, name='Knapsack', dims=-1, continuous=False, convex=False,
                 differentiable=False, multimodal=False, separable=False,
                 values=(0,), weights=(0,), max_capacity=0.0):
        """Initialization method.

        Args:
            name (str): Name of the function.
            dims (int): Number of allowed dimensions.
            continuous (bool): Whether the function is continuous.
            convex (bool): Whether the function is convex.
            differentiable (bool): Whether the function is differentiable.
            multimodal (bool): Whether the function is multimodal.
            separable (bool): Whether the function is separable.
            values (tuple): Tuple of items values.
            weights (tuple): Tuple of items weights.
            max_capacity: Maximum capacity of the knapsack.

        """

        # Override its parent class
        super(Knapsack, self).__init__(name, dims, continuous,
                                       convex, differentiable, multimodal, separable)

        # Checking if values and weights have the same length
        if len(values) != len(weights):
            raise e.SizeError('`values` and `weights` needs to have the same size')

        # Items values
        self.values = values

        # Items weights
        self.weights = weights

        # Maximum capacity of the knapsack
        self.max_capacity = max_capacity

        # Re-writes the correct number of dimensions
        self.dims = len(values)

    @property
    def values(self):
        """tuple: values of items in the knapsack.

        """

        return self._values

    @values.setter
    def values(self, values):
        if not isinstance(values, tuple):
            raise e.TypeError('`values` should be a tuple')

        self._values = values

    @property
    def weights(self):
        """tuple: Weights of items in the knapsack.

        """

        return self._weights

    @weights.setter
    def weights(self, weights):
        if not isinstance(weights, tuple):
            raise e.TypeError('`weights` should be a tuple')

        self._weights = weights

    @property
    def max_capacity(self):
        """float: Maximum capacity of the knapsack.

        """

        return self._max_capacity

    @max_capacity.setter
    def max_capacity(self, max_capacity):
        if not isinstance(max_capacity, (float, int)):
            raise e.TypeError('`max_capacity` should be a float or integer')
        if max_capacity < 0:
            raise e.ValueError('`max_capacity` should be >= 0')

        self._max_capacity = max_capacity

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

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
