import itertools as it

import numpy as np
import opytimark.utils.decorator as d
import opytimark.utils.exception as e
from opytimark.core import Benchmark


class Knapsack(Benchmark):
    """Knapsack class implements a boolean-based version of the Knapsack problem.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \max{\sum_{i=1}^{n}v_i x_i}

    s.t.

    .. math:: \sum_{i=1}^{n}w_i x_i \leq b

    Domain:
        The function is evaluated using :math:`x_i \in \{0, 1\} \mid i = \{1, 2, \ldots, n\}`.        

    """

    def __init__(self, name='Knapsack', dims=-1, continuous=False, convex=False,
                 differentiable=False, multimodal=False, separable=False, costs=None, weights=None, max_capacity=0):
        """Initialization method.

        Args:
            name (str): Name of the function.
            dims (int): Number of allowed dimensions.
            continuous (bool): Whether the function is continuous.
            convex (bool): Whether the function is convex.
            differentiable (bool): Whether the function is differentiable.
            multimodal (bool): Whether the function is multimodal.
            separable (bool): Whether the function is separable.
            costs (list): List of items costs.
            weights (list): List of items weights.
            max_capacity: Maximum capacity of the knapsack.

        """

        # Override its parent class
        super(Knapsack, self).__init__(name, dims, continuous,
                                       convex, differentiable, multimodal, separable)

        # Checking if costs and weights have the same length
        if len(costs) != len(weights):
            raise e.SizeError('`costs` and `weights` needs to have the same size')

        #
        self.costs = costs

        #
        self.weights = weights

        #
        self.max_capacity = max_capacity

        #
        self.dims = len(costs)
   

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        fixedCapacity = self.max_capacity

        profit = list(it.compress(self.costs, x))
        capacity = list(it.compress(self.weights, x))
        n = len(capacity)

        mat = [[0 for i in range(fixedCapacity + 1)]
               for i in range(2)]

        i = 0
        while i < n:
            j = 0
            if i % 2 == 0:
                while j < fixedCapacity:
                    j += 1
                    if capacity[i] <= j:
                        mat[1][j] = max(profit[i] + mat[0][j -
                                                           capacity[i]], mat[0][j])
                    else:
                        mat[1][j] = mat[0][j]

            else:
                while j < fixedCapacity:
                    j += 1
                    if capacity[i] <= j:
                        mat[0][j] = max(profit[i] + mat[1][j -
                                                           capacity[i]], mat[1][j])
                    else:
                        mat[0][j] = mat[1][j]
            i += 1

        if n % 2 == 0:
            return -mat[0][fixedCapacity]
        else:
            return -mat[1][fixedCapacity]
