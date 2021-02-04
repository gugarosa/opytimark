"""CEC2005 benchmarking functions.
"""

import numpy as np

import opytimark.utils.decorator as d
import opytimark.utils.loader as l
from opytimark.core import Benchmark


class F1(Benchmark):
    """F1 class implements the Shifted Sphere's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} z_i^2 - 450 \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -450 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(self, name='F1', dims=-1, continuous=True, convex=False,
                 differentiable=True, multimodal=False, separable=True):
        """Initialization method.

        Args:
            name (str): Name of the function.
            dims (int): Number of allowed dimensions.
            continuous (bool): Whether the function is continuous.
            convex (bool): Whether the function is convex.
            differentiable (bool): Whether the function is differentiable.
            multimodal (bool): Whether the function is multimodal.
            separable (bool): Whether the function is separable.

        """

        # Override its parent class
        super(F1, self).__init__(name, dims, continuous,
                                 convex, differentiable, multimodal, separable)

        # Loads auxiliary data and define it as a property
        self.o = l.load_cec_auxiliary('F1', '2005')

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Re-calculates the input
        z = x - self.o[:x.shape[0]]

        # Calculating the Shifted Sphere's function
        f = z ** 2

        return np.sum(f) - 450
