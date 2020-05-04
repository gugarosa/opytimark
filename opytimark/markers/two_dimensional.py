import numpy as np

import opytimark.utils.decorator as d
from opytimark.core import Benchmark


class Adjiman(Benchmark):
    """Adjiman class implements the Adjiman's benchmarking function.

    .. math:: f(\mathbf{x}) = = f(x_1, x_2) = cos(x_1) sin(x_2) - \\frac{x_1}{x_2^2+1}

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-1, 2], x_2 \in [-1, 1]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -2.02181 \mid \mathbf{x^*} = (2, 0.10578)`.

    """

    def __init__(self, name='Adjiman', dims=2, continuous=True, convex=False,
                 differentiable=True, multimodal=True, separable=False):
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
        super(Adjiman, self).__init__(name, dims, continuous,
                                      convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `y`.

        """

        # Calculating the Adjiman's function
        y = np.cos(x[0]) * np.sin(x[1]) - (x[0] / (x[1] ** 2 + 1))

        return np.sum(y)
