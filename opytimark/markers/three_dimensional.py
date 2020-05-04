import numpy as np

import opytimark.utils.decorator as d
from opytimark.core import Benchmark


class Wolfe(Benchmark):
    """Wolfe class implements the Wolfe's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3) = \\frac{4}{3}(x_1^2 + x_2^2 - x_1x_2)^{0.75} + x_3

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 2] \mid i = \{1, 2, 3\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, 0)`.

    """

    def __init__(self, name='Wolfe', dims=3, continuous=True, convex=False,
                 differentiable=True, multimodal=True, separable=True):
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
        super(Wolfe, self).__init__(name, dims, continuous,
                                    convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `y`.

        """

        # Calculating the Wolfe's function
        y = 4 / 3 * ((x[0] ** 2 + x[1] ** 2 - x[0] * x[1]) ** 0.75) + x[2]

        return np.sum(y)
