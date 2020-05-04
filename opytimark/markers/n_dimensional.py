import numpy as np

import opytimark.utils.decorator as d
from opytimark.core import Benchmark


class Sphere(Benchmark):
    """Sphere class implements the Sphere's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} x_i^2

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5.12, 5.12] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='Sphere', dims=-1, continuous=True, convex=True,
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
        super(Sphere, self).__init__(name, dims, continuous,
                                     convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `y`.

        """

        # Calculating the sphere's function
        y = x ** 2

        return np.sum(y)
