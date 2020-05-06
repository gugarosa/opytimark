import numpy as np

import opytimark.utils.constants as c
import opytimark.utils.decorator as d
from opytimark.core import Benchmark


class GramacyLee(Benchmark):
    """GramacyLee class implements the Gramacy & Lee's benchmarking function.

    .. math:: f(x) = \\frac{sin(10 \\pi x)}{2x} + (x - 1)^4

    Domain:
        The function is commonly evaluated using :math:`x \in [-0.5, 2.5]`.

    Global Minima:
        :math:`f(x^*) = -0.8690111349894997 \mid x^* = (0.548563444114526)`.

    """

    def __init__(self, name='GramacyLee', dims=-1, continuous=True, convex=False,
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
        super(GramacyLee, self).__init__(name, dims, continuous,
                                         convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Gramacy & Lee's function
        f = np.sin(10 * np.pi * x[0]) / (2 * x[0] + c.EPSILON) + ((x[0] - 1) ** 4)

        return np.sum(f)