import numpy as np

import opytimark.utils.decorator as d
from opytimark.core import Benchmark


class Ackley2(Benchmark):
    """Ackley2 class implements the Ackley's 2nd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = -200e^{-0.2\sqrt{x_1^2+x_2^2}}

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-32, 32], x_2 \in [-32, 32]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -200 \mid \mathbf{x^*} = (0, 0)`.

    """

    def __init__(self, name='Ackley2', dims=2, continuous=True, convex=True,
                 differentiable=True, multimodal=False, separable=False):
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
        super(Ackley2, self).__init__(name, dims, continuous,
                                      convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Ackley's 2nd function
        f = -200 * np.exp(-0.2 * np.sqrt(x[0] ** 2 + x[1] ** 2))

        return np.sum(f)


class Ackley3(Benchmark):
    """Ackley3 class implements the Ackley's 3rd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = -200e^{-0.2\sqrt{x_1^2+x_2^2}} + 5e^{cos(3x_1) + sin(3x_2)}

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-32, 32], x_2 \in [-32, 32]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) \\approx ? \mid \mathbf{x^*} \\approx (?, ?)`.

    """

    def __init__(self, name='Ackley3', dims=2, continuous=True, convex=False,
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
        super(Ackley3, self).__init__(name, dims, continuous,
                                      convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Ackley's 3rd function
        f = -200 * np.exp(-0.2 * np.sqrt(x[0] ** 2 + x[1] ** 2)) + \
            5 * np.exp(np.cos(3 * x[0]) + np.sin(3 * x[1]))

        return np.sum(f)


class Adjiman(Benchmark):
    """Adjiman class implements the Adjiman's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = cos(x_1) sin(x_2) - \\frac{x_1}{x_2^2+1}

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
            The benchmarking function output `f(x)`.

        """

        # Calculating the Adjiman's function
        f = np.cos(x[0]) * np.sin(x[1]) - (x[0] / (x[1] ** 2 + 1))

        return np.sum(f)
