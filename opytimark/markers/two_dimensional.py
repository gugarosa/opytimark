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

        return f


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

        return f


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

        return f


class BartelsConn(Benchmark):
    """BartelsConn class implements the Bartels Conn's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = |x_1^2+x_2^2+x_1x_2| + |sin(x_1)| + |cos(x_2)|

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 1 \mid \mathbf{x^*} = (0, 0)`.

    """

    def __init__(self, name='BartelsConn', dims=2, continuous=True, convex=False,
                 differentiable=False, multimodal=True, separable=False):
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
        super(BartelsConn, self).__init__(name, dims, continuous,
                                          convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Bartels Conn's function
        f = np.fabs(x[0] ** 2 + x[1] ** 2 + x[0] * x[1]) + \
            np.fabs(np.sin(x[0])) + np.fabs(np.cos(x[1]))

        return f


class Beale(Benchmark):
    """Beale class implements the Beale's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = (1.5-x_1+x_1x_2)^2 + (2.25-x_1+x_1x_2^2)^2 + (2.625-x_1+x_1x_2^3)^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-4.5, 4.5], x_2 \in [-4.5, 4.5]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (3, 0.5)`.

    """

    def __init__(self, name='Beale', dims=2, continuous=True, convex=False,
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
        super(Beale, self).__init__(name, dims, continuous,
                                    convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Beale's function
        f = (1.5 - x[0] + x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2 + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2

        return f


class BiggsExponential2(Benchmark):
    """BiggsExponential2 class implements the Biggs Exponential's 2nd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = \sum_{i=1}^{10}(e^{-t_ix_1} - 5e^{-t_ix_2} - y_i)^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [0, 20], x_2 \in [0, 20]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 10)`.

    """

    def __init__(self, name='BiggsExponential2', dims=2, continuous=True, convex=False,
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
        super(BiggsExponential2, self).__init__(name, dims, continuous,
                                                convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Instantiating function
        f = 0

        # For `i` ranging from 0 to 10
        for i in range(1, 11):
            # Calculating `z`
            z = i / 10

            # Calculating partial `y`
            y = np.exp(-z) - 5 * np.exp(-10 * z)
            
            # Calculating Biggs Exponential's 2nd function
            f += (np.exp(-z * x[0]) - 5 * np.exp(-z * x[1]) - y) ** 2

        return f


class Bird(Benchmark):
    """Bird class implements the Bird's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = sin(x_1)e^{(1-cos(x_2))^2}+cos(x_2)e^{(1-sin(x_1))^2}+(x_1-x_2)^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-2\\pi, 2\\pi], x_2 \in [-2\\pi, 2\\pi]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -106.764537 \mid \mathbf{x^*} = (4.70104, 3.15294) or (−1.58214, −3.13024)`.

    """

    def __init__(self, name='Bird', dims=2, continuous=True, convex=False,
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
        super(Bird, self).__init__(name, dims, continuous,
                                   convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Bird's function
        f = np.sin(x[0]) * np.exp((1 - np.cos(x[1])) ** 2) + np.cos(x[1]) * np.exp((1 - np.sin(x[0])) ** 2) + (x[0] - x[1]) ** 2

        return f
