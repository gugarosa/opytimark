"""Two-dimensional benchmarking functions.
"""

import numpy as np

import opytimark.utils.constants as c
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

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = -200e^{-0.02\sqrt{x_1^2+x_2^2}} + 5e^{cos(3x_1) + sin(3x_2)}

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-32, 32], x_2 \in [-32, 32]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) \\approx -195.62902823841935 \mid \mathbf{x^*} \\approx (\pm 0.682584587365898, -0.36075325513719)`.

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
        f = -200 * np.exp(-0.02 * np.sqrt(x[0] ** 2 + x[1] ** 2)) + \
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
        f = (1.5 - x[0] + x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0]
                                               * x[1] ** 2) ** 2 + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2

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
        :math:`f(\mathbf{x^*}) = -106.764537 \mid \mathbf{x^*} = (4.70104, 3.15294) ~ or ~(−1.58214, −3.13024)`.

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
        f = np.sin(x[0]) * np.exp((1 - np.cos(x[1])) ** 2) + np.cos(x[1]
                                                                    ) * np.exp((1 - np.sin(x[0])) ** 2) + (x[0] - x[1]) ** 2

        return f


class Bohachevsky1(Benchmark):
    """Bohachevsky1 class implements the Bohachevsky's 1st benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = x_1^2 + 2x_2^2 - 0.3cos(3\\pi x_1) - 0.4cos(4\\pi x_2) + 0.7

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-100, 100], x_2 \in [-100, 100]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0)`.

    """

    def __init__(self, name='Bohachevsky1', dims=2, continuous=True, convex=True,
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
        super(Bohachevsky1, self).__init__(name, dims, continuous,
                                           convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Bohachevsky's 1st function
        f = x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * \
            np.cos(3 * np.pi * x[0]) - 0.4 * np.cos(4 * np.pi * x[1]) + 0.7

        return f


class Bohachevsky2(Benchmark):
    """Bohachevsky2 class implements the Bohachevsky's 2nd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = x_1^2 + 2x_2^2 - 0.3cos(3\\pi x_1)cos(4\\pi x_2) + 0.7

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-100, 100], x_2 \in [-100, 100]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0)`.

    """

    def __init__(self, name='Bohachevsky2', dims=2, continuous=True, convex=False,
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
        super(Bohachevsky2, self).__init__(name, dims, continuous,
                                           convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Bohachevsky's 2nd function
        f = x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * \
            np.cos(3 * np.pi * x[0]) * np.cos(4 * np.pi * x[1]) + 0.3

        return f


class Bohachevsky3(Benchmark):
    """Bohachevsky3 class implements the Bohachevsky's 3rd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = x_1^2 + 2x_2^2 - 0.3cos(3\\pi x_1 + 4\\pi x_2) + 0.3

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-100, 100], x_2 \in [-100, 100]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0)`.

    """

    def __init__(self, name='Bohachevsky3', dims=2, continuous=True, convex=False,
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
        super(Bohachevsky3, self).__init__(name, dims, continuous,
                                           convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Bohachevsky's 3rd function
        f = x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * \
            np.cos(3 * np.pi * x[0] + 4 * np.pi * x[1]) + 0.3

        return f


class Booth(Benchmark):
    """Booth class implements the Booth's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = (x_1 + 2x_2 - 7)^2 + (2x_1 + x_2 - 5)^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 3)`.

    """

    def __init__(self, name='Booth', dims=2, continuous=True, convex=False,
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
        super(Booth, self).__init__(name, dims, continuous,
                                    convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Booth's function
        f = (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

        return f


class BraninHoo(Benchmark):
    """BraninHoo class implements the Branin Hoo's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = (x_2 - \\frac{5.1x_1^2}{4\\pi^2} + \\frac{5x_1}{\\pi} - 6)^2 + 10(1 - \\frac{1}{8\\pi})cos(x_1) + 10

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-5, 10], x_2 \in [0, 15]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0.39788735775266204 \mid \mathbf{x^*} = (-\\pi, 12.275) ~ or ~(\\pi, 2.275) ~ or ~(3\\pi, 2.425)`.

    """

    def __init__(self, name='BraninHoo', dims=2, continuous=True, convex=False,
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
        super(BraninHoo, self).__init__(name, dims, continuous,
                                        convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Branin Hoo's function
        f = (x[1] - ((5.1 * x[0] ** 2) / (4 * np.pi ** 2)) + ((5 * x[0]) /
                                                              np.pi) - 6) ** 2 + 10 * (1 - (1 / (8 * np.pi))) * np.cos(x[0]) + 10

        return f


class Brent(Benchmark):
    """Brent class implements the Brent's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = (x_1 + 10)^2 + (x_2 + 10)^2 + e^{-x_1^2 - x_2^2}

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = e^{-200} \mid \mathbf{x^*} = (-10, -10)`.

    """

    def __init__(self, name='Brent', dims=2, continuous=True, convex=True,
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
        super(Brent, self).__init__(name, dims, continuous,
                                    convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Brent's function
        f = (x[0] + 10) ** 2 + (x[1] + 10) ** 2 + \
            np.exp(-(x[0] ** 2) - (x[1] ** 2))

        return f


class Bukin2(Benchmark):
    """Bukin2 class implements the Bukin's 2nd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = 100(x_2 - 0.01x_1^2 + 1) + 0.01(x_1 + 10)^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-15, -5], x_2 \in [-3, 3]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = # \mid \mathbf{x^*} = (-10, 0)`.

    """

    def __init__(self, name='Bukin2', dims=2, continuous=True, convex=False,
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
        super(Bukin2, self).__init__(name, dims, continuous,
                                     convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Bukin's 2nd function
        f = 100 * (x[1] - 0.01 * x[0] ** 2 + 1) + 0.01 * (x[0] + 10) ** 2

        return f


class Bukin4(Benchmark):
    """Bukin4 class implements the Bukin's 4th benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = 100x_2^2 + 0.01|x_1 + 10|

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-15, -5], x_2 \in [-3, 3]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = # \mid \mathbf{x^*} = (-10, 0)`.

    """

    def __init__(self, name='Bukin4', dims=2, continuous=True, convex=False,
                 differentiable=False, multimodal=True, separable=True):
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
        super(Bukin4, self).__init__(name, dims, continuous,
                                     convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Bukin's 4th function
        f = 100 * x[1] ** 2 + 0.01 * np.fabs(x[0] + 10)

        return f


class Bukin6(Benchmark):
    """Bukin6 class implements the Bukin's 6th benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = 100sqrt{|x_2-0.01x_1^2|} + 0.01|x_1+10|

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-15, -5], x_2 \in [-3, 3]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = # \mid \mathbf{x^*} = (-10, 1)`.

    """

    def __init__(self, name='Bukin6', dims=2, continuous=True, convex=False,
                 differentiable=False, multimodal=True, separable=True):
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
        super(Bukin6, self).__init__(name, dims, continuous,
                                     convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Bukin's 6th function
        f = 100 * np.sqrt(np.fabs(x[1] - 0.01 * x[0] ** 2)
                          ) + 0.01 * np.fabs(x[0] + 10)

        return f


class Camel3(Benchmark):
    """Camel3 class implements the Camel's Three Hump benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = 2x_1^2 - 1.05x_1^4 + \\frac{x_1^6}{6} + x_1x_2 + x_2^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-5, 5], x_2 \in [-5, 5]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0)`.

    """

    def __init__(self, name='Camel3', dims=2, continuous=True, convex=False,
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
        super(Camel3, self).__init__(name, dims, continuous,
                                     convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Camel's Three Hump function
        f = 2 * x[0] ** 2 - 1.05 * x[0] ** 4 + \
            x[0] ** 6 / 6 + x[0] * x[1] + x[1] ** 2

        return f


class Camel6(Benchmark):
    """Camel6 class implements the Camel's Six Hump benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = (4 - 2.1x_1^2 + \\frac{x_1^4}{3})x_1^2 + x_1x_2 + (4x_2^2 - 4)x_2^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-5, 5], x_2 \in [-5, 5]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -1.0316284229280819 \mid \mathbf{x^*} = (−0.0898, 0.7126) ~ or ~(0.0898,−0.7126)`.

    """

    def __init__(self, name='Camel6', dims=2, continuous=True, convex=False,
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
        super(Camel6, self).__init__(name, dims, continuous,
                                     convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Camel's Six Hump function
        f = (4 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3) * x[0] ** 2 + \
            x[0] * x[1] + (4 * x[1] ** 2 - 4) * x[1] ** 2

        return f


class ChenBird(Benchmark):
    """ChenBird class implements the Chen Bird's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = -\\frac{0.001}{0.001^2 + (x_1^2 + x_2^2 - 1)^2} - \\frac{0.001}{0.001^2 + (x_1^2 + x_2^2 - 0.5)^2} - \\frac{0.001}{0.001^2 + (x_1^2 - x_2^2)^2}

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 2000.0039999840003 \mid \mathbf{x^*} = (0.5, 0.5)`.

    """

    def __init__(self, name='ChenBird', dims=2, continuous=True, convex=False,
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
        super(ChenBird, self).__init__(name, dims, continuous,
                                       convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Chen Bird's function
        f = -((0.001) / (0.001 ** 2 + (x[0] ** 2 + x[1] ** 2 - 1) ** 2)) - ((0.001) / (0.001 ** 2 + (
            x[0] ** 2 + x[1] ** 2 - 0.5) ** 2)) - ((0.001) / (0.001 ** 2 + (x[0] ** 2 - x[1] ** 2) ** 2))

        return f


class ChenV(Benchmark):
    """ChenV class implements the Chen V's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = \\frac{0.001}{0.001^2 + (x_1 - 0.4x_2 - 0.1)^2} + \\frac{0.001}{0.001^2 + (2x_1 + x_2 - 1.5)^2}

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 2000.0000000000002 \mid \mathbf{x^*} = (0.5, 0.5)`.

    """

    def __init__(self, name='ChenV', dims=2, continuous=True, convex=False,
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
        super(ChenV, self).__init__(name, dims, continuous,
                                    convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Chen V's function
        f = ((0.001) / (0.001 ** 2 + (x[0] - 0.4 * x[1] - 0.1) ** 2)) + (
            (0.001) / (0.001 ** 2 + (2 * x[0] + x[1] - 1.5) ** 2))

        return f


class Chichinadze(Benchmark):
    """Chichinadze class implements the Chichinadze's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = x_1^2 - 12x_1 + 11 + 10cos(\\frac{\\pi x_1}{2}) + 8sin(\\frac{5\\pi x_1}{2}) - \\frac{1}{5}^{0.5} e^{-0.5(x_2-0.5)^2}

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-30, 30], x_2 \in [-30, 30]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = −43.3159 \mid \mathbf{x^*} = (5.90133, 0.5)`.

    """

    def __init__(self, name='Chichinadze', dims=2, continuous=True, convex=False,
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
        super(Chichinadze, self).__init__(name, dims, continuous,
                                          convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Chichinadze's function
        f = x[0] ** 2 - 12 * x[0] + 11 + 10 * np.cos((np.pi * x[0]) / 2) + 8 * np.sin(
            (5 * np.pi * x[0]) / 2) - (1 / 5) ** 0.5 * np.exp(-0.5 * (x[1] - 0.5) ** 2)

        return f


class CrossTray(Benchmark):
    """CrossTray class implements the CrossTray's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = -0.0001(|sin(x_1)sin(x_2)e^{|100-\\frac{sqrt{x_1^2 + x_2^2}}{\\pi}|}| + 1)^0.1

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = −2.06261218 \mid \mathbf{x^*} = (\pm 1.349406685353340, \pm 1.349406608602084)`.

    """

    def __init__(self, name='CrossTray', dims=2, continuous=True, convex=False,
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
        super(CrossTray, self).__init__(name, dims, continuous,
                                        convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the CrossTray's function
        f = -0.0001 * (np.fabs(np.sin(x[0]) * np.sin(x[1]) * np.exp(
            np.fabs(100 - (np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi)))) + 1) ** 0.1

        return f


class Cube(Benchmark):
    """Cube class implements the Cube's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = 100(x_2 - x_1^3)^2 + (1 - x_1)^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (-1, 1)`.

    """

    def __init__(self, name='Cube', dims=2, continuous=True, convex=False,
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
        super(Cube, self).__init__(name, dims, continuous,
                                   convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Cube's function
        f = 100 * (x[1] - x[0] ** 3) ** 2 + (1 - x[0]) ** 2

        return f


class Damavandi(Benchmark):
    """Damavandi class implements the Damavandi's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = (1 - |\\frac{sin(\\pi (x_1-2))sin(\\pi (x_2-2))}{\\pi^2 (x_1-2)(x2_2)}|^5)(2 + (x_1-7)^2 + 2(x_2-7)^2)

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [0, 14], x_2 \in [0, 14]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (2.00000001, 2.00000001)`.

    """

    def __init__(self, name='Damavandi', dims=2, continuous=True, convex=False,
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
        super(Damavandi, self).__init__(name, dims, continuous,
                                        convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Damavandi's function
        # f = (1 - np.fabs((np.sin(np.pi * (x[0] - 2)) * np.sin(np.pi * (x[1] - 2))) / (np.pi ** 2 * (x[0] - 2) * (x[1] - 2)) + c.EPSILON) ** 5) * (2 + (x[0] - 7) ** 2 + 2 * (x[1] - 7) ** 2)
        f = (1 - np.fabs((np.sin(np.pi * (x[0] - 2)) * np.sin(np.pi * (x[1] - 2))) / (np.pi ** 2 * (
            x[0] - 2) * (x[1] - 2) + c.EPSILON)) ** 5) * (2 + (x[0] - 7) ** 2 + 2 * (x[1] - 7) ** 2)

        return f


class DeckkersAarts(Benchmark):
    """DeckkersAarts class implements the Deckkers Aarts' benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = 10^5x_1^2 + x_2^2 - (x_1^2 + x_2^2)^2 + 10^{-5}(x_1^2 + x_2^2)^4

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-20, 20], x_2 \in [-20, 20]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -24771.093749999996 \mid \mathbf{x^*} = (0, \pm 15)`.

    """

    def __init__(self, name='DeckkersAarts', dims=2, continuous=True, convex=False,
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
        super(DeckkersAarts, self).__init__(name, dims, continuous,
                                            convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Deckkers Aarts' function
        f = 10 ** 5 * x[0] ** 2 + x[1] ** 2 - \
            (x[0] ** 2 + x[1] ** 2) ** 2 + 10 ** - \
            5 * (x[0] ** 2 + x[1] ** 2) ** 4

        return f


class DropWave(Benchmark):
    """DropWave class implements the Drop Wave's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = - \\frac{1 + cos(12\sqrt{x_1^2+x_2^2})}{0.5(x_1^2+x_2^2) + 2}

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-5.2, 5.2], x_2 \in [-5.2, 5.2]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -1 \mid \mathbf{x^*} = (0, 0)`.

    """

    def __init__(self, name='DropWave', dims=2, continuous=True, convex=False,
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
        super(DropWave, self).__init__(name, dims, continuous,
                                       convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Drop Wave's function
        f = - (1 + np.cos(12 * np.sqrt(x[0] ** 2 + x[1] ** 2))
               ) / (0.5 * (x[0] ** 2 + x[1] ** 2) + 2)

        return f


class Easom(Benchmark):
    """Easom class implements the Easom's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = -cos(x_1)cos(x_2)e^{-(x_1-\\pi)^2 -(x_2-\\pi)^2}

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-100, 100], x_2 \in [-100, 100]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -1 \mid \mathbf{x^*} = (\\pi, \\pi)`.

    """

    def __init__(self, name='Easom', dims=2, continuous=True, convex=False,
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
        super(Easom, self).__init__(name, dims, continuous,
                                    convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Easom's function
        f = -np.cos(x[0]) * np.cos(x[1]) * \
            np.exp(-(x[0] - np.pi) ** 2 - (x[1] - np.pi) ** 2)

        return f


class ElAttarVidyasagarDutta(Benchmark):
    """ElAttarVidyasagarDutta class implements the El-Attar-Vidyasagar-Dutta's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = (x_1^2 + x_2 - 10)^2 + (x_1 + x_2^2 - 7)^2 + (x_1^2 + x_2^3 - 1)^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 1.7127803548622027 \mid \mathbf{x^*} = (3.4091868222, −2.1714330361)`.

    """

    def __init__(self, name='ElAttarVidyasagarDutta', dims=2, continuous=True, convex=False,
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
        super(ElAttarVidyasagarDutta, self).__init__(name, dims, continuous,
                                                     convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the El-Attar-Vidyasagar-Dutta's function
        f = (x[0] ** 2 + x[1] - 10) ** 2 + (x[0] + x[1] **
                                            2 - 7) ** 2 + (x[0] ** 2 + x[1] ** 3 - 1) ** 2

        return f


class EggCrate(Benchmark):
    """EggCrate class implements the Egg Crate's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = x_1^2 + x_2^2 + 25(sin^2(x_1) + sin^2(x_2))

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-5, 5], x_2 \in [-5, 5]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0)`.

    """

    def __init__(self, name='EggCrate', dims=2, continuous=True, convex=False,
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
        super(EggCrate, self).__init__(name, dims, continuous,
                                       convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Egg Crate's function
        f = x[0] ** 2 + x[1] ** 2 + 25 * \
            (np.sin(x[0]) ** 2 + np.sin(x[1]) ** 2)

        return f


class EggHolder(Benchmark):
    """EggHolder class implements the Egg Holder's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = -(x_2 + 47)sin(\sqrt{|x_2+\\frac{x_1}{2}+47|})-x_1 sin(\sqrt{|x_1-(x_2+47)|})

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-512, 512], x_2 \in [-512, 512]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -959.6406627106155 \mid \mathbf{x^*} = (512, 404.2319)`.

    """

    def __init__(self, name='EggHolder', dims=2, continuous=True, convex=False,
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
        super(EggHolder, self).__init__(name, dims, continuous,
                                        convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Egg Holder's function
        f = -(x[1] + 47) * np.sin(np.sqrt(np.fabs(x[1] + (x[0] / 2) + 47))
                                  ) - x[0] * np.sin(np.sqrt(np.fabs(x[0] - (x[1] + 47))))

        return f


class FreudensteinRoth(Benchmark):
    """FreudensteinRoth class implements the Freudenstein Roth's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = (x_1 - 13 + ((5 - x_2)x_2 - 2)x_2) ** 2 + (x_1 - 29 + ((x_2 + 1)x_2 - 14)x_2) ** 2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (5, 4)`.

    """

    def __init__(self, name='FreudensteinRoth', dims=2, continuous=True, convex=False,
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
        super(FreudensteinRoth, self).__init__(name, dims, continuous,
                                               convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the FreudensteinRoth's function
        f = (x[0] - 13 + ((5 - x[1]) * x[1] - 2) * x[1]) ** 2 + \
            (x[0] - 29 + ((x[1] + 1) * x[1] - 14) * x[1]) ** 2

        return f


class Giunta(Benchmark):
    """Giunta class implements the Giunta's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = 0.6 + \sum_{i=1}^{2}[sin(\\frac{16}{15}x_i-1) + sin^2(\\frac{16}{15}x_i-1) + \\frac{1}{50}sin(4(\\frac{16}{15}x_i-1))]

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-1, 1], x_2 \in [-1, 1]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0.06447042053690571 \mid \mathbf{x^*} = (0.4673200277395354, 0.4673200169591304)`.

    """

    def __init__(self, name='Giunta', dims=2, continuous=True, convex=False,
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
        super(Giunta, self).__init__(name, dims, continuous,
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
        f = 0.6

        # For every input dimension
        for i in range(x.shape[0]):
            # Calculating the Giunta's function
            f += np.sin(16 / 15 * x[i] - 1) + np.sin(16 / 15 * x[i] -
                                                     1) ** 2 + (1 / 50) * np.sin(4 * (16 / 15 * x[i] - 1))

        return f


class GoldsteinPrice(Benchmark):
    """GoldsteinPrice class implements the Goldstein Price's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = [1 + (x_1 + x_2 + 1)^2 (19 - 14x_1 + 3x_1^2 - 14x_2 + 6x_1x_2 + 3x_2^2)] [30 + (2x_1 - 3x_2)^2 (18 - 32x_1 + 12x_1^2 + 48x_2 - 36x_1x_2 + 27x_2^2)]

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-2, 2], x_2 \in [-2, 2]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 3 \mid \mathbf{x^*} = (0, -1)`.

    """

    def __init__(self, name='GoldsteinPrice', dims=2, continuous=True, convex=False,
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
        super(GoldsteinPrice, self).__init__(name, dims, continuous,
                                             convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Goldstein Price's function
        f = (1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)) * \
            (30 + (2 * x[0] - 3 * x[1]) ** 2 * (18 - 32 * x[0] + 12 *
                                                x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))

        return f


class Himmelblau(Benchmark):
    """Himmelblau class implements the Himmelblau's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-5, 5], x_2 \in [-5, 5]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (3, 2)`.

    """

    def __init__(self, name='Himmelblau', dims=2, continuous=True, convex=False,
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
        super(Himmelblau, self).__init__(name, dims, continuous,
                                         convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Himmelblau's function
        f = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

        return f


class HolderTable(Benchmark):
    """HolderTable class implements the HolderTable's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = -|sin(x_1)cos(x_2)e^{|1 - \\frac{\sqrt{x_1^2 + x_2^2}}{\\pi}|}|

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -19.208502567767606 \mid \mathbf{x^*} = (\pm 8.05502, \pm 9.66459)`.

    """

    def __init__(self, name='HolderTable', dims=2, continuous=True, convex=False,
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
        super(HolderTable, self).__init__(name, dims, continuous,
                                          convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the HolderTable's function
        f = -np.fabs(np.sin(x[0]) * np.cos(x[1]) *
                     np.exp(np.fabs(1 - (np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))))

        return f


class Hosaki(Benchmark):
    """Hosaki class implements the Hosaki's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = (1 - 8x_1 + 7x_1^2 - \\frac{7}{3x_1^3} + \\frac{1}{4x_1^4})x_2^2e^{-x_2}

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [0, 5], x_2 \in [0, 6]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -2.345811576101292 \mid \mathbf{x^*} = (4, 2)`.

    """

    def __init__(self, name='Hosaki', dims=2, continuous=True, convex=False,
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
        super(Hosaki, self).__init__(name, dims, continuous,
                                     convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Hosaki's function
        f = (1 - 8 * x[0] + 7 * x[0] ** 2 - 7 / 3 * x[0] **
             3 + 1 / 4 * x[0] ** 4) * x[1] ** 2 * np.exp(-x[1])

        return f


class JennrichSampson(Benchmark):
    """JennrichSampson class implements the Jennrich Sampson's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = \sum_{i=1}^{10}(2 + 2i - (e^{ix_1} + e^{ix_2}))^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-1, 1], x_2 \in [-1, 1]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 124.36218236181409 \mid \mathbf{x^*} = (0.257825, 0.257825)`.

    """

    def __init__(self, name='JennrichSampson', dims=2, continuous=True, convex=False,
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
        super(JennrichSampson, self).__init__(name, dims, continuous,
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

        # For `i` ranging from 1 to 10
        for i in range(1, 11):
            # Calculating the Jennrich Sampson's function
            f += (2 + 2 * i - (np.exp(i * x[0]) + np.exp(i * x[1]))) ** 2

        return f


class Keane(Benchmark):
    """Keane class implements the Keane's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = \\frac{sin^2(x_1-x_2)sin^2(x_1+x_2)}{sqrt{x_1^2+x_2^2}}

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [0, 10], x_2 \in [0, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0.6736675211468548 \mid \mathbf{x^*} = (1.393249070031784, 0) ~ or ~(0, 1.393249070031784)`.

    """

    def __init__(self, name='Keane', dims=2, continuous=True, convex=False,
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
        super(Keane, self).__init__(name, dims, continuous,
                                    convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Keane's function
        f = (np.sin(x[0] - x[1]) ** 2 * np.sin(x[0] + x[1])
             ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2)

        return f


class Leon(Benchmark):
    """Leon class implements the Leon's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = 100(x_2 - x_1^2)^2 + (1 - x_1)^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-1.2, 1.2], x_2 \in [-1.2, 1.2]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 1)`.

    """

    def __init__(self, name='Leon', dims=2, continuous=True, convex=False,
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
        super(Leon, self).__init__(name, dims, continuous,
                                   convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Leon's function
        f = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

        return f


class Levy13(Benchmark):
    """Levy13 class implements the Levy's 13th benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = sin^2(3\pi x_1)+(x_1-1)^2(1+sin^2(3\pi x_2))+(x_2-1)^2(1+sin^2(2\pi x_2))

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 1)`.

    """

    def __init__(self, name='Levy13', dims=2, continuous=True, convex=False,
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
        super(Levy13, self).__init__(name, dims, continuous,
                                     convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Levy's 13th function
        f = np.sin(3 * np.pi * x[0]) ** 2 + (x[0] - 1) ** 2 * (1 + np.sin(
            3 * np.pi * x[1]) ** 2) + (x[1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[1]) ** 2)

        return f


class Matyas(Benchmark):
    """Matyas class implements the Matyas' benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = 0.26(x_1^2 + x_2^2) - 0.48x_1x_2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0)`.

    """

    def __init__(self, name='Matyas', dims=2, continuous=True, convex=False,
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
        super(Matyas, self).__init__(name, dims, continuous,
                                     convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Matyas' function
        f = 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]

        return f


class McCormick(Benchmark):
    """McCormick class implements the McCormick's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = sin(x_1 + x_2) + (x_1 - x_2)^2 - \\frac{3}{2}x_1 + \\frac{5}{2}x_2 + 1

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-1.5, 4], x_2 \in [-3, 3]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -1.9132228873800594 \mid \mathbf{x^*} = (−0.547, −1.547)`.

    """

    def __init__(self, name='McCormick', dims=2, continuous=True, convex=False,
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
        super(McCormick, self).__init__(name, dims, continuous,
                                        convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the McCormick's function
        f = np.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - \
            3 / 2 * x[0] + 5 / 2 * x[1] + 1

        return f


class Mishra3(Benchmark):
    """Mishra3 class implements the Mishra's 3rd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = |cos(sqrt{|x_1^2+x_2|})|^{0.5} + 0.01(x_1+x_2)

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -0.18465133334298883 \mid \mathbf{x^*} = (−8.466613775046579, −9.998521308999999)`.

    """

    def __init__(self, name='Mishra3', dims=2, continuous=True, convex=False,
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
        super(Mishra3, self).__init__(name, dims, continuous,
                                      convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Mishra's 3rd function
        f = np.fabs(
            np.cos(np.sqrt(np.fabs(x[0] ** 2 + x[1])))) ** 0.5 + 0.01 * (x[0] + x[1])

        return f


class Mishra4(Benchmark):
    """Mishra4 class implements the Mishra's 4th benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = |sin(sqrt{|x_1^2+x_2|})|^{0.5} + 0.01(x_1+x_2)

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -0.1994069700888328 \mid \mathbf{x^*} = (−9.941127263635860, −9.999571661999983)`.

    """

    def __init__(self, name='Mishra4', dims=2, continuous=True, convex=False,
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
        super(Mishra4, self).__init__(name, dims, continuous,
                                      convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Mishra's 4th function
        f = np.fabs(
            np.sin(np.sqrt(np.fabs(x[0] ** 2 + x[1])))) ** 0.5 + 0.01 * (x[0] + x[1])

        return f


class Mishra5(Benchmark):
    """Mishra5 class implements the Mishra's 5th benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = [sin^2(cos(x_1)+cos(x_2))^2 + cos^2(sin(x_1)+sin(x_2)) + x_1]^2 + 0.01x_1 + 0.1x_2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -1.019829519930943 \mid \mathbf{x^*} = (−1.986820662153768, −10)`.

    """

    def __init__(self, name='Mishra5', dims=2, continuous=True, convex=False,
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
        super(Mishra5, self).__init__(name, dims, continuous,
                                      convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Mishra's 5th function
        f = (np.sin((np.cos(x[0]) + np.cos(x[1])) ** 2) ** 2 + np.cos(
            (np.sin(x[0]) + np.sin(x[1])) ** 2) ** 2 + x[0]) ** 2 + 0.01 * x[0] + 0.1 * x[1]

        return f


class Mishra6(Benchmark):
    """Mishra6 class implements the Mishra's 6th benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = -log[sin^2(cos(x_1)+cos(x_2))^2 - cos^2(sin(x_1)+sin(x_2)) + x_1]^2 + 0.1((x_1-1)^2 + (x_2-1)^2)

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -2.2839498384747587 \mid \mathbf{x^*} = (2.886307215440481, 1.823260331422321)`.

    """

    def __init__(self, name='Mishra6', dims=2, continuous=True, convex=False,
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
        super(Mishra6, self).__init__(name, dims, continuous,
                                      convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Mishra's 6th function
        f = -np.log((np.sin((np.cos(x[0]) + np.cos(x[1])) ** 2) ** 2 - np.cos((np.sin(
            x[0]) + np.sin(x[1])) ** 2) ** 2 + x[0]) ** 2) + 0.1 * ((x[0] - 1) ** 2 + (x[1] - 1) ** 2)

        return f


class Mishra8(Benchmark):
    """Mishra8 class implements the Mishra's 8th benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = 0.001(|g(x_1)| + |h(x_2)|)^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (2, -3)`.

    """

    def __init__(self, name='Mishra8', dims=2, continuous=True, convex=False,
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
        super(Mishra8, self).__init__(name, dims, continuous,
                                      convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating partial `g`
        g = x[0] ** 10 - 20 * x[0] ** 9 + 180 * x[0] ** 8 - 960 * x[0] ** 7 + 3360 * x[0] ** 6 - 8064 * \
            x[0] ** 5 + 13340 * x[0] ** 4 - 15360 * x[0] ** 3 + \
            11520 * x[0] ** 2 - 5120 * x[0] + 2624

        # Calculating partial `h`
        h = x[1] ** 4 + 12 * x[1] ** 3 + 54 * x[1] ** 2 + 108 * x[1] + 81

        # Calculating the Mishra's 8th function
        f = 0.001 * (np.fabs(g) + np.fabs(h)) ** 2

        return f


class Parsopoulos(Benchmark):
    """Parsopoulos class implements the Parsopoulos' benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = cos(x_1)^2 + sin(x_2)^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-5, 5], x_2 \in [-5, 5]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (k\\frac{\\pi}{2}, \lambda \\pi)`.

    """

    def __init__(self, name='Parsopoulos', dims=2, continuous=True, convex=False,
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
        super(Parsopoulos, self).__init__(name, dims, continuous,
                                          convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Parsopoulos' function
        f = np.cos(x[0]) ** 2 + np.sin(x[1]) ** 2

        return f


class PenHolder(Benchmark):
    """PenHolder class implements the Pen Holder's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = -e^{[cos(x_1)cos(x_2)e^{|1-[(x_1^2+x_2^2)]^{0.5} / \\pi|}|]^{-1}}

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-11, 11], x_2 \in [-11, 11]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -0.9635348327265058 \mid \mathbf{x^*} = (\pm 9.646167671043401, \pm 9.646167671043401)`.

    """

    def __init__(self, name='PenHolder', dims=2, continuous=True, convex=False,
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
        super(PenHolder, self).__init__(name, dims, continuous,
                                        convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Pen Holder's function
        f = -np.exp(-1 / np.fabs(np.cos(x[0]) * np.cos(x[1]) * np.exp(
            np.fabs(1 - (x[0] ** 2 + x[1] ** 2) ** 0.5 / np.pi))))

        return f


class Periodic(Benchmark):
    """Periodic class implements the Periodic's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = 1 + sin^2(x_1) + sin^2(x_2) - 0.1e^{-(x_1^2+x_2^2)}

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0.9 \mid \mathbf{x^*} = (0, 0)`.

    """

    def __init__(self, name='Periodic', dims=2, continuous=True, convex=False,
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
        super(Periodic, self).__init__(name, dims, continuous,
                                       convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Periodic's function
        f = 1 + np.sin(x[0]) ** 2 + np.sin(x[1]) ** 2 - \
            0.1 * np.exp(-(x[0] ** 2 + x[1] ** 2))

        return f


class Price1(Benchmark):
    """Price1 class implements the Price's 1st benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = (|x_1| - 5)^2 + (|x_2| - 5)^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (-5, -5) ~ or ~(-5, 5) ~ or ~(5, -5) ~ or ~(5, 5)`.

    """

    def __init__(self, name='Price1', dims=2, continuous=True, convex=False,
                 differentiable=False, multimodal=True, separable=True):
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
        super(Price1, self).__init__(name, dims, continuous,
                                     convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Price's 1st function
        f = (np.fabs(x[0]) - 5) ** 2 + (np.fabs(x[1]) - 5) ** 2

        return f


class Price2(Benchmark):
    """Price2 class implements the Price's 2nd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = 1 + sin^2(x_1) + sin^2(x_2) - 0.1e^{-x_1^2-x_2^2}

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0.9 \mid \mathbf{x^*} = (0, 0)`.

    """

    def __init__(self, name='Price2', dims=2, continuous=True, convex=False,
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
        super(Price2, self).__init__(name, dims, continuous,
                                     convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Price's 2nd function
        f = 1 + np.sin(x[0]) ** 2 + np.sin(x[1]) ** 2 - \
            0.1 * np.exp(-x[0] ** 2 - x[1] ** 2)

        return f


class Price3(Benchmark):
    """Price3 class implements the Price's 3rd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = 100(x_2 - x_1^2)^2 + [6.4(x_2 - 0.5)^2 - x_1 - 0.6]^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-5, 5], x_2 \in [-5, 5]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0.341307503353524, 0.116490811845416) ~ or ~(1, 1)`.

    """

    def __init__(self, name='Price3', dims=2, continuous=True, convex=False,
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
        super(Price3, self).__init__(name, dims, continuous,
                                     convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Price's 3rd function
        f = 100 * (x[1] - x[0] ** 2) ** 2 + \
            (6.4 * (x[1] - 0.5) ** 2 - x[0] - 0.6) ** 2

        return f


class Price4(Benchmark):
    """Price4 class implements the Price's 4th benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = (2x_1^3x_2 - x_2^3)^2 + (6x_1 - x_2^2 + x_2)^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0) ~ or ~(2, 4) ~ or ~(1.464, -2.506)`.

    """

    def __init__(self, name='Price4', dims=2, continuous=True, convex=False,
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
        super(Price4, self).__init__(name, dims, continuous,
                                     convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Price's 4th function
        f = (2 * x[0] ** 3 * x[1] - x[1] ** 3) ** 2 + \
            (6 * x[0] - x[1] ** 2 + x[1]) ** 2

        return f


class Quadratic(Benchmark):
    """Quadratic class implements the Quadratic's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = -3803.84 - 138.08x_1 - 232.92x_2 + 128.08x_1^2 + 203.64x_2^2 + 182.25x_1x_2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -3873.7243 \mid \mathbf{x^*} = (0.19388, 0.48513)`.

    """

    def __init__(self, name='Quadratic', dims=2, continuous=True, convex=False,
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
        super(Quadratic, self).__init__(name, dims, continuous,
                                        convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Quadratic's function
        f = -3803.84 - 138.08 * x[0] - 232.92 * x[1] + 128.08 * \
            x[0] ** 2 + 203.64 * x[1] ** 2 + 182.25 * x[0] * x[1]

        return f


class RotatedEllipse1(Benchmark):
    """RotatedEllipse1 class implements the Rotated Ellipse's 1st benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = 7x_1^2 - 6\sqrt{3}x_1x_2 + 13x_2^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0)`.

    """

    def __init__(self, name='RotatedEllipse1', dims=2, continuous=True, convex=False,
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
        super(RotatedEllipse1, self).__init__(name, dims, continuous,
                                              convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Rotated Ellipse's 1st function
        f = 7 * x[0] ** 2 - 6 * np.sqrt(3) * x[0] * x[1] + 13 * x[1] ** 2

        return f


class RotatedEllipse2(Benchmark):
    """RotatedEllipse2 class implements the Rotated Ellipse's 2nd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = x_1^2 - x_1x_2 + x_2^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0)`.

    """

    def __init__(self, name='RotatedEllipse2', dims=2, continuous=True, convex=False,
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
        super(RotatedEllipse2, self).__init__(name, dims, continuous,
                                              convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Rotated Ellipse's 2nd function
        f = x[0] ** 2 - x[0] * x[1] + x[1] ** 2

        return f


class Rump(Benchmark):
    """Rump class implements the Rump's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = (333.75 - x_1^2)x_2^6 + x_1^2(11x_1^2x_2^2 - 121x_2^4 - 2) + 5.5x_2^8 + \\frac{x_1}{2x_2}

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0)`.

    """

    def __init__(self, name='Rump', dims=2, continuous=True, convex=False,
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
        super(Rump, self).__init__(name, dims, continuous,
                                   convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Rump's function
        f = (333.75 - x[0] ** 2) * x[1] ** 6 + x[0] ** 2 * (11 * x[0] ** 2 * x[1] **
                                                            2 - 121 * x[1] ** 4 - 2) + 5.5 * x[1] ** 8 + x[0] / (2 * x[1] + c.EPSILON)

        return f


class Schaffer1(Benchmark):
    """Schaffer1 class implements the Schaffer's 1st benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = 0.5 + \\frac{sin^2(x_1^2 + x_2^2)^2 - 0.5}{1 + 0.001(x_1^2 + x_2^2)^2}

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-100, 100], x_2 \in [-100, 100]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0)`.

    """

    def __init__(self, name='Schaffer1', dims=2, continuous=True, convex=False,
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
        super(Schaffer1, self).__init__(name, dims, continuous,
                                        convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Schaffer's 1st function
        f = 0.5 + (np.sin((x[0] ** 2 + x[1] ** 2) ** 2) **
                   2 - 0.5) / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2) ** 2)

        return f


class Schaffer2(Benchmark):
    """Schaffer2 class implements the Schaffer's 2nd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = 0.5 + \\frac{sin^2(x_1^2 - x_2^2)^2 - 0.5}{1 + 0.001(x_1^2 + x_2^2)^2}

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-100, 100], x_2 \in [-100, 100]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0)`.

    """

    def __init__(self, name='Schaffer2', dims=2, continuous=True, convex=False,
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
        super(Schaffer2, self).__init__(name, dims, continuous,
                                        convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Schaffer's 1st function
        f = 0.5 + (np.sin((x[0] ** 2 - x[1] ** 2) ** 2) **
                   2 - 0.5) / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2) ** 2)

        return f


class Schaffer3(Benchmark):
    """Schaffer3 class implements the Schaffer's 3rd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = 0.5 + \\frac{sin^2(cos|x_1^2 + x_2^2|) - 0.5}{(1 + 0.001(x_1^2 + x_2^2))^2}

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-100, 100], x_2 \in [-100, 100]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0.0015668545260126288 \mid \mathbf{x^*} = (0, 1.253115)`.

    """

    def __init__(self, name='Schaffer3', dims=2, continuous=True, convex=False,
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
        super(Schaffer3, self).__init__(name, dims, continuous,
                                        convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Schaffer's 3rd function
        f = 0.5 + (np.sin(np.cos(np.fabs(x[0] ** 2 + x[1] ** 2)))
                   ** 2 - 0.5) / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2

        return f


class Schaffer4(Benchmark):
    """Schaffer4 class implements the Schaffer's 4th benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = 0.5 + \\frac{cos^2(sin(x_1^2 - x_2^2)) - 0.5}{1 + 0.001(x_1^2 + x_2^2)^2}

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-100, 100], x_2 \in [-100, 100]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0.29243850703298857 \mid \mathbf{x^*} = (0, 1.253115)`.

    """

    def __init__(self, name='Schaffer4', dims=2, continuous=True, convex=False,
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
        super(Schaffer4, self).__init__(name, dims, continuous,
                                        convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Schaffer's 4th function
        f = 0.5 + (np.cos(np.sin(x[0] ** 2 - x[1] ** 2)) **
                   2 - 0.5) / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2) ** 2)

        return f


class Schwefel26(Benchmark):
    """Schwefel26 class implements the Schwefel's 2.6 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = \max(|x_1 + 2x_2 - 7|, |2x_1 + x_2 - 5|)

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-100, 100], x_2 \in [-100, 100]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 3)`.

    """

    def __init__(self, name='Schwefel26', dims=2, continuous=True, convex=False,
                 differentiable=False, multimodal=False, separable=False):
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
        super(Schwefel26, self).__init__(name, dims, continuous,
                                         convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Schwefel's 2.6 function
        f = max(np.fabs(x[0] + 2 * x[1] - 7), np.fabs(2 * x[0] + x[1] - 5))

        return f


class Schwefel236(Benchmark):
    """Schwefel236 class implements the Schwefel's 2.36 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = -x_1x_2(72 - 2x_1 - 2x_2)

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [0, 500], x_2 \in [0, 500]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -3456 \mid \mathbf{x^*} = (12, 12)`.

    """

    def __init__(self, name='Schwefel236', dims=2, continuous=True, convex=False,
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
        super(Schwefel236, self).__init__(name, dims, continuous,
                                          convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Schwefel's 2.36 function
        f = -x[0] * x[1] * (72 - 2 * x[0] - 2 * x[1])

        return f


class Table1(Benchmark):
    """Table1 class implements the Table's 1st benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = -|cos(x_1)cos(x_2)e^{|1-(x_1+x_2)^{0.5}/\\pi|}

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -26.920335555515848 \mid \mathbf{x^*} = (\pm 9.646168, \pm 9.646168)`.

    """

    def __init__(self, name='Table1', dims=2, continuous=True, convex=False,
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
        super(Table1, self).__init__(name, dims, continuous,
                                     convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Table's 1st function
        f = -np.fabs(np.cos(x[0]) * np.cos(x[1]) *
                     np.exp(np.fabs(1 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi)))

        return f


class Table2(Benchmark):
    """Table2 class implements the Table's 2nd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = -|cos(x_1)cos(x_2)e^{|1-(x_1+x_2)^{0.5}/\\pi|}

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -19.20850256788675 \mid \mathbf{x^*} = (\pm 8.055023472141116, \pm 9.664590028909654)`.

    """

    def __init__(self, name='Table2', dims=2, continuous=True, convex=False,
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
        super(Table2, self).__init__(name, dims, continuous,
                                     convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Table's 2nd function
        f = -np.fabs(np.sin(x[0]) * np.cos(x[1]) *
                     np.exp(np.fabs(1 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi)))

        return f


class Table3(Benchmark):
    """Table3 class implements the Table's 3rd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = -\\frac{1}{30}e^{2|1 - \\frac{sqrt{x_1^2+x_2^2}{\\pi}}}cos^2(x_1)cos^2(x_2)

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -19.20850256788675 \mid \mathbf{x^*} = (\pm 9.646157266348881, \pm 9.646134286497169)`.

    """

    def __init__(self, name='Table3', dims=2, continuous=True, convex=False,
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
        super(Table3, self).__init__(name, dims, continuous,
                                     convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Table's 3rd function
        f = -np.cos(x[0]) ** 2 * np.cos(x[1]) ** 2 * np.exp(2 *
                                                            np.fabs(1 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi)) / 30

        return f


class TesttubeHolder(Benchmark):
    """TesttubeHolder class implements the Testtube Holder's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = -4[(sin(x_1)cos(x_2)e^{|cos[(x_1^2+x_2^2)/200]|})]

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -10.872299901558 \mid \mathbf{x^*} = (\pm \\frac{\\pi}{2}, 0)`.

    """

    def __init__(self, name='TesttubeHolder', dims=2, continuous=True, convex=False,
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
        super(TesttubeHolder, self).__init__(name, dims, continuous,
                                             convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Testtube Holder's function
        f = -4 * (np.sin(x[0]) * np.cos(x[1]) *
                  np.exp(np.fabs(np.cos((x[0] ** 2 + x[1] ** 2) / 200))))

        return f


class Trecani(Benchmark):
    """Trecani class implements the Trecani's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = x_1^4 - 4x_1^3 + 4x_1 + x_2^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-5, 5], x_2 \in [-5, 5]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0) ~ or ~(-2, 0)`.

    """

    def __init__(self, name='Trecani', dims=2, continuous=True, convex=False,
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
        super(Trecani, self).__init__(name, dims, continuous,
                                      convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Trecani's function
        f = x[0] ** 4 - 4 * x[0] ** 3 + 4 * x[0] + x[1] ** 2

        return f


class Trefethen(Benchmark):
    """Trefethen class implements the Trefethen's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = e^{sin(50x_1)} + sin(60e^{x^2}) + sin(70sin(x_1)) + sin(sin(80x_2)) - sin(10(x_1 + x_2)) + \\frac{1}{4}(x_1^2 + x_2^2)

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -3.3068686465567008 \mid \mathbf{x^*} = (−0.024403, 0.210612)`.

    """

    def __init__(self, name='Trefethen', dims=2, continuous=True, convex=False,
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
        super(Trefethen, self).__init__(name, dims, continuous,
                                        convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Trefethen's function
        f = np.exp(np.sin(50 * x[0])) + np.sin(60 * np.exp(x[1])) + np.sin(70 * np.sin(x[0])) + np.sin(
            np.sin(80 * x[1])) - np.sin(10 * (x[0] + x[1])) + (1 / 4) * (x[0] ** 2 + x[1] ** 2)

        return f


class VenterSobiezcczanskiSobieski(Benchmark):
    """VenterSobiezcczanskiSobieski class implements the Venter Sobiezcczanski-Sobieski's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = x_1^2 - 100cos(x_1)^2 - 100cos(\\frac{x_1^2}{30}) + x_2^2 - 100cos(x_2)^2 - 100cos(\\frac{x_2^2}{30})

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-50, 50], x_2 \in [-50, 50]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -400 \mid \mathbf{x^*} = (0, 0)`.

    """

    def __init__(self, name='VenterSobiezcczanskiSobieski', dims=2, continuous=True, convex=False,
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
        super(VenterSobiezcczanskiSobieski, self).__init__(name, dims, continuous,
                                                           convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Venter Sobiezcczanski-Sobieski's function
        f = x[0] ** 2 - 100 * np.cos(x[0]) ** 2 - 100 * np.cos(x[0] ** 2 / 30) + \
            x[1] ** 2 - 100 * np.cos(x[1]) ** 2 - 100 * np.cos(x[1] ** 2 / 30)

        return f


class WayburnSeader1(Benchmark):
    """WayburnSeader1 class implements the WayburnSeader's 1st benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = (x_1^6 + x_2^4 - 17)^2 + (2x_1 + x_2 - 4)^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 2) ~ or ~(1.596804153876933, 0.806391692246134)`.

    """

    def __init__(self, name='WayburnSeader1', dims=2, continuous=True, convex=False,
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
        super(WayburnSeader1, self).__init__(name, dims, continuous,
                                             convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the WayburnSeader's 1st function
        f = (x[0] ** 6 + x[1] ** 4 - 17) ** 2 + (2 * x[0] + x[1] - 4) ** 2

        return f


class WayburnSeader2(Benchmark):
    """WayburnSeader2 class implements the WayburnSeader's 2nd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = [1.613 - 4(x_1 - 0.3125)^2 - 4(x_2 - 1.625)^2]^2 + (x_2 - 1)^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0.200138974728779, 1) ~ or ~(0.424861025271221, 1)`.

    """

    def __init__(self, name='WayburnSeader2', dims=2, continuous=True, convex=False,
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
        super(WayburnSeader2, self).__init__(name, dims, continuous,
                                             convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the WayburnSeader's 2nd function
        f = (1.613 - 4 * (x[0] - 0.3125) ** 2 - 4 *
             (x[1] - 1.625) ** 2) ** 2 + (x[1] - 1) ** 2

        return f


class WayburnSeader3(Benchmark):
    """WayburnSeader3 class implements the WayburnSeader's 3rd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = \\frac{2}{3}x_1^3 - 8x_1^2 + 33x_1 - x_1x_2 + 5 + [(x_1 - 4)^2 + (x_2 - 5)^2 - 4]^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 19.105879794568 \mid \mathbf{x^*} = (5.146896745324582, 6.839589743000071)`.

    """

    def __init__(self, name='WayburnSeader3', dims=2, continuous=True, convex=False,
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
        super(WayburnSeader3, self).__init__(name, dims, continuous,
                                             convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the WayburnSeader's 3rd function
        f = (2 / 3) * x[0] ** 3 - 8 * x[0] ** 2 + 33 * x[0] - x[0] * \
            x[1] + 5 + ((x[0] - 4) ** 2 + (x[1] - 5) ** 2 - 4) ** 2

        return f


class Zettl(Benchmark):
    """Zettl class implements the Zettl's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = (x_1^2 + x_2^2 - 2x_1)^2 + 0.25x_1

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-5, 10], x_2 \in [-5, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -0.0037912371501199 \mid \mathbf{x^*} = (−0.0299, 0)`.

    """

    def __init__(self, name='Zettl', dims=2, continuous=True, convex=False,
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
        super(Zettl, self).__init__(name, dims, continuous,
                                    convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Zettl's function
        f = (x[0] ** 2 + x[1] ** 2 - 2 * x[0]) ** 2 + 0.25 * x[0]

        return f


class Zirilli(Benchmark):
    """Zirilli class implements the Zirilli's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = 0.25x_1^4 - 0.5x_1^2 + 0.1x_1 + 0.5x_2^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -0.3523860365437344 \mid \mathbf{x^*} = (-1.0465, 0)`.

    """

    def __init__(self, name='Zirilli', dims=2, continuous=True, convex=False,
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
        super(Zirilli, self).__init__(name, dims, continuous,
                                      convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Zirilli's function
        f = 0.25 * x[0] ** 4 - 0.5 * x[0] ** 2 + 0.1 * x[0] + 0.5 * x[1] ** 2

        return f
