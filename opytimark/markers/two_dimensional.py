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
        :math:`f(\mathbf{x^*}) \\approx −195.629028238419 \mid \mathbf{x^*} \\approx (\pm 0.682584587365898, -0.36075325513719)`.

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
        :math:`f(\mathbf{x^*}) = 0.39788735775266204 \mid \mathbf{x^*} = (-\\pi, 12.275) or (\\pi, 2.275) or (3\\pi, 2.425)`.

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
        f = (x[1] - ((5.1 * x[0] ** 2) / (4 * np.pi ** 2)) + ((5 * x[0]) / np.pi) - 6) ** 2 + 10 * (1 - (1 / (8 * np.pi))) * np.cos(x[0]) + 10

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
        f = (x[0] + 10) ** 2 + (x[1] + 10) ** 2 + np.exp(-(x[0] ** 2) - (x[1] ** 2))

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
        f = 100 * np.sqrt(np.fabs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * np.fabs(x[0] + 10)

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
        f = 2 * x[0] ** 2 - 1.05 * x[0] ** 4 + x[0] ** 6 / 6 + x[0] * x[1] + x[1] ** 2

        return f


class Camel6(Benchmark):
    """Camel6 class implements the Camel's Six Hump benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2) = (4 - 2.1x_1^2 + \\frac{x_1^4}{3})x_1^2 + x_1x_2 + (4x_2^2 - 4)x_2^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [-5, 5], x_2 \in [-5, 5]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -1.0316284229280819 \mid \mathbf{x^*} = (−0.0898, 0.7126) or (0.0898,−0.7126)`.

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
        f = (4 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3) * x[0] ** 2 + x[0] * x[1] + (4 * x[1] ** 2 - 4) * x[1] ** 2

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
        f = -((0.001) / (0.001 ** 2 + (x[0] ** 2 + x[1] ** 2 - 1) ** 2)) - ((0.001) / (0.001 ** 2 + (x[0] ** 2 + x[1] ** 2 - 0.5) ** 2)) - ((0.001) / (0.001 ** 2 + (x[0] ** 2 - x[1] ** 2) ** 2))

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
        f = ((0.001) / (0.001 ** 2 + (x[0] - 0.4 * x[1] - 0.1) ** 2)) + ((0.001) / (0.001 ** 2 + (2 * x[0] + x[1] - 1.5) ** 2)) 

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
        f = x[0] ** 2 - 12 * x[0] + 11 + 10 * np.cos((np.pi * x[0]) / 2) + 8 * np.sin((5 * np.pi * x[0]) / 2) - (1 / 5) ** 0.5 * np.exp(-0.5 * (x[1] - 0.5) ** 2)

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
        f = -0.0001 * (np.fabs(np.sin(x[0]) * np.sin(x[1]) * np.exp(np.fabs(100 - (np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi)))) + 1) ** 0.1

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
