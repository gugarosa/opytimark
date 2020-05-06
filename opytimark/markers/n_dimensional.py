import numpy as np

import opytimark.utils.decorator as d
from opytimark.core import Benchmark


class Ackley1(Benchmark):
    """Ackley1 class implements the Ackley's 1st benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = -20e^{-0.2\sqrt{\\frac{1}{n}\sum_{i=1}^{n}x_i^2}}-e^{\\frac{1}{n}\sum_{i=1}^{n}cos(2 \\pi x_i)}+ 20 + e

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-32, -32] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.
        
    """

    def __init__(self, name='Ackley1', dims=-1, continuous=True, convex=False,
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
        super(Ackley1, self).__init__(name, dims, continuous,
                                      convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the 1 / n term
        inv = 1 / x.shape[0]

        # Calculating first term
        term1 = -0.2 * np.sqrt(inv * np.sum(x ** 2))

        # Calculating second term
        term2 = inv * np.sum(np.cos(2 * np.pi * x))

        # Calculating Ackley's 1st function
        f = - 20 * np.exp(term1) - np.exp(term2) + 20 + np.e

        return np.sum(f)

        
class Alpine1(Benchmark):
    """Alpine1 class implements the Alpine's 1st benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}|x_i sin(x_i)+0.1x_i|

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='Alpine1', dims=-1, continuous=True, convex=False,
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
        super(Alpine1, self).__init__(name, dims, continuous,
                                      convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Alpine's 1st function
        f = np.fabs(x * np.sin(x) + 0.1 * x)

        return np.sum(f)


class Alpine2(Benchmark):
    """Alpine2 class implements the Alpine's 2nd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n}\sqrt{x_i}sin(x_i)

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 10] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 2.808^n \mid \mathbf{x^*} = (7.917, 7.917, \ldots, 7.917)`.

    """

    def __init__(self, name='Alpine2', dims=-1, continuous=True, convex=False,
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
        super(Alpine2, self).__init__(name, dims, continuous,
                                      convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Alpine's 2nd function
        f = np.sqrt(x) * np.sin(x)

        return -np.prod(f)


class Brown(Benchmark):
    """Brown class implements the Brown's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-1}(x_i^2)^{(x_{i+1}^{2}+1)}+(x_{i+1}^2)^{(x_{i}^{2}+1)}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-1, 4] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='Brown', dims=-1, continuous=True, convex=True,
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
        super(Brown, self).__init__(name, dims, continuous,
                                    convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating first term squares
        term1 = x[:-1] ** 2

        # Calculating second term squares
        term2 = x[1:] ** 2

        # Calculating Brown's function
        f = np.sum(term1 ** (term2 + 1) + term2 ** (term1 + 1))

        return f


class ChungReynolds(Benchmark):
    """ChungReynolds class implements the Chung Reynolds' benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = (\sum_{i=1}^{n} x_i^2)^2

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='ChungReynolds', dims=-1, continuous=True, convex=True,
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
        super(ChungReynolds, self).__init__(name, dims, continuous,
                                            convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Chung Reynolds' function
        f = np.sum(x ** 2) ** 2

        return f


class CosineMixture(Benchmark):
    """CosineMixture class implements the Cosine Mixture's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = -0.1\sum_{i=1}^{n}cos(5 \\pi x_i) - \sum_{i=1}^{n}x_i^2

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-1, 1] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0.1n \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='CosineMixture', dims=-1, continuous=False, convex=False,
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
        super(CosineMixture, self).__init__(name, dims, continuous,
                                            convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating first term
        term1 = np.sum(np.cos(5 * np.pi * x))

        # Calculating second term
        term2 = np.sum(x ** 2)

        # Calculating Cosine's Mixture function
        f = -0.1 * term1 - term2

        return f


class Csendes(Benchmark):
    """Csendes class implements the Csendes' benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}x_i^6(2 + sin(\\frac{1}{x_i}))

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-1, 1] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='Csendes', dims=-1, continuous=True, convex=True,
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
        super(Csendes, self).__init__(name, dims, continuous,
                                      convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Csendes' function
        f = (x ** 6) * (2 + np.sin(1 / x))

        return np.sum(f)


class Deb1(Benchmark):
    """Deb1 class implements the Deb's 1st benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = -\\frac{1}{n}\sum_{i=1}^{n}sin^6(5 \\pi x_i)

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-1, 1] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -1 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='Deb1', dims=-1, continuous=True, convex=True,
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
        super(Deb1, self).__init__(name, dims, continuous,
                                   convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating partial term
        term = np.sum(np.sin(5 * np.pi * x) ** 6)

        # Declaring Deb's 1st function
        f = -1 / x.shape[0] * term

        return f


class Deb3(Benchmark):
    """Deb3 class implements the Deb's 3rd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = -\\frac{1}{n}\sum_{i=1}^{n}sin^6(5 \\pi (x_i^{\\frac{3}{4}}-0.05))

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-1, 1] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -1 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='Deb3', dims=-1, continuous=True, convex=True,
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
        super(Deb3, self).__init__(name, dims, continuous,
                                   convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating partial term
        term = np.sum(np.sin(5 * np.pi * (x ** (3/4) - 0.05)) ** 6)

        # Declaring Deb's 3rd function
        f = -1 / x.shape[0] * term

        return f


class Exponential(Benchmark):
    """Exponential class implements the Exponential's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = -e^{-0.5\sum_{i=1}^n{x_i^2}}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-1, 1] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='Exponential', dims=-1, continuous=True, convex=True,
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
        super(Exponential, self).__init__(name, dims, continuous,
                                          convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Exponential's function
        f = -np.exp(-0.5 * np.sum(x ** 2))

        return f


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
            The benchmarking function output `f(x)`.

        """

        # Calculating the sphere's function
        f = x ** 2

        return np.sum(f)
