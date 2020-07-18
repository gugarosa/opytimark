"""N-dimensional benchmarking functions.
"""

import numpy as np

import opytimark.utils.constants as c
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
        f = 20 + np.e - np.exp(term2) - 20 * np.exp(term1)

        return np.sum(f)


class Ackley4(Benchmark):
    """Ackley4 class implements the Ackley's 4th benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-1}(e^{-0.2}\sqrt{x_i^2+x_{i+1}^2}+3(cos(2x_i)+sin(2x_{i+1})))

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-35, -35] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = −4.590101633799122 \mid \mathbf{x^*} = (-1.51, -0.755)`.

    """

    def __init__(self, name='Ackley4', dims=-1, continuous=True, convex=False,
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
        super(Ackley4, self).__init__(name, dims, continuous,
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

        # For every input dimension
        for i in range(x.shape[0] - 1):
            # Calculating Ackley's 4th function
            f += np.exp(-0.2) * np.sqrt(x[i] ** 2 + x[i + 1] ** 2) + 3 * (np.cos(2 * x[i]) + np.sin(2 * x[i + 1]))

        return f


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
        :math:`f(\mathbf{x^*}) = -0.1n \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

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
        f = (x ** 6) * (2 + np.sin(1 / (x + c.EPSILON)))

        return np.sum(f)


class Deb1(Benchmark):
    """Deb1 class implements the Deb's 1st benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = -\\frac{1}{n}\sum_{i=1}^{n}sin^6(5 \\pi x_i)

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-1, 1] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -1 \mid \mathbf{x^*} = (-0.9, -0.7, \ldots, 0.9)`.

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
        The function is commonly evaluated using :math:`x_i \in [0, 1] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = ? \mid \mathbf{x^*} = (?, ?, \ldots, ?)`.

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


class DixonPrice(Benchmark):
    """DixonPrice class implements the Dixon & Price's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = (x_1 - 1)^2 + \sum_{i=2}^{n}i(2x_i^2 - x_{i-1})^2

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid x_i^* = 2^{-\\frac{2^i-2}{2^i}}`.

    """

    def __init__(self, name='DixonPrice', dims=-1, continuous=True, convex=True,
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
        super(DixonPrice, self).__init__(name, dims, continuous,
                                         convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating first partial term
        term1 = (x[0] - 1) ** 2

        # Initializing second partial term
        term2 = 0

        # For every possible dimension of `x`
        for i in range(1, x.shape[0]):
            # Calculating second partial term
            term2 += (i + 1) * ((2 * (x[i] ** 2) - x[i - 1]) ** 2)

        # Calculating the Dixon & Price's function
        f = term1 + term2

        return f


class Exponential(Benchmark):
    """Exponential class implements the Exponential's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = e^{-0.5\sum_{i=1}^n{x_i^2}}

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
        f = np.exp(-0.5 * np.sum(x ** 2))

        return f


class Griewank(Benchmark):
    """Griewank class implements the Griewank's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = 1 + \sum_{i=1}^{n}\\frac{x_i^2}{4000} - \prod cos(\\frac{x_i}{\sqrt{i}})

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='Griewank', dims=-1, continuous=True, convex=False,
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
        super(Griewank, self).__init__(name, dims, continuous,
                                       convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Initializing terms
        term1, term2 = 0, 1

        # For every possible dimension of `x`
        for i in range(x.shape[0]):
            # Calculating first term
            term1 += (x[i] ** 2) / 4000

            # Calculating second term
            term2 *= np.cos(x[i] / np.sqrt(i + 1))

        # Calculating the Griewank's function
        f = 1 + term1 - term2

        return f


class HappyCat(Benchmark):
    """HappyCat class implements the HappyCat's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = [(||\mathbf{x}||_2 - n)^2]^{\\alpha} + \\frac{1}{n}(\\frac{1}{2}||\mathbf{x}||_2 + \sum_{i=1}^{n}x_i) + \\frac{1}{2}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-2, 2] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (-1, -1, \ldots, -1)`.

    """

    def __init__(self, name='HappyCat', dims=-1, continuous=True, convex=False,
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
        super(HappyCat, self).__init__(name, dims, continuous,
                                       convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Gathering the input's dimension
        n = x.shape[0]

        # Calculating norm of `x`
        square = np.sum(x ** 2)

        # Calculating the HappyCat's function
        f = ((square - n) ** 2) ** (1 / 8) + (1 / n) * (1 / 2 * square + np.sum(x)) + 1 / 2

        return f


class Levy(Benchmark):
    """Levy class implements the Levy's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = sin^2(\\pi w_1) + \sum_{i=1}^{n-1}(w_i-1)^2 [1+10sin^2(\\pi w_i + 1)]
    .. math:: + (w_n - 1)^2 [1 + sin^2(2 \\pi w_n)] \mid w_i = 1 + \\frac{x_i - 1}{4}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 1, \ldots, 1)`.

    """

    def __init__(self, name='Levy', dims=-1, continuous=True, convex=True,
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
        super(Levy, self).__init__(name, dims, continuous,
                                   convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating `w` term
        w = 1 + (x - 1) / 4

        # Defining first term
        term1 = np.sin(np.pi * w[0]) ** 2

        # Defining third term
        term3 = ((w[-1] - 1) ** 2) * (1 + (np.sin(2 * np.pi * w[-1]) ** 2))

        # Reshaping `w`
        w = w[0:x.shape[0]-1]

        # Calculating second term
        term2 = np.sum(((w - 1) ** 2) * (1 + 10 * (np.sin(np.pi * w + 1) ** 2)))

        # Calculating the Levy's function
        f = term1 + term2 + term3

        return f


class Michalewicz(Benchmark):
    """Michalewicz class implements the Michalewicz's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = - \sum_{i=1}^{n}sin(x_i)sin^{20}(\\frac{ix_i^2}{\\pi})

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, \\pi] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = ? \mid \mathbf{x^*} = (?, ?, \ldots, ?)`.

    """

    def __init__(self, name='Michalewicz', dims=-1, continuous=True, convex=True,
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
        super(Michalewicz, self).__init__(name, dims, continuous,
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

        # For every input dimension
        for i in range(x.shape[0]):
            # Calculating the Michalewicz's function
            f += np.sin(x[i]) * (np.sin((i + 1) * x[i] ** 2 / np.pi) ** 20)

        return -f


class Pathological(Benchmark):
    """Pathological class implements the Pathological's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-1}0.5 + \\frac{sin^2(\sqrt{100x_i^2+x_{i+1}^2})-0.5}{1 + 0.001(x_i^2 - 2x_i x_{i+1} + x_{i+1}^2)^2}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='Pathological', dims=-1, continuous=True, convex=True,
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
        super(Pathological, self).__init__(name, dims, continuous,
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

        # For every input dimension
        for i in range(x.shape[0] - 1):
            # Calculating the Pathological's function
            f += 0.5 + (np.sin(np.sqrt(100 * x[i] ** 2 + x[i + 1] ** 2)) ** 2 - 0.5) / (
                1 + 0.001 * ((x[i] ** 2 - 2 * x[i] * x[i + 1] + x[i + 1] ** 2) ** 2))

        return f


class Periodic(Benchmark):
    """Periodic class implements the Periodic's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = 1 + \sum_{i=1}^{n}sin^2(x_i) - 0.1e^{\sum_{i=1}^{n}x_i^2}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0.9 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='Periodic', dims=-1, continuous=True, convex=False,
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
        f = 1 + np.sum(np.sin(x) ** 2) - 0.1 * np.exp(np.sum(x))

        return f


class Perm0DBeta(Benchmark):
    """Perm0DBeta class implements the Perm 0, D, Beta's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}(\sum_{j=1}^{n} (j + 10)(x_j^i - \\frac{1}{j^i}))^2

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-n, n] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, \\frac{1}{2}, \ldots, \\frac{1}{n})`.

    """

    def __init__(self, name='Perm0DBeta', dims=-1, continuous=True, convex=True,
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
        super(Perm0DBeta, self).__init__(name, dims, continuous,
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

        # For every input dimension
        for i in range(x.shape[0]):
            # For every input dimension
            for j in range(x.shape[0]):
                # Calculating the Perm 0, D, Beta's function
                f += ((j + 1 + 10) * (x[j] ** (i + 1) - (1 / (j + 1) ** (i + 1)))) ** 2

        return f

class PermDBeta(Benchmark):
    """PermDBeta class implements the Perm D, Beta's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}(\sum_{j=1}^{n} (j^i + 10)((\\frac{x_j}{j})^i - 1))^2

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-n, n] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 2, \ldots, n)`.

    """

    def __init__(self, name='PermDBeta', dims=-1, continuous=True, convex=True,
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
        super(PermDBeta, self).__init__(name, dims, continuous,
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

        # For every input dimension
        for i in range(x.shape[0]):
            # For every input dimension
            for j in range(x.shape[0]):
                # Calculating the Perm D, Beta's function
                f += (((j + 1) ** (i + 1) + 10) * ((x[j] / (j + 1)) ** (i + 1) - 1)) ** 2

        return f


class PowellSingular2(Benchmark):
    """PowellSingular2 class implements the Powell's Singular 2nd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-2}(x_{i-1}+10x_i)^2 + 5(x_{i+1} - x_{i+2})^2 + (x_i - 2x_{i+1})^4 + 10(x_{i-1} - x_{i+2})^4

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-4, 5] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='PowellSingular2', dims=-1, continuous=True, convex=True,
                 differentiable=False, multimodal=False, separable=True):
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
        super(PowellSingular2, self).__init__(name, dims, continuous,
                                              convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Instanciating function
        f = 0

        # For every input dimension
        for i in range(x.shape[0] - 2):
            # Calculating the Powell's Singular 2nd function
            f += (x[i - 1] + 10 * x[i]) ** 2 + 5 * (x[i + 1] - x[i + 2]) ** 2 + (x[i] - 2 * x[i + 1]) ** 4 + 10 * (x[i - 1] - x[i + 2]) ** 4

        return f


class PowellSum(Benchmark):
    """PowellSum class implements the Powell's Sum benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}|x_i|^{i+1}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-1, 1] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='PowellSum', dims=-1, continuous=True, convex=True,
                 differentiable=False, multimodal=False, separable=True):
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
        super(PowellSum, self).__init__(name, dims, continuous,
                                        convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Instanciating function
        f = 0

        # For every input dimension
        for i in range(x.shape[0]):
            # Calculating the Powell's Sum function
            f += np.fabs(x[i]) ** (i + 2)

        return f


class Qing(Benchmark):
    """Qing class implements the Qing's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}(x_i^2 - i)^2

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-500, 500] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid x_i^* = (\pm \sqrt{i}, \pm \sqrt{i}, \ldots, \pm \sqrt{i})`.

    """

    def __init__(self, name='Qing', dims=-1, continuous=True, convex=False,
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
        super(Qing, self).__init__(name, dims, continuous,
                                   convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Instanciating function
        f = 0

        # For every input dimension
        for i in range(x.shape[0]):
            # Calculating the Qing's function
            f += (x[i] ** 2 - (i + 1)) ** 2

        return f


class Quartic(Benchmark):
    """Quartic class implements the Quartic's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}ix_i^4 + rand()

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-1.28, 1.28] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 + rand() \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='Quartic', dims=-1, continuous=True, convex=False,
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
        super(Quartic, self).__init__(name, dims, continuous,
                                      convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Instanciating function
        f = 0

        # For every input dimension
        for i in range(x.shape[0]):
            # Calculating the Quartic's function
            f += (i + 1) * (x[i] ** 4)

        return f + np.random.uniform()


class Quintic(Benchmark):
    """Quintic class implements the Quintic's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}|x_i^5 - 3x_i^4 + 4x_i^3 + 2x_i^2 - 10x_i - 4|

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (-1 or 2, -1 or 2, \ldots, -1 or 2)`.

    """

    def __init__(self, name='Quintic', dims=-1, continuous=True, convex=True,
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
        super(Quintic, self).__init__(name, dims, continuous,
                                      convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Quintic's function
        f = np.fabs(x ** 5 - 3 * x ** 4 + 4 * x ** 3 + 2 * x ** 2 - 10 * x - 4)

        return np.sum(f)


class Rana(Benchmark):
    """Rana class implements the Rana's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-2}(x_{i+1} + 1)cos(t_2)sin(t_1) + x_i cos(t_1)sin(t_2)

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-500, 500] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='Rana', dims=-1, continuous=True, convex=True,
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
        super(Rana, self).__init__(name, dims, continuous,
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

        # For every input dimension
        for i in range(x.shape[0] - 2):
            # Calculating `t1`
            t1 = np.sqrt(np.fabs(x[i + 1] + x[i] + 1))

            # Calculating `t2`
            t2 = np.sqrt(np.fabs(x[i + 1] - x[i] + 1))

            # Calculating the Rana's function
            f += (x[i + 1] + 1) * np.cos(t2) * np.sin(t1) + x[i] * np.cos(t1) * np.sin(t2)

        return f


class Rastrigin(Benchmark):
    """Rastrigin class implements the Rastrigin's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = 10n + \sum_{i=1}^{n}(x_i^2 - 10cos(2 \\pi x_i))

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5.12, 5.12] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='Rastrigin', dims=-1, continuous=True, convex=True,
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
        super(Rastrigin, self).__init__(name, dims, continuous,
                                        convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Rastrigin's function
        f = x ** 2 - 10 * np.cos(2 * np.pi * x)

        return 10 * x.shape[0] + np.sum(f)


class Ridge(Benchmark):
    """Ridge class implements the Ridge's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = x_1 + (\sum_{i=2}^{n}x_i^2)^{0.5}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-\lambda, \lambda]^n \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -\lambda \mid \mathbf{x^*} = (-\lambda, 0, \ldots, 0)`.

    """

    def __init__(self, name='Ridge', dims=-1, continuous=True, convex=False,
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
        super(Ridge, self).__init__(name, dims, continuous,
                                    convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Ridge's function
        f = x[1:] ** 2

        return x[0] + np.sum(f) ** 0.5


class Rosenbrock(Benchmark):
    """Rosenbrock class implements the Rosenbrock's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-1}[100(x_{i+1}-x_i^2)^2 + (x_i - 1)^2]

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-30, 30] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 1, \ldots, 1)`.

    """

    def __init__(self, name='Rosenbrock', dims=-1, continuous=True, convex=True,
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
        super(Rosenbrock, self).__init__(name, dims, continuous,
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

        # For every input dimension
        for i in range(x.shape[0] - 1):
            # Calculating the Rosenbrock's function
            f += 100 * ((x[i + 1] - x[i] ** 2) ** 2) + ((x[i] - 1) ** 2)

        return f

class RotatedHyperEllipsoid(Benchmark):
    """RotatedHyperEllipsoid class implements the Rotated Hyper-Ellipsoid's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}\sum_{j=1}^{i}x_j^2

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-65.536, 65.536] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='RotatedHyperEllipsoid', dims=-1, continuous=True, convex=True,
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
        super(RotatedHyperEllipsoid, self).__init__(name, dims, continuous,
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

        # For every input dimension
        for i in range(x.shape[0]):
            # For `j` in `i` range
            for j in range(i):
                # Calculating the Rotated Hyper-Ellipsoid's function
                f += x[j] ** 2

        return f


class Salomon(Benchmark):
    """Salomon class implements the Salomon's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = 1 - cos(2 \\pi \sqrt{\sum_{i=1}^{n}x_i^2}) + 0.1\sqrt{\sum_{i=1}^{n}x_i^2}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='Salomon', dims=-1, continuous=True, convex=False,
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
        super(Salomon, self).__init__(name, dims, continuous,
                                      convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Salomon's function
        f = 1 - np.cos(2 * np.pi * np.sqrt(np.sum(x ** 2))) + \
            0.1 * np.sqrt(np.sum(x ** 2))

        return f


class SchumerSteiglitz(Benchmark):
    """SchumerSteiglitz class implements the Schumer Steiglitz's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}x_i^4

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='SchumerSteiglitz', dims=-1, continuous=True, convex=True,
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
        super(SchumerSteiglitz, self).__init__(name, dims, continuous,
                                               convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Schumer Steiglitz's function
        f = x ** 4

        return np.sum(f)


class Schwefel(Benchmark):
    """Schwefel class implements the Schwefel's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = 418.9829n -\sum_{i=1}^{n} x_i sin(\sqrt{|x_i|})

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (420.9687, 420.9687, \ldots, 420.9687)`.

    """

    def __init__(self, name='Schwefel', dims=-1, continuous=True, convex=False,
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
        super(Schwefel, self).__init__(name, dims, continuous,
                                       convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Schwefel's function
        f = x * np.sin(np.sqrt(np.fabs(x)))

        return 418.9829 * x.shape[0] - np.sum(f)


class Schwefel220(Benchmark):
    """Schwefel220 class implements the Schwefel's 2.20 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}|x_i|

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='Schwefel220', dims=-1, continuous=True, convex=True,
                 differentiable=False, multimodal=False, separable=True):
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
        super(Schwefel220, self).__init__(name, dims, continuous,
                                          convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Schwefel's 2.20 function
        f = np.fabs(x)

        return np.sum(f)


class Schwefel221(Benchmark):
    """Schwefel221 class implements the Schwefel's 2.21 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \max_{i=1, \ldots, n}|x_i|

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='Schwefel221', dims=-1, continuous=True, convex=True,
                 differentiable=False, multimodal=False, separable=True):
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
        super(Schwefel221, self).__init__(name, dims, continuous,
                                          convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Schwefel's 2.21 function
        f = np.fabs(x)

        return np.amax(f)


class Schwefel222(Benchmark):
    """Schwefel222 class implements the Schwefel's 2.22 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}|x_i| + \prod_{i=1}^{n}|x_i|

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='Schwefel222', dims=-1, continuous=True, convex=True,
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
        super(Schwefel222, self).__init__(name, dims, continuous,
                                          convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Schwefel's 2.22 function
        f = np.fabs(x)

        return np.sum(f) + np.prod(f)


class Schwefel223(Benchmark):
    """Schwefel223 class implements the Schwefel's 2.23 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}x_i^{10}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='Schwefel223', dims=-1, continuous=True, convex=True,
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
        super(Schwefel223, self).__init__(name, dims, continuous,
                                          convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Schwefel's 2.23 function
        f = x ** 10

        return np.sum(f)


class Schwefel225(Benchmark):
    """Schwefel225 class implements the Schwefel's 2.25 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=2}^{n}(x_i - 1)^2 + (x_1 - x_i^2)^2

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 10] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 1, \ldots, 1)`.

    """

    def __init__(self, name='Schwefel225', dims=-1, continuous=True, convex=True,
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
        super(Schwefel225, self).__init__(name, dims, continuous,
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

        # For every input dimension starting from `1`
        for i in range(1, x.shape[0]):
            # Calculating the Schwefel's 2.25 function
            f += (x[i] - 1) ** 2 + (x[0] - x[i] ** 2) ** 2

        return f


class Schwefel226(Benchmark):
    """Schwefel226 class implements the Schwefel's 2.26 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = -\\frac{1}{n} \sum_{i=1}^{n}x_i sin(\sqrt{|x_i|})

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-500, 500] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -418.983 \mid \mathbf{x^*} = (\pm[\\pi (0.5+k)]^2, \pm[\\pi (0.5+k)]^2, \ldots, \pm[\\pi (0.5+k)]^2)`.

    """

    def __init__(self, name='Schwefel226', dims=-1, continuous=True, convex=True,
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
        super(Schwefel226, self).__init__(name, dims, continuous,
                                          convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Schwefel's 2.26 function
        f = x * np.sin(np.sqrt(np.fabs(x)))

        return -1 / x.shape[0] * np.sum(f)


class Shubert(Benchmark):
    """Shubert class implements the Shubert's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \prod_{i=1}^n \sum_{j=1}^{5}cos((j+1)x_i+j)

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -186.7309 \mid \mathbf{x^*} = \\text{multiple solutions}`.

    """

    def __init__(self, name='Shubert', dims=-1, continuous=True, convex=False,
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
        super(Shubert, self).__init__(name, dims, continuous,
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
        f = 1

        # For every input dimension
        for i in range(x.shape[0]):
            # For `j` from 1 to 5:
            for j in range(1, 6):
                # Calculating the Shubert's function
                f *= np.cos((j + 1) * x[i] + j)

        return f


class Shubert3(Benchmark):
    """Shubert3 class implements the Shubert's 3rd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^n \sum_{j=1}^{5}j sin((j+1)x_i+j)

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -29.6733337 \mid \mathbf{x^*} = (?, ?, \ldots, ?)`.

    """

    def __init__(self, name='Shubert3', dims=-1, continuous=True, convex=False,
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
        super(Shubert3, self).__init__(name, dims, continuous,
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

        # For every input dimension
        for i in range(x.shape[0]):
            # For `j` from 1 to 5:
            for j in range(1, 6):
                # Calculating the Shubert's 3rd function
                f += j * np.sin((j + 1) * x[i] + j)

        return f


class Shubert4(Benchmark):
    """Shubert4 class implements the Shubert's 4th benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^n \sum_{j=1}^{5}j cos((j+1)x_i+j)

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -25.740858 \mid \mathbf{x^*} = (?, ?, \ldots, ?)`.

    """

    def __init__(self, name='Shubert4', dims=-1, continuous=True, convex=False,
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
        super(Shubert4, self).__init__(name, dims, continuous,
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

        # For every input dimension
        for i in range(x.shape[0]):
            # For `j` from 1 to 5:
            for j in range(1, 6):
                # Calculating the Shubert's 4th function
                f += j * np.cos((j + 1) * x[i] + j)

        return f


class SchafferF6(Benchmark):
    """SchafferF6 class implements the Schaffer's F6 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-1}0.5 + \\frac{sin^2(\sqrt{x_i^2+x_{i+1}^2})-0.5}{[1 + 0.001(x_i^2 + x_{i+1}^2)]^2}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='SchafferF6', dims=-1, continuous=True, convex=False,
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
        super(SchafferF6, self).__init__(name, dims, continuous,
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

        # For every input dimension
        for i in range(x.shape[0] - 1):
            # Calculating the Schaffer's F6 function
            f += 0.5 + (np.sin(np.sqrt(x[i] ** 2 + x[i + 1] ** 2)) ** 2 - 0.5) / ((
                1 + 0.001 * (x[i] ** 2 + x[i + 1] ** 2)) ** 2)

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

        # Calculating the Sphere's function
        f = x ** 2

        return np.sum(f)


class Step(Benchmark):
    """Step class implements the Step's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} ⌊x_i⌋

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='Step', dims=-1, continuous=False, convex=False,
                 differentiable=False, multimodal=False, separable=True):
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
        super(Step, self).__init__(name, dims, continuous,
                                   convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Step's function
        f = np.floor(np.fabs(x))

        return np.sum(f)


class Step2(Benchmark):
    """Step2 class implements the Step's 2nd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} ⌊x_i + 0.5⌋^2

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (-0.5, -0.5, \ldots, -0.5)`.

    """

    def __init__(self, name='Step2', dims=-1, continuous=False, convex=False,
                 differentiable=False, multimodal=False, separable=True):
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
        super(Step2, self).__init__(name, dims, continuous,
                                    convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Step's 2nd function
        f = np.floor(x + 0.5) ** 2

        return np.sum(f)


class Step3(Benchmark):
    """Step3 class implements the Step's 3rd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} ⌊x_i^2⌋

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='Step3', dims=-1, continuous=False, convex=False,
                 differentiable=False, multimodal=False, separable=True):
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
        super(Step3, self).__init__(name, dims, continuous,
                                    convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Step's 3rd function
        f = np.floor(x ** 2)

        return np.sum(f)


class StrechedVSineWave(Benchmark):
    """StrechedVSineWave class implements the Streched V Sine Wave's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-1}(x_{i+1}^2 + x_i^2)^{0.25}[sin^2(50(x_{i+1}^2 + x_i^2)^{0.1})+0.1]

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='StrechedVSineWave', dims=-1, continuous=True, convex=True,
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
        super(StrechedVSineWave, self).__init__(name, dims, continuous,
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

        # For every input dimension
        for i in range(x.shape[0] - 1):
            # Calculating the Streched V Sine Wave's function
            f += (((x[i + 1] ** 2) + (x[i] ** 2)) ** 0.25) * ((np.sin(50 * ((x[i + 1] ** 2) + (x[i] ** 2)) ** 0.1) ** 2) + 0.1)

        return f

class StyblinskiTang(Benchmark):
    """StyblinskiTang class implements the Styblinski-Tang's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \\frac{1}{2}\sum_{i=1}^{n}(x_i^4 - 16x_i^2 + 5x_i)

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = −39.16599n \mid \mathbf{x^*} = (-2.903534, -2.903534, \ldots, -2.903534)`.

    """

    def __init__(self, name='StyblinskiTang', dims=-1, continuous=True, convex=False,
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
        super(StyblinskiTang, self).__init__(name, dims, continuous,
                                             convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Styblinski-Tang's function
        f = 1 / 2 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x)

        return f

class SumDifferentPowers(Benchmark):
    """SumDifferentPowers class implements the Sum of Different Powers' benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}|x_i|^{i+1}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-1, 1] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='SumDifferentPowers', dims=-1, continuous=True, convex=True,
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
        super(SumDifferentPowers, self).__init__(name, dims, continuous,
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

        # For every input dimension
        for i in range(x.shape[0]):
            # Calculating the Sum of Different Powers' function
            f += np.fabs(x[i]) ** (i + 2)

        return f

class SumSquares(Benchmark):
    """SumSquares class implements the Sum of Squares' benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}ix_i^2

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='SumSquares', dims=-1, continuous=True, convex=True,
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
        super(SumSquares, self).__init__(name, dims, continuous,
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

        # For every input dimension
        for i in range(x.shape[0]):
            # Calculating the Sum of Squares' function
            f += (i + 1) * (x[i] ** 2)

        return f


class Trid(Benchmark):
    """Trid class implements the Trid's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}(x_i - 1)^2 - \sum_{i=2}^{n}x_i x_{i-1}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-n^2, n^2] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -\\frac{n(n+4)(n-1)}{6} \mid x_i = i(n+1-i)`.

    """

    def __init__(self, name='Trid', dims=-1, continuous=True, convex=True,
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
        super(Trid, self).__init__(name, dims, continuous,
                                   convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Instantiating term
        term = 0

        for i in range(1, x.shape[0]):
            term += x[i] * x[i - 1]

        # Calculating the Trid's function
        f = np.sum((x - 1) ** 2) - term

        return f

class Trigonometric1(Benchmark):
    """Trigonometric1 class implements the Trigonometric's 1st benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}[n - \sum_{j=1}^{n} cos(x_j) + i(1 - cos(x_i) - sin(x_i))]^2

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, \\pi] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='Trigonometric1', dims=-1, continuous=True, convex=True,
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
        super(Trigonometric1, self).__init__(name, dims, continuous,
                                             convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Defining the input dimension
        n = x.shape[0]

        # Instantiating function
        f = 0

        # For every input dimension
        for i in range(n):
            # Resetting partial term
            partial = 0

            # For every input dimension
            for j in range(n):
                # Calculating partial term
                partial += np.cos(x[j])

            # Calculating the Trigonometric's 1st function
            f += (n - partial + i * (1 - np.cos(x[i] - np.sin(x[i])))) ** 2

        return f

class Trigonometric2(Benchmark):
    """Trigonometric2 class implements the Trigonometric's 2nd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = 1 + \sum_{i=1}^{n}8sin^2[7(x_i - 0.9)^2] + 6sin^2[14(x_1-0.9)^2] + (x_i-0.9)^2

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-500, 500] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 1 \mid \mathbf{x^*} = (0.9, 0.9, \ldots, 0.9)`.

    """

    def __init__(self, name='Trigonometric2', dims=-1, continuous=True, convex=True,
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
        super(Trigonometric2, self).__init__(name, dims, continuous,
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

        # For every input dimension
        for i in range(x.shape[0]):
            # Calculating the Trigonometric's 2nd function
            f += 8 * (np.sin(7 * (x[i] - 0.9) ** 2) ** 2) + 6 * (np.sin(14 * (x[0] - 0.9) ** 2) ** 2) + ((x[i] - 0.9) ** 2)

        return 1 + f


class Wavy(Benchmark):
    """Wavy class implements the Wavy's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = 1 - \\frac{1}{n} \sum_{i=1}^{n}cos(10x_i)e^{\\frac{-x_i^2}{2}}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-\\pi, \\pi] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='Wavy', dims=-1, continuous=True, convex=True,
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
        super(Wavy, self).__init__(name, dims, continuous,
                                   convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Wavy's function
        f = np.cos(10 * x) * np.exp(-1 * (x ** 2) / 2)

        return 1 - (1 / x.shape[0]) * np.sum(f)


class XinSheYang(Benchmark):
    """XinSheYang class implements the Xin-She Yang's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}\epsilon_i|x_i|^i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='XinSheYang', dims=-1, continuous=True, convex=False,
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
        super(XinSheYang, self).__init__(name, dims, continuous,
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

        # For every decision variable
        for i in range(x.shape[0]):
            # Calculating the Xin-She Yang's function
            f += np.random.uniform() * (np.fabs(x[i]) ** (i + 1))

        return f


class XinSheYang2(Benchmark):
    """XinSheYang2 class implements the Xin-She Yang's 2nd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = (\sum_{i=1}^{n}|x_i|)e^{-\sum_{i=1}^{n}sin(x_i^2)}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-2 \\pi, 2 \\pi] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='XinSheYang2', dims=-1, continuous=True, convex=False,
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
        super(XinSheYang2, self).__init__(name, dims, continuous,
                                          convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Xin-She Yang's 2nd function
        f = np.sum(np.fabs(x)) * np.exp(-np.sum(np.sin(x ** 2)))

        return f

class XinSheYang3(Benchmark):
    """XinSheYang3 class implements the Xin-She Yang's 3rd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = e^{-\sum_{i=1}^{n}(\\frac{x_i}{\\beta})^{2m}} - 2e^{-\sum_{i=1}^{n}x_i^2} \prod_{i=1}^{n} cos^2(x_i)

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-2 \\pi, 2 \\pi] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -1 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='XinSheYang3', dims=-1, continuous=True, convex=False,
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
        super(XinSheYang3, self).__init__(name, dims, continuous,
                                          convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Xin-She Yang's 3rd function
        f = np.exp(-np.sum((x / 15) ** 10)) - 2 * np.exp(-np.sum(x ** 2)) * np.prod(np.cos(x) ** 2)

        return f

class XinSheYang4(Benchmark):
    """XinSheYang4 class implements the Xin-She Yang's 4th benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = (\sum_{i=1}^{n} sin^2(x_i) - e^{-\sum_{i=1}^{n}x_i^2})e^{-\sum_{i=1}^{n}sin^2(\sqrt{|x_i|})}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -1 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='XinSheYang4', dims=-1, continuous=True, convex=False,
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
        super(XinSheYang4, self).__init__(name, dims, continuous,
                                          convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Calculating the Xin-She Yang's 4th function
        f = (np.sum(np.sin(x) ** 2) - np.exp(-np.sum(x ** 2))) * np.exp(-np.sum(np.sin(np.sqrt(np.fabs(x)) ** 2)))

        return f


class Zakharov(Benchmark):
    """Zakharov class implements the Zakharov's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^n x_i^{2}+(\sum_{i=1}^n 0.5ix_i)^2 + (\sum_{i=1}^n 0.5ix_i)^4

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5, 10] \mid i = \{1, 2, \ldots, n\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    def __init__(self, name='Zakharov', dims=-1, continuous=True, convex=True,
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
        super(Zakharov, self).__init__(name, dims, continuous,
                                       convex, differentiable, multimodal, separable)

    @d.check_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Instantiating term
        term = 0

        # For every input dimension
        for i in range(x.shape[0]):
            term += 0.5 * i * x[i]

        # Calculating the Zakharov's function
        f = np.sum(x) + (term ** 2) + (term ** 4)

        return f
