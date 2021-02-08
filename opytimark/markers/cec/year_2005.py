"""CEC2005 benchmarking functions.
"""

import numpy as np

import opytimark.utils.decorator as d
import opytimark.utils.loader as l
from opytimark.core import Benchmark


class F1(Benchmark):
    """F1 class implements the Shifted Sphere's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} z_i^2 - 450 \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -450 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(self, name='F1', dims=100, continuous=True, convex=False,
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
        super(F1, self).__init__(name, dims, continuous,
                                 convex, differentiable, multimodal, separable)

        # Loads auxiliary data and define it as a property
        self.o = l.load_cec_auxiliary('F1', '2005')

    @d.check_less_equal_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Re-calculates the input
        z = x - self.o[:x.shape[0]]

        # Calculating the Shifted Sphere's function
        f = z ** 2

        return np.sum(f) - 450


class F2(Benchmark):
    """F2 class implements the Shifted Schwefel's 1.2 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (\sum_{j=1}^i z_j)^2 - 450 \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -450 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(self, name='F2', dims=100, continuous=True, convex=False,
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
        super(F2, self).__init__(name, dims, continuous,
                                 convex, differentiable, multimodal, separable)

        # Loads auxiliary data and define it as a property
        self.o = l.load_cec_auxiliary('F2', '2005')

    @d.check_less_equal_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Re-calculates the input
        z = x - self.o[:x.shape[0]]

        # Instantiating function
        f = 0

        # For every input dimension
        for i in range(x.shape[0]):
            # Resetting partial term
            partial = 0

            # For every dimension till `i`
            for j in range(i):
                # Sums up the partial term
                partial += z[j]

            # Calculating the Shifted Schwefel's 1.2 function
            f += partial ** 2

        return f - 450


class F3(Benchmark):
    """F3 class implements the Shifted Rotated High Conditioned Elliptic's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (10^6)^\\frac{i-1}{n-1} z_i^2 - 450 \mid z_i = (x_i - o_i) * M_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \in \{2, 10, 30, 50\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -450 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(self, name='F3', dims=-1, continuous=True, convex=True,
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
        super(F3, self).__init__(name, dims, continuous,
                                 convex, differentiable, multimodal, separable)

        # Loads auxiliary data and define it as a property
        self.o = l.load_cec_auxiliary('F3', '2005')

        # Pre-loads every auxiliary matrix for faster computing
        self.M_2 = l.load_cec_auxiliary('F3_D2', '2005')
        self.M_10 = l.load_cec_auxiliary('F3_D10', '2005')
        self.M_30 = l.load_cec_auxiliary('F3_D30', '2005')
        self.M_50 = l.load_cec_auxiliary('F3_D50', '2005')

    @d.check_exact_dimension_and_auxiliary_matrix
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Re-calculates the input
        z = np.matmul(x - self.o[:x.shape[0]], self.M)

        # Instantiating function
        f = 0

        # For every input dimension
        for i in range(x.shape[0]):
            # Calculating the Shifted Rotated High Conditioned Elliptic's function
            f += 10e6 ** (i / (x.shape[0] - 1)) * z[i] ** 2

        return f - 450


class F4(Benchmark):
    """F4 class implements the Shifted Schwefel's 1.2 with Noise in Fitness benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (\sum_{j=1}^i z_j)^2 * (1 + 0.4|N(0,1)|) - 450 \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -450 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(self, name='F4', dims=100, continuous=True, convex=False,
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
        super(F4, self).__init__(name, dims, continuous,
                                 convex, differentiable, multimodal, separable)

        # Loads auxiliary data and define it as a property
        self.o = l.load_cec_auxiliary('F4', '2005')

    @d.check_less_equal_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Re-calculates the input
        z = x - self.o[:x.shape[0]]

        # Instantiating function
        f = 0

        # For every input dimension
        for i in range(x.shape[0]):
            # Resetting partial term
            partial = 0

            # For every dimension till `i`
            for j in range(i):
                # Sums up the partial term
                partial += z[j]

            # Calculating the Shifted Schwefel's 1.2 with Noise in Fitness function
            f += partial ** 2

        # Generates a random uniform noise
        noise = np.random.uniform()

        return f * (1 + 0.4 * noise) - 450


class F6(Benchmark):
    """F6 class implements the Shifted Rosenbrock's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-1} (100(z_i^2-z_{i+1})^2 + (z_i - 1)^2) + 390 \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -390 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(self, name='F6', dims=100, continuous=True, convex=False,
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
        super(F6, self).__init__(name, dims, continuous,
                                 convex, differentiable, multimodal, separable)

        # Loads auxiliary data and define it as a property
        self.o = l.load_cec_auxiliary('F6', '2005')

    @d.check_less_equal_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Re-calculates the input
        z = x - self.o[:x.shape[0]] + 1

        # Instantiating function
        f = 0

        # For every input dimension
        for i in range(x.shape[0] - 1):
            # Calculating the Shifted Rosenbrock's function
            f += (100 * (z[i] ** 2 - z[i+1]) ** 2 + (z[i] - 1) ** 2)

        return f + 390


class F7(Benchmark):
    """F7 class implements the Shifted Rotated Griewank's Function without Bounds benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = 1 + \sum_{i=1}^{n}\\frac{x_i^2}{4000} - \prod cos(\\frac{x_i}{\sqrt{i}}) - 180 \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 600] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -180 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(self, name='F7', dims=-1, continuous=True, convex=False,
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
        super(F7, self).__init__(name, dims, continuous,
                                 convex, differentiable, multimodal, separable)

        # Loads auxiliary data and define it as a property
        self.o = l.load_cec_auxiliary('F7', '2005')

        # Pre-loads every auxiliary matrix for faster computing
        self.M_2 = l.load_cec_auxiliary('F7_D2', '2005')
        self.M_10 = l.load_cec_auxiliary('F7_D10', '2005')
        self.M_30 = l.load_cec_auxiliary('F7_D30', '2005')
        self.M_50 = l.load_cec_auxiliary('F7_D50', '2005')

    @d.check_exact_dimension_and_auxiliary_matrix
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Re-calculates the input
        z = np.matmul(x - self.o[:x.shape[0]], self.M)

        # Initializing terms
        term1, term2 = 0, 1

        # For every possible dimension of `x`
        for i in range(x.shape[0]):
            # Calculating first term
            term1 += (z[i] ** 2) / 4000

            # Calculating second term
            term2 *= np.cos(z[i] / np.sqrt(i + 1))

        # Calculating the Shifted Rotated Griewank's Function without Bounds function
        f = 1 + term1 - term2

        return f - 180
