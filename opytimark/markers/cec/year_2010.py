"""CEC2010 benchmarking functions.
"""

import numpy as np

import opytimark.markers.n_dimensional as n_dim
import opytimark.utils.decorator as d
import opytimark.utils.exception as e
from opytimark.core import CECBenchmark


class F1(CECBenchmark):
    """F1 class implements the Shifted Elliptic's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (10^6)^\\frac{i-1}{n-1} z_i^2 \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(self, name='F1', year='2010', auxiliary_data=('o'), dims=1000,
                 continuous=True, convex=True, differentiable=True, multimodal=False, separable=True):
        """Initialization method.

        Args:
            name (str): Name of the function.
            year (str): Year of the function.
            auxiliary_data (tuple): Auxiliary variables to be externally loaded.
            dims (int): Number of allowed dimensions.
            continuous (bool): Whether the function is continuous.
            convex (bool): Whether the function is convex.
            differentiable (bool): Whether the function is differentiable.
            multimodal (bool): Whether the function is multimodal.
            separable (bool): Whether the function is separable.

        """

        # Override its parent class
        super(F1, self).__init__(name, year, auxiliary_data, dims, continuous,
                                 convex, differentiable, multimodal, separable)

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
            # Calculating the Shifted Elliptic's function
            f += 10e6 ** (i / (x.shape[0] - 1)) * z[i] ** 2

        return f


class F2(CECBenchmark):
    """F2 class implements the Shifted Rastrigin's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (z_i^2 - 10cos(2 \\pi z_i) + 10) \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(self, name='F2', year='2010', auxiliary_data=('o'), dims=1000,
                 continuous=True, convex=True, differentiable=True, multimodal=True, separable=True):
        """Initialization method.

        Args:
            name (str): Name of the function.
            year (str): Year of the function.
            auxiliary_data (tuple): Auxiliary variables to be externally loaded.
            dims (int): Number of allowed dimensions.
            continuous (bool): Whether the function is continuous.
            convex (bool): Whether the function is convex.
            differentiable (bool): Whether the function is differentiable.
            multimodal (bool): Whether the function is multimodal.
            separable (bool): Whether the function is separable.

        """

        # Override its parent class
        super(F2, self).__init__(name, year, auxiliary_data, dims, continuous,
                                 convex, differentiable, multimodal, separable)

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

        # Calculating the Shifted Rastrigin's function
        f = z ** 2 - 10 * np.cos(2 * np.pi * z) + 10

        return np.sum(f)


class F3(CECBenchmark):
    """F3 class implements the Shifted Ackley's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = -20e^{-0.2\sqrt{\\frac{1}{n}\sum_{i=1}^{n}x_i^2}}-e^{\\frac{1}{n}\sum_{i=1}^{n}cos(2 \\pi x_i)}+ 20 + e \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-32, 32] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(self, name='F3', year='2010', auxiliary_data=('o'), dims=1000,
                 continuous=True, convex=True, differentiable=True, multimodal=True, separable=True):
        """Initialization method.

        Args:
            name (str): Name of the function.
            year (str): Year of the function.
            auxiliary_data (tuple): Auxiliary variables to be externally loaded.
            dims (int): Number of allowed dimensions.
            continuous (bool): Whether the function is continuous.
            convex (bool): Whether the function is convex.
            differentiable (bool): Whether the function is differentiable.
            multimodal (bool): Whether the function is multimodal.
            separable (bool): Whether the function is separable.

        """

        # Override its parent class
        super(F3, self).__init__(name, year, auxiliary_data, dims, continuous,
                                 convex, differentiable, multimodal, separable)

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

        # Calculating the 1 / n term
        inv = 1 / x.shape[0]

        # Calculating first term
        term1 = -0.2 * np.sqrt(inv * np.sum(z ** 2))

        # Calculating second term
        term2 = inv * np.sum(np.cos(2 * np.pi * z))

        # Calculating Shifted Ackley's function
        f = 20 + np.e - np.exp(term2) - 20 * np.exp(term1)

        return np.sum(f)


class F4(CECBenchmark):
    """F4 class implements the Single-group Shifted and m-rotated Elliptic Function benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = f_{rot\_elliptic}[z(P_1:P_m)] * 10^6 + f_{elliptic}[z(P_{m+1}:P_n)] \mid z_i = x_i - o_i and z_i = (x_i - o_i) \\ast M_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(self, name='F4', year='2010', auxiliary_data=('o', 'p', 'M'), dims=1000, group_size=50,
                 continuous=True, convex=True, differentiable=True, multimodal=True, separable=True):
        """Initialization method.

        Args:
            name (str): Name of the function.
            year (str): Year of the function.
            auxiliary_data (tuple): Auxiliary variables to be externally loaded.
            dims (int): Number of allowed dimensions.
            group_size (int): Size of function's group, i.e., `m` variable.
            continuous (bool): Whether the function is continuous.
            convex (bool): Whether the function is convex.
            differentiable (bool): Whether the function is differentiable.
            multimodal (bool): Whether the function is multimodal.
            separable (bool): Whether the function is separable.

        """

        # Override its parent class
        super(F4, self).__init__(name, year, auxiliary_data, dims, continuous,
                                 convex, differentiable, multimodal, separable)

        # Defines the size of the group
        self.m = group_size

    @d.check_less_equal_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions and the benchmarking function
        D = x.shape[0]
        f = n_dim.HighConditionedElliptic()

        #
        if self.m >= D:
            raise e.SizeError('`group_size` should be smaller than number of input dimensions')

        # Calculates an array of permutations and defines both groups' indexes
        p = np.random.permutation(D)
        p_1 = p[:self.m]
        p_2 = p[self.m:]

        # Shifts the input data
        s = x - self.o[:D]

        # Re-calculates both groups' inputs
        z_rot = np.dot(s[p_1], self.M[D][D])
        z = s[p_2]

        return f(z_rot) * 10 ** 6 + f(z)
