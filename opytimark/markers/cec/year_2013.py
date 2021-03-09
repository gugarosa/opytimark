"""CEC2014 benchmarking functions.
"""

import numpy as np

import opytimark.markers.n_dimensional as n_dim
import opytimark.utils.decorator as d
import opytimark.utils.exception as e
from opytimark.core import CECBenchmark

# Fixes Numpy's random seed
np.random.seed(0)


class F4(CECBenchmark):
    """F4 class implements the Shifted Elliptic's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (10^6)^\\frac{i-1}{n-1} z_i^2 \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(self, name='F4', year='2013', auxiliary_data=('o', 'R25', 'R50', 'R100'), dims=1000,
                 continuous=True, convex=True, differentiable=True, multimodal=False, separable=False):
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
        super(F4, self).__init__(name, year, auxiliary_data, dims, continuous,
                                 convex, differentiable, multimodal, separable)

        #
        self.S = [50, 25, 25, 100, 50, 25, 25]
        self.W = [45.6996306147733, 1.56461588893232, 18465.3234457619, 0.0110894989182919, 13.6259848988855, 0.301515061772251, 59.6078373100912]
        self.f = n_dim.HighConditionedElliptic()

    @d.check_less_equal_dimension
    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions
        D = x.shape[0]

        # If group size is bigger or equal to number of dimensions
        if D < 300:
            # Raises an error
            raise e.SizeError('`D` should be greater than 300')

        # Calculates an array of permutations and defines groups' indexes
        P = np.random.permutation(D)
        p = 0

        # Re-calculates the input
        y = x - self.o[:D]

        #
        f = 0

        #
        for s, w in zip(self.S, self.W):
            if s == 25:
                z = np.matmul(self.R25, y[P[p:p+s]])
                fit = self.f(z)
            elif s == 50:
                z = np.matmul(self.R50, y[P[p:p+s]])
                fit = self.f(z)
            elif s == 100:
                z = np.matmul(self.R100, y[P[p:p+s]])
                fit = self.f(z)
            p += s
            f += w * fit

        if D >= 302:
            z = y[P[p:]]
            fit = self.f(z)
            f += fit
        
        return f