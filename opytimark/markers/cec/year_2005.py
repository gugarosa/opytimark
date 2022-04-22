"""CEC2005 benchmarking functions.
"""

from typing import Optional, Tuple

import numpy as np

import opytimark.markers.n_dimensional as n_dim
import opytimark.utils.decorator as d
from opytimark.core import CECBenchmark, CECCompositeBenchmark

# Fixes Numpy's random seed
np.random.seed(0)


class F1(CECBenchmark):
    """F1 class implements the Shifted Sphere's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} z_i^2 - 450 \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -450 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F1",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o"),
        dims: Optional[int] = 100,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = False,
        separable: Optional[bool] = True,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F1, self).__init__(
            name,
            year,
            auxiliary_data,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Re-calculates the input
        z = x - self.o[: x.shape[0]]

        # Calculating the Shifted Sphere's function
        f = z**2

        return np.sum(f) - 450


class F2(CECBenchmark):
    """F2 class implements the Shifted Schwefel's 1.2 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (\sum_{j=1}^i z_j)^2 - 450 \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -450 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F2",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o"),
        dims: Optional[int] = 100,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = False,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F2, self).__init__(
            name,
            year,
            auxiliary_data,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Re-calculates the input
        z = x - self.o[: x.shape[0]]

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
            f += partial**2

        return f - 450


class F3(CECBenchmark):
    """F3 class implements the Shifted Rotated High Conditioned Elliptic's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (10^6)^\\frac{i-1}{n-1} z_i^2 - 450 \mid z_i = (x_i - o_i) * M_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \in \{2, 10, 30, 50\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -450 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F3",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M2", "M10", "M30", "M50"),
        dims: Optional[int] = -1,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = False,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F3, self).__init__(
            name,
            year,
            auxiliary_data,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )

    @d.check_exact_dimension_and_auxiliary_matrix
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions and an equally-spaced interval between 0 and D-1
        D = x.shape[0]
        dims = np.linspace(1, D, D) - 1

        # Re-calculates the input
        z = np.matmul(x - self.o[:D], self.M)

        # Calculating the Shifted Rotated High Conditioned Elliptic's function
        z = 10e6 ** (dims / (D - 1)) * z**2

        return np.sum(z) - 450


class F4(CECBenchmark):
    """F4 class implements the Shifted Schwefel's 1.2 with Noise in Fitness benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (\sum_{j=1}^i z_j)^2 * (1 + 0.4|N(0,1)|) - 450 \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -450 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F4",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o"),
        dims: Optional[int] = 100,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = False,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F4, self).__init__(
            name,
            year,
            auxiliary_data,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Re-calculates the input
        z = x - self.o[: x.shape[0]]

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
            f += partial**2

        # Generates a random uniform noise
        noise = np.random.uniform()

        return f * (1 + 0.4 * noise) - 450


class F5(CECBenchmark):
    """F5 class implements the Schwefel's Problem 2.6 with Global Optimum on Bounds benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \max{|A_i x - B_i|} - 310 \mid B_i = A_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -310 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F5",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "A"),
        dims: Optional[int] = 100,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = False,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F5, self).__init__(
            name,
            year,
            auxiliary_data,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the shift re-arrangement points
        shift_1 = int(x.shape[0] / 4)
        shift_2 = int(3 * x.shape[0] / 4)

        # Re-sets `o` values
        self.o[:shift_1] = -100
        self.o[shift_2:] = 100

        # Gathers the correct input
        A = self.A[: x.shape[0], : x.shape[0]]

        # Calculates the `B` matrix
        B = np.matmul(A, self.o[: x.shape[0]])

        # Calculating the Schwefel's Problem 2.6 with Global Optimum on Bounds function
        f = np.max(np.fabs(np.matmul(A, x) - B))

        return f - 310


class F6(CECBenchmark):
    """F6 class implements the Shifted Rosenbrock's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-1} (100(z_i^2-z_{i+1})^2 + (z_i - 1)^2) + 390 \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -390 \mid \mathbf{x^*} = \mathbf{o} + 1`.

    """

    def __init__(
        self,
        name: Optional[str] = "F6",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o"),
        dims: Optional[int] = 100,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F6, self).__init__(
            name,
            year,
            auxiliary_data,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Re-calculates the input
        z = x - self.o[: x.shape[0]]

        # Instantiating function
        f = 0

        # For every input dimension
        for i in range(x.shape[0] - 1):
            # Calculating the Shifted Rosenbrock's function
            f += 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2

        return f + 390


class F7(CECBenchmark):
    """F7 class implements the Shifted Rotated Griewank's without Bounds benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = 1 + \sum_{i=1}^{n}\\frac{x_i^2}{4000} - \prod cos(\\frac{x_i}{\sqrt{i}}) - 180 \mid z_i = (x_i - o_i) * M_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 600] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -180 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F7",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M2", "M10", "M30", "M50"),
        dims: Optional[int] = -1,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F7, self).__init__(
            name,
            year,
            auxiliary_data,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )

    @d.check_exact_dimension_and_auxiliary_matrix
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Re-calculates the input
        z = np.matmul(x - self.o[: x.shape[0]], self.M)

        # Initializing terms
        term1, term2 = 0, 1

        # For every possible dimension of `x`
        for i in range(x.shape[0]):
            # Calculating first term
            term1 += (z[i] ** 2) / 4000

            # Calculating second term
            term2 *= np.cos(z[i] / np.sqrt(i + 1))

        # Calculating the Shifted Rotated Griewank's without Bounds function
        f = 1 + term1 - term2

        return f - 180


class F8(CECBenchmark):
    """F8 class implements the Shifted Rotated Ackley's with Global Optimum on Bounds benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = -20e^{-0.2\sqrt{\\frac{1}{n}\sum_{i=1}^{n}x_i^2}}-e^{\\frac{1}{n}\sum_{i=1}^{n}cos(2 \\pi x_i)}+ 20 + e - 140 \mid z_i = (x_i - o_i) * M_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-32, 32] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -140 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F8",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M2", "M10", "M30", "M50"),
        dims: Optional[int] = -1,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F8, self).__init__(
            name,
            year,
            auxiliary_data,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )

    @d.check_exact_dimension_and_auxiliary_matrix
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the shift point for re-arrangement
        shift = int(x.shape[0] / 2)

        # Iterates till reach the shift point
        for j in range(shift):
            # Re-sets the value of `o`
            self.o[2 * j] = -32 * self.o[2 * j + 1]

        # Re-calculates the input
        z = np.matmul(x - self.o[: x.shape[0]], self.M)

        # Calculating the 1 / n term
        inv = 1 / x.shape[0]

        # Calculating first term
        term1 = -0.2 * np.sqrt(inv * np.sum(z**2))

        # Calculating second term
        term2 = inv * np.sum(np.cos(2 * np.pi * z))

        # Calculating Shifted Rotated Ackley's Function with Global Optimum on Bounds function
        f = 20 + np.e - np.exp(term2) - 20 * np.exp(term1)

        return np.sum(f) - 140


class F9(CECBenchmark):
    """F9 class implements the Shifted Rastrigin's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (z_i^2 - 10cos(2 \\pi z_i) + 10) - 330 \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -330 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F9",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o"),
        dims: Optional[int] = 100,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = True,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F9, self).__init__(
            name,
            year,
            auxiliary_data,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Re-calculates the input
        z = x - self.o[: x.shape[0]]

        # Calculating the Shifted Rastrigin's function
        f = z**2 - 10 * np.cos(2 * np.pi * z) + 10

        return np.sum(f) - 330


class F10(CECBenchmark):
    """F10 class implements the Shifted Rotated Rastrigin's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (z_i^2 - 10cos(2 \\pi z_i) + 10) - 330 \mid z_i = (x_i - o_i) * M_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -330 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F10",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M2", "M10", "M30", "M50"),
        dims: Optional[int] = 100,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F10, self).__init__(
            name,
            year,
            auxiliary_data,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )

    @d.check_exact_dimension_and_auxiliary_matrix
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Re-calculates the input
        z = np.matmul(x - self.o[: x.shape[0]], self.M)

        # Calculating the Shifted Rastrigin's function
        f = z**2 - 10 * np.cos(2 * np.pi * z) + 10

        return np.sum(f) - 330


class F11(CECBenchmark):
    """F11 class implements the Shifted Rotated Weierstrass's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (\sum_{k=0}^{20} [0.5^k cos(2\\pi 3^k(z_i+0.5))]) - n \sum_{k=0}^{20}[0.5^k cos(2\\pi 3^k 0.5)] \mid z_i = (x_i - o_i) * M_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-0.5, 0.5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 90 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F11",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M2", "M10", "M30", "M50"),
        dims: Optional[int] = 100,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F11, self).__init__(
            name,
            year,
            auxiliary_data,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )

    @d.check_exact_dimension_and_auxiliary_matrix
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Re-calculates the input
        z = np.matmul(x - self.o[: x.shape[0]], self.M)

        # Instantiates the function
        f = 0

        # For every possible dimension of `x`
        for i in range(x.shape[0]):
            # Iterates until `k_max = 20`
            for k in range(21):
                # Adds the first term
                f += 0.5**k * np.cos(2 * np.pi * 3**k * (z[i] + 0.5))

        # Iterates again until `k_max = 20`
        for k in range(21):
            # Adds the second term
            f -= x.shape[0] * (0.5**k * np.cos(2 * np.pi * 3**k * 0.5))

        return f + 90


class F12(CECBenchmark):
    """F12 class implements the Schwefel's Problem 2.13 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (A_i - B_i)^2 - 460 \mid A_i = \sum_{j=1}^{n} a_{ij} sin(\\alpha_j) + b_{ij} cos(\\alpha_j), A_i = \sum_{j=1}^{n} a_{ij} sin(x_j) + b_{ij} cos(x_j)

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-\\pi, \\pi] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -460 \mid \mathbf{x^*} = \mathbf{\alpha}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F12",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("alpha", "a", "b"),
        dims: Optional[int] = 100,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F12, self).__init__(
            name,
            year,
            auxiliary_data,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Gathers the correct input
        alpha = self.alpha[: x.shape[0]]
        a = self.a[: x.shape[0], : x.shape[0]]
        b = self.b[: x.shape[0], : x.shape[0]]

        # Calculates the `A` and `B` matrices
        A = a * np.sin(alpha) + b * np.cos(alpha)
        B = a * np.sin(x) + b * np.cos(x)

        # Calculating the Schwefel's Problem 2.13 function
        f = (A - B) ** 2

        return np.sum(f) - 460


class F13(CECBenchmark):
    """F13 class implements the Shifted Expanded Griewank's plus Rosenbrock's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) =  f(x_1, x_2) + f(x_2, x_3) + \ldots + f(x_n, x_1) - 130 \mid z_i = x_i - o_i + 1

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-3, 1] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -130 \mid \mathbf{x^*} = \mathbf{\alpha}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F13",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o"),
        dims: Optional[int] = 100,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F13, self).__init__(
            name,
            year,
            auxiliary_data,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        def _griewank(x):
            return x**2 / 4000 - np.cos(x / np.sqrt(1)) + 1

        def _rosenbrock(x, y):
            return 100 * (x**2 - y) ** 2 + (x - 1) ** 2

        # Re-calculates the input
        z = x - self.o[: x.shape[0]] + 1

        # Instantiating function
        f = 0

        # Iterates through every dimension
        for i in range(x.shape[0]):
            # Checks if it is the last dimension
            if i == (x.shape[0] - 1):
                # Calculates the Shifted Expanded Griewank's plus Rosenbrock's function using indexes `n` and `0`
                f += _griewank(_rosenbrock(z[i], z[0]))

            # Checks if it is not the last dimension
            else:
                # Calculates the Shifted Expanded Griewank's plus Rosenbrock's function using indexes `i` and `i+1`
                f += _griewank(_rosenbrock(z[i], z[i + 1]))

        return f - 130


class F14(CECBenchmark):
    """F14 class implements the Shifted Rotated Expanded Scaffer's F6 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) =  f(x_1, x_2) + f(x_2, x_3) + \ldots + f(x_n, x_1) - 300 \mid z_i = x_i - o_i + 1

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -300 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F14",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M2", "M10", "M30", "M50"),
        dims: Optional[int] = 100,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F14, self).__init__(
            name,
            year,
            auxiliary_data,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )

    @d.check_exact_dimension_and_auxiliary_matrix
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        def _scaffer(x, y):
            return 0.5 + (np.sin(np.sqrt(x**2 + y**2)) ** 2 - 0.5) / (
                (1 + 0.0001 * (x**2 + y**2)) ** 2
            )

        # Re-calculates the input
        z = np.matmul(x - self.o[: x.shape[0]], self.M)

        # Instantiating function
        f = 0

        # Iterates through every dimension
        for i in range(x.shape[0]):
            # Checks if it is the last dimension
            if i == (x.shape[0] - 1):
                # Calculates the Shifted Rotated Expanded Scaffer's F6 function using indexes `n` and `0`
                f += _scaffer(z[i], z[0])

            # Checks if it is not the last dimension
            else:
                # Calculates the Shifted Rotated Expanded Scaffer's F6 function using indexes `i` and `i+1`
                f += _scaffer(z[i], z[i + 1])

        return f - 300


class F15(CECCompositeBenchmark):
    """F15 class implements the Hybrid Composition 1 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}{w_i \\ast [f_i'((\mathbf{x}-\mathbf{o_i})/ \\lambda_i \\ast \mathbf{M_i}) + bias_i]} + f_{bias}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 120 \mid \mathbf{x^*} = \mathbf{o_1}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F15",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M2", "M10", "M30", "M50"),
        bias: Optional[int] = 120,
        dims: Optional[int] = 100,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = True,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            bias: Composite function bias.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        # Defines `sigma` and `lambda` parameters
        sigma = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        l = (1, 1, 10, 10, 5 / 60, 5 / 60, 5 / 32, 5 / 32, 5 / 100, 5 / 100)

        # Defines the composite functions
        functions = (
            n_dim.Rastrigin(),
            n_dim.Rastrigin(),
            n_dim.Weierstrass(),
            n_dim.Weierstrass(),
            n_dim.Griewank(),
            n_dim.Griewank(),
            n_dim.Ackley1(),
            n_dim.Ackley1(),
            n_dim.Sphere(),
            n_dim.Sphere(),
        )

        super(F15, self).__init__(
            name,
            year,
            auxiliary_data,
            sigma,
            l,
            functions,
            bias,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )


class F16(CECCompositeBenchmark):
    """F16 class implements the Rotated Hybrid Composition 1 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}{w_i \\ast [f_i'((\mathbf{x}-\mathbf{o_i})/ \\lambda_i \\ast \mathbf{M_i}) + bias_i]} + f_{bias}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 120 \mid \mathbf{x^*} = \mathbf{o_1}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F16",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M2", "M10", "M30", "M50"),
        bias: Optional[int] = 120,
        dims: Optional[int] = 100,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            bias: Composite function bias.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        # Defines `sigma` and `lambda` parameters
        sigma = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        l = (1, 1, 10, 10, 5 / 60, 5 / 60, 5 / 32, 5 / 32, 5 / 100, 5 / 100)

        # Defines the composite functions
        functions = (
            n_dim.Rastrigin(),
            n_dim.Rastrigin(),
            n_dim.Weierstrass(),
            n_dim.Weierstrass(),
            n_dim.Griewank(),
            n_dim.Griewank(),
            n_dim.Ackley1(),
            n_dim.Ackley1(),
            n_dim.Sphere(),
            n_dim.Sphere(),
        )

        super(F16, self).__init__(
            name,
            year,
            auxiliary_data,
            sigma,
            l,
            functions,
            bias,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )


class F17(CECCompositeBenchmark):
    """F17 class implements the Rotated Hybrid Composition 1 with Noise benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}{w_i \\ast [f_i'((\mathbf{x}-\mathbf{o_i})/ \\lambda_i \\ast \mathbf{M_i}) + bias_i]} + f_{bias}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 120 \mid \mathbf{x^*} = \mathbf{o_1}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F17",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M2", "M10", "M30", "M50"),
        bias: Optional[int] = 120,
        dims: Optional[int] = 100,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            bias: Composite function bias.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        # Defines `sigma` and `lambda` parameters
        sigma = (1, 2, 1.5, 1.5, 1, 1, 1.5, 1.5, 2, 2)
        l = (5 / 16, 5 / 32, 2, 1, 1 / 10, 1 / 20, 20, 10, 1 / 6, 5 / 60)

        # Defines the composite functions
        functions = (
            n_dim.Ackley1(),
            n_dim.Ackley1(),
            n_dim.Rastrigin(),
            n_dim.Rastrigin(),
            n_dim.Sphere(),
            n_dim.Sphere(),
            n_dim.Weierstrass(),
            n_dim.Weierstrass(),
            n_dim.Griewank(),
            n_dim.Griewank(),
        )

        super(F17, self).__init__(
            name,
            year,
            auxiliary_data,
            sigma,
            l,
            functions,
            bias,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )

    @d.check_exact_dimension_and_auxiliary_matrix
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines some constants used throughout the method
        D = x.shape[0]
        n_composition = len(self.f)
        y = 5 * np.ones(x.shape[0])

        # Defines the array of `w`, fitness and maximum fitness
        w = np.zeros(n_composition)
        f_max = np.zeros(n_composition)
        fit = np.zeros(n_composition)

        # Iterates through every possible composition function
        for i, f in enumerate(self.f):
            # Re-calculates the solution
            z = x - self.o[i][:D]

            # Calculates the `w`
            w[i] = np.exp(-np.sum(z**2) / (2 * D * self.sigma[i] ** 2))

            # Calculates the start and end indexes of the shift matrix
            start, end = i * x.shape[0], (i + 1) * x.shape[0]

            # Calculates the maximum fitness
            f_max[i] = f(np.matmul(y / self.l[i], self.M[start:end]))

            # Calculates the fitness
            fit[i] = self.C * f(np.matmul(z / self.l[i], self.M[start:end])) / f_max[i]

        # Calculates the sum of `w` and the maximum `w`
        w_sum = np.sum(w)
        w_max = np.max(w)

        # Iterates through the number of composition functions
        for i in range(n_composition):
            # If current `w` is different than `w_max`
            if w[i] != w_max:
                # Re-scales its value
                w[i] *= 1 - w_max**10

            # Normalizes `w`
            w[i] /= w_sum

        # Calculates the fitness without noise
        g = np.sum(np.matmul(w, (fit + self.f_bias)))

        return g * (1 + 0.2 * np.fabs(np.random.normal())) + self.bias


class F18(CECCompositeBenchmark):
    """F18 class implements the Rotated Hybrid Composition 2 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}{w_i \\ast [f_i'((\mathbf{x}-\mathbf{o_i})/ \\lambda_i \\ast \mathbf{M_i}) + bias_i]} + f_{bias}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 10 \mid \mathbf{x^*} = \mathbf{o_1}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F18",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M2", "M10", "M30", "M50"),
        bias: Optional[int] = 10,
        dims: Optional[int] = 100,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            bias: Composite function bias.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        # Defines `sigma` and `lambda` parameters
        sigma = (1, 2, 1.5, 1.5, 1, 1, 1.5, 1.5, 2, 2)
        l = (5 / 16, 5 / 32, 2, 1, 1 / 10, 1 / 20, 20, 10, 1 / 6, 5 / 60)

        # Defines the composite functions
        functions = (
            n_dim.Ackley1(),
            n_dim.Ackley1(),
            n_dim.Rastrigin(),
            n_dim.Rastrigin(),
            n_dim.Sphere(),
            n_dim.Sphere(),
            n_dim.Weierstrass(),
            n_dim.Weierstrass(),
            n_dim.Griewank(),
            n_dim.Griewank(),
        )

        super(F18, self).__init__(
            name,
            year,
            auxiliary_data,
            sigma,
            l,
            functions,
            bias,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )


class F19(CECCompositeBenchmark):
    """F19 class implements the Rotated Hybrid Composition 2 with Narrow Basin Global Optimum benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}{w_i \\ast [f_i'((\mathbf{x}-\mathbf{o_i})/ \\lambda_i \\ast \mathbf{M_i}) + bias_i]} + f_{bias}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 10 \mid \mathbf{x^*} = \mathbf{o_1}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F19",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M2", "M10", "M30", "M50"),
        bias: Optional[int] = 10,
        dims: Optional[int] = 100,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            bias: Composite function bias.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        # Defines `sigma` and `lambda` parameters
        sigma = (0.1, 2, 1.5, 1.5, 1, 1, 1.5, 1.5, 2, 2)
        l = (0.1 * 5 / 32, 5 / 32, 2, 1, 1 / 10, 1 / 20, 20, 10, 1 / 6, 5 / 60)

        # Defines the composite functions
        functions = (
            n_dim.Ackley1(),
            n_dim.Ackley1(),
            n_dim.Rastrigin(),
            n_dim.Rastrigin(),
            n_dim.Sphere(),
            n_dim.Sphere(),
            n_dim.Weierstrass(),
            n_dim.Weierstrass(),
            n_dim.Griewank(),
            n_dim.Griewank(),
        )

        super(F19, self).__init__(
            name,
            year,
            auxiliary_data,
            sigma,
            l,
            functions,
            bias,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )


class F20(CECCompositeBenchmark):
    """F20 class implements the Rotated Hybrid Composition 2 with Global Optimum on the Bounds benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}{w_i \\ast [f_i'((\mathbf{x}-\mathbf{o_i})/ \\lambda_i \\ast \mathbf{M_i}) + bias_i]} + f_{bias}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 10 \mid \mathbf{x^*} = \mathbf{o_1}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F20",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M2", "M10", "M30", "M50"),
        bias: Optional[int] = 10,
        dims: Optional[int] = 100,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            bias: Composite function bias.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        # Defines `sigma` and `lambda` parameters
        sigma = (0.1, 2, 1.5, 1.5, 1, 1, 1.5, 1.5, 2, 2)
        l = (0.1 * 5 / 32, 5 / 32, 2, 1, 1 / 10, 1 / 20, 20, 10, 1 / 6, 5 / 60)

        # Defines the composite functions
        functions = (
            n_dim.Ackley1(),
            n_dim.Ackley1(),
            n_dim.Rastrigin(),
            n_dim.Rastrigin(),
            n_dim.Sphere(),
            n_dim.Sphere(),
            n_dim.Weierstrass(),
            n_dim.Weierstrass(),
            n_dim.Griewank(),
            n_dim.Griewank(),
        )

        super(F20, self).__init__(
            name,
            year,
            auxiliary_data,
            sigma,
            l,
            functions,
            bias,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )

    @d.check_exact_dimension_and_auxiliary_matrix
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines some constants used throughout the method
        D = x.shape[0]
        n_composition = len(self.f)
        y = 5 * np.ones(x.shape[0])

        # Defines the array of `w`, fitness and maximum fitness
        w = np.zeros(n_composition)
        f_max = np.zeros(n_composition)
        fit = np.zeros(n_composition)

        # Iterates through half of available dimensions
        for j in range(int(D / 2)):
            # Re-arranges the values in `o`
            self.o[0][2 * j + 1] = 5

        # Iterates through every possible composition function
        for i, f in enumerate(self.f):
            # Re-calculates the solution
            z = x - self.o[i][:D]

            # Calculates the `w`
            w[i] = np.exp(-np.sum(z**2) / (2 * D * self.sigma[i] ** 2))

            # Calculates the start and end indexes of the shift matrix
            start, end = i * x.shape[0], (i + 1) * x.shape[0]

            # Calculates the maximum fitness
            f_max[i] = f(np.matmul(y / self.l[i], self.M[start:end]))

            # Calculates the fitness
            fit[i] = self.C * f(np.matmul(z / self.l[i], self.M[start:end])) / f_max[i]

        # Calculates the sum of `w` and the maximum `w`
        w_sum = np.sum(w)
        w_max = np.max(w)

        # Iterates through the number of composition functions
        for i in range(n_composition):
            # If current `w` is different than `w_max`
            if w[i] != w_max:
                # Re-scales its value
                w[i] *= 1 - w_max**10

            # Normalizes `w`
            w[i] /= w_sum

        # Calculates the final fitness
        f = np.sum(np.matmul(w, (fit + self.f_bias)))

        return f + self.bias


class F21(CECCompositeBenchmark):
    """F21 class implements the Rotated Hybrid Composition 3 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}{w_i \\ast [f_i'((\mathbf{x}-\mathbf{o_i})/ \\lambda_i \\ast \mathbf{M_i}) + bias_i]} + f_{bias}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 360 \mid \mathbf{x^*} = \mathbf{o_1}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F21",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M2", "M10", "M30", "M50"),
        bias: Optional[int] = 360,
        dims: Optional[int] = 100,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            bias: Composite function bias.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        # Defines `sigma` and `lambda` parameters
        sigma = (1, 1, 1, 1, 1, 2, 2, 2, 2, 2)
        l = (1 / 4, 5 / 100, 5, 1, 5, 1, 50, 10, 1 / 8, 5 / 200)

        # Defines the composite functions
        functions = (
            n_dim.RotatedExpandedScafferF6(),
            n_dim.RotatedExpandedScafferF6(),
            n_dim.Rastrigin(),
            n_dim.Rastrigin(),
            n_dim.F8F2(),
            n_dim.F8F2(),
            n_dim.Weierstrass(),
            n_dim.Weierstrass(),
            n_dim.Griewank(),
            n_dim.Griewank(),
        )

        super(F21, self).__init__(
            name,
            year,
            auxiliary_data,
            sigma,
            l,
            functions,
            bias,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )


class F22(CECCompositeBenchmark):
    """F22 class implements the Rotated Hybrid Composition 3 with High Condition Number Matrix benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}{w_i \\ast [f_i'((\mathbf{x}-\mathbf{o_i})/ \\lambda_i \\ast \mathbf{M_i}) + bias_i]} + f_{bias}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 360 \mid \mathbf{x^*} = \mathbf{o_1}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F22",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M2", "M10", "M30", "M50"),
        bias: Optional[int] = 360,
        dims: Optional[int] = 100,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            bias: Composite function bias.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        # Defines `sigma` and `lambda` parameters
        sigma = (1, 1, 1, 1, 1, 2, 2, 2, 2, 2)
        l = (1 / 4, 5 / 100, 5, 1, 5, 1, 50, 10, 1 / 8, 5 / 200)

        # Defines the composite functions
        functions = (
            n_dim.RotatedExpandedScafferF6(),
            n_dim.RotatedExpandedScafferF6(),
            n_dim.Rastrigin(),
            n_dim.Rastrigin(),
            n_dim.F8F2(),
            n_dim.F8F2(),
            n_dim.Weierstrass(),
            n_dim.Weierstrass(),
            n_dim.Griewank(),
            n_dim.Griewank(),
        )

        super(F22, self).__init__(
            name,
            year,
            auxiliary_data,
            sigma,
            l,
            functions,
            bias,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )


class F23(CECCompositeBenchmark):
    """F23 class implements the Non-Continuous Rotated Hybrid Composition 3 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}{w_i \\ast [f_i'((\mathbf{x}-\mathbf{o_i})/ \\lambda_i \\ast \mathbf{M_i}) + bias_i]} + f_{bias}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) \\approx 360 \mid \mathbf{x^*} = \mathbf{o_1}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F23",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M2", "M10", "M30", "M50"),
        bias: Optional[int] = 360,
        dims: Optional[int] = 100,
        continuous: Optional[bool] = False,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = False,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            bias: Composite function bias.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        # Defines `sigma` and `lambda` parameters
        sigma = (1, 1, 1, 1, 1, 2, 2, 2, 2, 2)
        l = (1 / 4, 5 / 100, 5, 1, 5, 1, 50, 10, 1 / 8, 5 / 200)

        # Defines the composite functions
        functions = (
            n_dim.RotatedExpandedScafferF6(),
            n_dim.RotatedExpandedScafferF6(),
            n_dim.Rastrigin(),
            n_dim.Rastrigin(),
            n_dim.F8F2(),
            n_dim.F8F2(),
            n_dim.Weierstrass(),
            n_dim.Weierstrass(),
            n_dim.Griewank(),
            n_dim.Griewank(),
        )

        super(F23, self).__init__(
            name,
            year,
            auxiliary_data,
            sigma,
            l,
            functions,
            bias,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )

    @d.check_exact_dimension_and_auxiliary_matrix
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines some constants used throughout the method
        D = x.shape[0]
        n_composition = len(self.f)
        y = 5 * np.ones(x.shape[0])

        # Defines the array of `w`, fitness and maximum fitness
        w = np.zeros(n_composition)
        f_max = np.zeros(n_composition)
        fit = np.zeros(n_composition)

        # Creates the discontinuity
        x = np.where(np.fabs(x - self.o[0][:D]) < 0.5, x, np.round(2 * x) / 2)

        # Iterates through every possible composition function
        for i, f in enumerate(self.f):
            # Re-calculates the solution
            z = x - self.o[i][:D]

            # Calculates the `w`
            w[i] = np.exp(-np.sum(z**2) / (2 * D * self.sigma[i] ** 2))

            # Calculates the start and end indexes of the shift matrix
            start, end = i * x.shape[0], (i + 1) * x.shape[0]

            # Calculates the maximum fitness
            f_max[i] = f(np.matmul(y / self.l[i], self.M[start:end]))

            # Calculates the fitness
            fit[i] = self.C * f(np.matmul(z / self.l[i], self.M[start:end])) / f_max[i]

        # Calculates the sum of `w` and the maximum `w`
        w_sum = np.sum(w)
        w_max = np.max(w)

        # Iterates through the number of composition functions
        for i in range(n_composition):
            # If current `w` is different than `w_max`
            if w[i] != w_max:
                # Re-scales its value
                w[i] *= 1 - w_max**10

            # Normalizes `w`
            w[i] /= w_sum

        # Calculates the final fitness
        f = np.sum(np.matmul(w, (fit + self.f_bias)))

        return f + self.bias


class F24(CECCompositeBenchmark):
    """F24 class implements the Rotated Hybrid Composition 4 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}{w_i \\ast [f_i'((\mathbf{x}-\mathbf{o_i})/ \\lambda_i \\ast \mathbf{M_i}) + bias_i]} + f_{bias}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 260 \mid \mathbf{x^*} = \mathbf{o_1}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F24",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M2", "M10", "M30", "M50"),
        bias: Optional[int] = 260,
        dims: Optional[int] = 100,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            bias: Composite function bias.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        # Defines `sigma` and `lambda` parameters
        sigma = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
        l = (10, 1 / 4, 1, 5 / 32, 1, 5 / 100, 1 / 10, 1, 5 / 100, 5 / 100)

        # Defines the composite functions
        functions = (
            n_dim.Weierstrass(),
            n_dim.RotatedExpandedScafferF6(),
            n_dim.F8F2(),
            n_dim.Ackley1(),
            n_dim.Rastrigin(),
            n_dim.Griewank(),
            n_dim.NonContinuousExpandedScafferF6(),
            n_dim.NonContinuousRastrigin(),
            n_dim.HighConditionedElliptic(),
            n_dim.SphereWithNoise(),
        )

        super(F24, self).__init__(
            name,
            year,
            auxiliary_data,
            sigma,
            l,
            functions,
            bias,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )


class F25(CECCompositeBenchmark):
    """F25 class implements the Rotated Hybrid Composition 4 without Bounds benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}{w_i \\ast [f_i'((\mathbf{x}-\mathbf{o_i})/ \\lambda_i \\ast \mathbf{M_i}) + bias_i]} + f_{bias}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [?, ?] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 260 \mid \mathbf{x^*} = \mathbf{o_1}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F25",
        year: Optional[str] = "2005",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M2", "M10", "M30", "M50"),
        bias: Optional[int] = 260,
        dims: Optional[int] = 100,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            bias: Composite function bias.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        # Defines `sigma` and `lambda` parameters
        sigma = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
        l = (10, 1 / 4, 1, 5 / 32, 1, 5 / 100, 1 / 10, 1, 5 / 100, 5 / 100)

        # Defines the composite functions
        functions = (
            n_dim.Weierstrass(),
            n_dim.RotatedExpandedScafferF6(),
            n_dim.F8F2(),
            n_dim.Ackley1(),
            n_dim.Rastrigin(),
            n_dim.Griewank(),
            n_dim.NonContinuousExpandedScafferF6(),
            n_dim.NonContinuousRastrigin(),
            n_dim.HighConditionedElliptic(),
            n_dim.SphereWithNoise(),
        )

        super(F25, self).__init__(
            name,
            year,
            auxiliary_data,
            sigma,
            l,
            functions,
            bias,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )
