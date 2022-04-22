"""CEC2008 benchmarking functions.
"""

from typing import Optional, Tuple

import numpy as np

import opytimark.utils.decorator as d
from opytimark.core import CECBenchmark

# Fixes Numpy's random seed
np.random.seed(0)


class F1(CECBenchmark):
    """F1 class implements the Shifted Sphere's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} z_i^2 - 450 \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -450 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F1",
        year: Optional[str] = "2008",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o"),
        dims: Optional[int] = 1000,
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
    """F2 class implements the Shifted Schwefel's 2.21 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \max_{i=1, \ldots, n}|z_i| - 450 \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -450 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F2",
        year: Optional[str] = "2008",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o"),
        dims: Optional[int] = 1000,
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

        # Calculating the Schwefel's 2.21 function
        f = np.fabs(z)

        return np.amax(f) - 450


class F3(CECBenchmark):
    """F3 class implements the Shifted Rosenbrock's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-1} (100(z_i^2-z_{i+1})^2 + (z_i - 1)^2) + 390 \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -390 \mid \mathbf{x^*} = \mathbf{o} + 1`.

    """

    def __init__(
        self,
        name: Optional[str] = "F3",
        year: Optional[str] = "2008",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o"),
        dims: Optional[int] = 1000,
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


class F4(CECBenchmark):
    """F4 class implements the Shifted Rastrigin's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (z_i^2 - 10cos(2 \\pi z_i) + 10) - 330 \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -330 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F4",
        year: Optional[str] = "2008",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o"),
        dims: Optional[int] = 1000,
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

        # Calculating the Shifted Rastrigin's function
        f = z**2 - 10 * np.cos(2 * np.pi * z) + 10

        return np.sum(f) - 330


class F5(CECBenchmark):
    """F5 class implements the Shifted Griewank's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = 1 + \sum_{i=1}^{n}\\frac{x_i^2}{4000} - \prod cos(\\frac{x_i}{\sqrt{i}}) - 180 \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-600, 600] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -180 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F5",
        year: Optional[str] = "2008",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o"),
        dims: Optional[int] = 1000,
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

        # Re-calculates the input
        z = x - self.o[: x.shape[0]]

        # Initializing terms
        term1, term2 = 0, 1

        # For every possible dimension of `x`
        for i in range(x.shape[0]):
            # Calculating first term
            term1 += (z[i] ** 2) / 4000

            # Calculating second term
            term2 *= np.cos(z[i] / np.sqrt(i + 1))

        # Calculating the Shifted Griewank's function
        f = 1 + term1 - term2

        return f - 180


class F6(CECBenchmark):
    """F6 class implements the Shifted Ackley's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = -20e^{-0.2\sqrt{\\frac{1}{n}\sum_{i=1}^{n}x_i^2}}-e^{\\frac{1}{n}\sum_{i=1}^{n}cos(2 \\pi x_i)}+ 20 + e - 140 \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-32, 32] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -140 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F6",
        year: Optional[str] = "2008",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o"),
        dims: Optional[int] = 1000,
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

        # Calculating the 1 / n term
        inv = 1 / x.shape[0]

        # Calculating first term
        term1 = -0.2 * np.sqrt(inv * np.sum(z**2))

        # Calculating second term
        term2 = inv * np.sum(np.cos(2 * np.pi * z))

        # Calculating Shifted Ackley's function
        f = 20 + np.e - np.exp(term2) - 20 * np.exp(term1)

        return np.sum(f) - 140


class F7(CECBenchmark):
    """F7 class implements the Fast Fractal Double Dip's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} fractal1D(x_i + twist(x_{(i mod n) + 1}))

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-1, 1] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 1, \ldots, 1)`.

    """

    def __init__(
        self,
        name: Optional[str] = "F7",
        year: Optional[str] = "2008",
        auxiliary_data: Optional[Tuple[str, ...]] = (),
        dims: Optional[int] = 1000,
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

    def _double_dip(self, x: float, c: float, s: float) -> float:
        """Calculates the Double Dip's function.

        Args:
            x: Float-valued point.
            c: Float-valued point.
            s: Float-valued point.

        Returns:
            (float): The value over the function.

        """

        # If `x` is between -0.5 and 0.5
        if -0.5 < x < 0.5:
            # Calculates the Double Dip's function.
            return (
                -6144 * (x - c) ** 6 + 3088 * (x - c) ** 4 - 392 * (x - c) ** 2 + 1
            ) * s

        return 0

    def _twist(self, y: float) -> float:
        """Twists the function.

        Args:
            y: Float-valued point.

        Returns:
            (float): The value over the twisted function.

        """

        return 4 * (y**4 - 2 * y**3 + y**2)

    def _fractal_1d(self, x: float) -> float:
        """Calculates the 1-dimensional fractal function.

        Args:
            x: Float-valued point.

        Returns:
            (float): The value over the function.

        """

        # Instantiates the function
        f = 0

        # Iterates through 1 to 3
        for k in range(1, 4):
            # Calculates the upper limit
            upper_limit = 2 ** (k - 1)

            # Iterates through 1 to `upper_limit`
            for _ in range(1, upper_limit):
                # Makes a random choice between an integer {0, 1, 2}
                r2 = np.random.choice([0, 1, 2])

                # Iterates through 1 to `r2`
                for _ in range(1, r2):
                    # Calculates a uniform random number between 0 and 1
                    r1 = np.random.uniform()

                    # Calculates the Double Dip's function
                    f += self._double_dip(x, r1, 1 / (2 ** (k - 1) * (2 - r1)))

        return f

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Instantiates the function
        f = 0

        # Iterates through every dimension
        for i in range(x.shape[0]):
            # Calculates the twist output over `x[i mod D]`
            t = self._twist(x[i % x.shape[0]])

            # Calculates the one-dimensional fractal function.
            f += self._fractal_1d(x[i] + t)

        return f
