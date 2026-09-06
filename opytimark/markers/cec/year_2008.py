# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from typing import Any

import numpy as np
from numpy.typing import NDArray

import opytimark.utils.decorator as d
from opytimark.core import CECBenchmark
from opytimark.typing import BenchmarkValue


class F1(CECBenchmark):
    r"""Shifted Sphere's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} z_i^2 - 450 \mid z_i = x_i - o_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -450 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F1", 1000, True, True, True, False, True)
    _year = "2008"
    _auxiliary_data = ("o",)

    @d.check_less_equal_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F1 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        z = x - self.o[: x.shape[0]]

        f = z**2

        return np.sum(f) - 450


class F2(CECBenchmark):
    r"""Shifted Schwefel's 2.21 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \max_{i=1, \ldots, n}|z_i| - 450 \mid z_i = x_i - o_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -450 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F2", 1000, True, True, True, False, False)
    _year = "2008"
    _auxiliary_data = ("o",)

    @d.check_less_equal_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F2 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        z = x - self.o[: x.shape[0]]

        f = np.fabs(z)

        return np.amax(f) - 450


class F3(CECBenchmark):
    r"""Shifted Rosenbrock's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-1} (100(z_i^2-z_{i+1})^2 + (z_i - 1)^2) + 390 \mid z_i = x_i - o_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -390 \mid \mathbf{x^*} = \mathbf{o} + 1`.

    """

    _defaults = ("F3", 1000, True, True, True, True, False)
    _year = "2008"
    _auxiliary_data = ("o",)

    @d.check_less_equal_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F3 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        z = x - self.o[: x.shape[0]]

        f = 0

        for i in range(x.shape[0] - 1):
            f += 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2

        return f + 390


class F4(CECBenchmark):
    r"""Shifted Rastrigin's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (z_i^2 - 10cos(2 \pi z_i) + 10) - 330 \mid z_i = x_i - o_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -330 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F4", 1000, True, True, True, True, True)
    _year = "2008"
    _auxiliary_data = ("o",)

    @d.check_less_equal_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F4 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        z = x - self.o[: x.shape[0]]

        f = z**2 - 10 * np.cos(2 * np.pi * z) + 10

        return np.sum(f) - 330


class F5(CECBenchmark):
    r"""Shifted Griewank's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = 1 + \sum_{i=1}^{n}\frac{x_i^2}{4000} - \prod cos(\frac{x_i}{\sqrt{i}}) - 180 \mid z_i = x_i - o_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-600, 600] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -180 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F5", 1000, True, True, True, True, False)
    _year = "2008"
    _auxiliary_data = ("o",)

    @d.check_less_equal_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F5 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        z = x - self.o[: x.shape[0]]

        term1, term2 = 0, 1

        for i in range(x.shape[0]):
            term1 += (z[i] ** 2) / 4000

            term2 *= np.cos(z[i] / np.sqrt(i + 1))

        f = 1 + term1 - term2

        return f - 180


class F6(CECBenchmark):
    r"""Shifted Ackley's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = -20e^{-0.2\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2}}-e^{\frac{1}{n}\sum_{i=1}^{n}cos(2 \pi x_i)}+ 20 + e - 140 \mid z_i = x_i - o_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-32, 32] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -140 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F6", 1000, True, True, True, True, False)
    _year = "2008"
    _auxiliary_data = ("o",)

    @d.check_less_equal_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F6 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        z = x - self.o[: x.shape[0]]

        inv = 1 / x.shape[0]

        term1 = -0.2 * np.sqrt(inv * np.sum(z**2))

        term2 = inv * np.sum(np.cos(2 * np.pi * z))

        f = 20 + np.e - np.exp(term2) - 20 * np.exp(term1)

        return np.sum(f) - 140


class F7(CECBenchmark):
    r"""Fast Fractal Double Dip's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} fractal1D(x_i + twist(x_{(i mod n) + 1}))

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-1, 1] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 1, \ldots, 1)`.

    """

    _defaults = ("F7", 1000, True, True, True, True, False)
    _year = "2008"
    _auxiliary_data = ()

    def _double_dip(self, x: float, c: float, s: float) -> float:

        if -0.5 < x < 0.5:
            return (-6144 * (x - c) ** 6 + 3088 * (x - c) ** 4 - 392 * (x - c) ** 2 + 1) * s

        return 0

    def _twist(self, y: float) -> float:

        return 4 * (y**4 - 2 * y**3 + y**2)

    def _fractal_1d(self, x: float) -> float:

        f = 0

        for k in range(1, 4):
            upper_limit = 2 ** (k - 1)

            for _ in range(1, upper_limit):
                r2 = np.random.choice([0, 1, 2])

                for _ in range(1, r2):
                    r1 = np.random.uniform()

                    f += self._double_dip(x, r1, 1 / (2 ** (k - 1) * (2 - r1)))

        return f

    @d.check_less_equal_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F7 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(x.shape[0]):
            t = self._twist(x[i % x.shape[0]])

            f += self._fractal_1d(x[i] + t)

        return f
