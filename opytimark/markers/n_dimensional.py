# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from typing import Any

import numpy as np
from numpy.typing import NDArray

import opytimark.utils.constants as c
import opytimark.utils.decorator as d
from opytimark.core import Benchmark
from opytimark.typing import BenchmarkValue


def _elliptic(x: NDArray[Any]) -> BenchmarkValue:
    coefficients = 1e6 ** np.linspace(0, 1, x.shape[0])
    return np.sum(coefficients * x**2)


class Ackley1(Benchmark):
    r"""Ackley's 1st benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = -20e^{-0.2\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2}}-e^{\frac{1}{n}\sum_{i=1}^{n}cos(2 \pi x_i)}+ 20 + e

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-32, -32] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("Ackley1", -1, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Ackley1 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        inv = 1 / x.shape[0]

        term1 = -0.2 * np.sqrt(inv * np.sum(x**2))

        term2 = inv * np.sum(np.cos(2 * np.pi * x))

        f = 20 + np.e - np.exp(term2) - 20 * np.exp(term1)

        return np.sum(f)


class Ackley4(Benchmark):
    r"""Ackley's 4th benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-1}(e^{-0.2}\sqrt{x_i^2+x_{i+1}^2}+3(cos(2x_i)+sin(2x_{i+1})))

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-35, -35] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = −4.590101633799122 \mid \mathbf{x^*} = (-1.51, -0.755)`.

    """

    _defaults = ("Ackley4", -1, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Ackley4 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(x.shape[0] - 1):
            f += np.exp(-0.2) * np.sqrt(x[i] ** 2 + x[i + 1] ** 2) + 3 * (np.cos(2 * x[i]) + np.sin(2 * x[i + 1]))

        return f


class Alpine1(Benchmark):
    r"""Alpine's 1st benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}|x_i sin(x_i)+0.1x_i|

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("Alpine1", -1, True, False, False, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Alpine1 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = np.fabs(x * np.sin(x) + 0.1 * x)

        return np.sum(f)


class Alpine2(Benchmark):
    r"""Alpine's 2nd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n}\sqrt{x_i}sin(x_i)

        Domain:
            The function is commonly evaluated using :math:`x_i \in [0, 10] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 2.808^n \mid \mathbf{x^*} = (7.917, 7.917, \ldots, 7.917)`.

    """

    _defaults = ("Alpine2", -1, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Alpine2 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = np.sqrt(x) * np.sin(x)

        return -np.prod(f)


class Brown(Benchmark):
    r"""Brown's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-1}(x_i^2)^{(x_{i+1}^{2}+1)}+(x_{i+1}^2)^{(x_{i}^{2}+1)}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-1, 4] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("Brown", -1, True, True, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Brown benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        term1 = x[:-1] ** 2

        term2 = x[1:] ** 2

        f = np.sum(term1 ** (term2 + 1) + term2 ** (term1 + 1))

        return f


class ChungReynolds(Benchmark):
    r"""Chung Reynolds' benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = (\sum_{i=1}^{n} x_i^2)^2

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("ChungReynolds", -1, True, True, True, False, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the ChungReynolds benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = np.sum(x**2) ** 2

        return f


class CosineMixture(Benchmark):
    r"""Cosine Mixture's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = -0.1\sum_{i=1}^{n}cos(5 \pi x_i) - \sum_{i=1}^{n}x_i^2

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-1, 1] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -0.1n \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("CosineMixture", -1, False, False, False, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the CosineMixture benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        term1 = np.sum(np.cos(5 * np.pi * x))

        term2 = np.sum(x**2)

        f = -0.1 * term1 - term2

        return f


class Csendes(Benchmark):
    r"""Csendes' benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}x_i^6(2 + sin(\frac{1}{x_i}))

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-1, 1] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("Csendes", -1, True, True, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Csendes benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (x**6) * (2 + np.sin(1 / (x + c.EPSILON)))

        return np.sum(f)


class Deb1(Benchmark):
    r"""Deb's 1st benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = -\frac{1}{n}\sum_{i=1}^{n}sin^6(5 \pi x_i)

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-1, 1] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -1 \mid \mathbf{x^*} = (-0.9, -0.7, \ldots, 0.9)`.

    """

    _defaults = ("Deb1", -1, True, True, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Deb1 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        term = np.sum(np.sin(5 * np.pi * x) ** 6)

        f = -1 / x.shape[0] * term

        return f


class Deb3(Benchmark):
    r"""Deb's 3rd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = -\frac{1}{n}\sum_{i=1}^{n}sin^6(5 \pi (x_i^{\frac{3}{4}}-0.05))

        Domain:
            The function is commonly evaluated using :math:`x_i \in [0, 1] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = ? \mid \mathbf{x^*} = (?, ?, \ldots, ?)`.

    """

    _defaults = ("Deb3", -1, True, True, True, False, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Deb3 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        term = np.sum(np.sin(5 * np.pi * (x ** (3 / 4) - 0.05)) ** 6)

        f = -1 / x.shape[0] * term

        return f


class DixonPrice(Benchmark):
    r"""Dixon & Price's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = (x_1 - 1)^2 + \sum_{i=2}^{n}i(2x_i^2 - x_{i-1})^2

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid x_i^* = 2^{-\frac{2^i-2}{2^i}}`.

    """

    _defaults = ("DixonPrice", -1, True, True, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the DixonPrice benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        term1 = (x[0] - 1) ** 2

        term2 = 0

        for i in range(1, x.shape[0]):
            term2 += (i + 1) * ((2 * (x[i] ** 2) - x[i - 1]) ** 2)

        f = term1 + term2

        return f


class Exponential(Benchmark):
    r"""Exponential's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = e^{-0.5\sum_{i=1}^n{x_i^2}}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-1, 1] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("Exponential", -1, True, True, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Exponential benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = np.exp(-0.5 * np.sum(x**2))

        return f


class F8F2(Benchmark):
    r"""Shifted Expanded Griewank's plus Rosenbrock's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) =  f(x_1, x_2) + f(x_2, x_3) + \ldots + f(x_n, f_1)

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 1, \ldots, 1)`.

    """

    _defaults = ("F8F2", -1, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F8F2 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        def _griewank(x):
            return x**2 / 4000 - np.cos(x / np.sqrt(1)) + 1

        def _rosenbrock(x, y):
            return 100 * (x**2 - y) ** 2 + (x - 1) ** 2

        f = 0

        for i in range(x.shape[0]):
            if i == (x.shape[0] - 1):
                f += _griewank(_rosenbrock(x[i], x[0]))

            else:
                f += _griewank(_rosenbrock(x[i], x[i + 1]))

        return f


class Griewank(Benchmark):
    r"""Griewank's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = 1 + \sum_{i=1}^{n}\frac{x_i^2}{4000} - \prod cos(\frac{x_i}{\sqrt{i}})

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("Griewank", -1, True, False, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Griewank benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        term1, term2 = 0, 1

        for i in range(x.shape[0]):
            term1 += (x[i] ** 2) / 4000

            term2 *= np.cos(x[i] / np.sqrt(i + 1))

        f = 1 + term1 - term2

        return f


class HappyCat(Benchmark):
    r"""HappyCat's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = [(||\mathbf{x}||_2 - n)^2]^{\alpha} + \frac{1}{n}(\frac{1}{2}||\mathbf{x}||_2 + \sum_{i=1}^{n}x_i) + \frac{1}{2}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-2, 2] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (-1, -1, \ldots, -1)`.

    """

    _defaults = ("HappyCat", -1, True, False, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the HappyCat benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        n = x.shape[0]

        square = np.sum(x**2)

        f = ((square - n) ** 2) ** (1 / 8) + (1 / n) * (1 / 2 * square + np.sum(x)) + 1 / 2

        return f


class HighConditionedElliptic(Benchmark):
    r"""High Conditioned Elliptic's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (10^6)^\frac{i-1}{n-1} x_i^2

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

        For one input variable, the coefficient is one.

    """

    _defaults = ("HighConditionedElliptic", -1, True, True, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the HighConditionedElliptic benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        return _elliptic(x)


class Levy(Benchmark):
    r"""Levy's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = sin^2(\pi w_1) + \sum_{i=1}^{n-1}(w_i-1)^2 [1+10sin^2(\pi w_i + 1)]
        .. math:: + (w_n - 1)^2 [1 + sin^2(2 \pi w_n)] \mid w_i = 1 + \frac{x_i - 1}{4}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 1, \ldots, 1)`.

    """

    _defaults = ("Levy", -1, True, True, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Levy benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        w = 1 + (x - 1) / 4

        term1 = np.sin(np.pi * w[0]) ** 2

        term3 = ((w[-1] - 1) ** 2) * (1 + (np.sin(2 * np.pi * w[-1]) ** 2))

        w = w[0 : x.shape[0] - 1]

        term2 = np.sum(((w - 1) ** 2) * (1 + 10 * (np.sin(np.pi * w + 1) ** 2)))

        f = term1 + term2 + term3

        return f


class Michalewicz(Benchmark):
    r"""Michalewicz's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = - \sum_{i=1}^{n}sin(x_i)sin^{20}(\frac{ix_i^2}{\pi})

        Domain:
            The function is commonly evaluated using :math:`x_i \in [0, \pi] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = ? \mid \mathbf{x^*} = (?, ?, \ldots, ?)`.

    """

    _defaults = ("Michalewicz", -1, True, True, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Michalewicz benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(x.shape[0]):
            f += np.sin(x[i]) * (np.sin((i + 1) * x[i] ** 2 / np.pi) ** 20)

        return -f


class NonContinuousExpandedScafferF6(Benchmark):
    r"""Non-Continuous Expanded Scaffer's F6 benchmark.

    Notes:
        .. math:: f(\mathbf{y}) = f(y_1, y_2, \ldots, y_n) =  f(y_1, y_2) + f(y_2, y_3) + \ldots + f(y_n, y_1) \mid y_i = round(2x_i)/2, |x_i| >= 0.5

        Domain:
            The function is commonly evaluated using :math:`y_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{y^*}) = 0 \mid \mathbf{y^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("NonContinuousExpandedScafferF6", -1, False, False, False, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the NonContinuousExpandedScafferF6 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        def _scaffer(x, y):
            return 0.5 + (np.sin(np.sqrt(x**2 + y**2)) ** 2 - 0.5) / ((1 + 0.0001 * (x**2 + y**2)) ** 2)

        x = np.where(np.fabs(x) < 0.5, x, np.round(2 * x) / 2)

        f = 0

        for i in range(x.shape[0]):
            if i == (x.shape[0] - 1):
                f += _scaffer(x[i], x[0])

            else:
                f += _scaffer(x[i], x[i + 1])

        return f


class NonContinuousRastrigin(Benchmark):
    r"""Non-Continuous Rastrigin's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(y_1, y_2, \ldots, y_n) = 10n + \sum_{i=1}^{n}(y_i^2 - 10cos(2 \pi y_i)) \mid y_i = round(2x_i)/2, |x_i| >= 0.5

        Domain:
            The function is commonly evaluated using :math:`y_i \in [-5.12, 5.12] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{y^*}) = 0 \mid \mathbf{y^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("NonContinuousRastrigin", -1, False, True, False, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the NonContinuousRastrigin benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        x = np.where(np.fabs(x) < 0.5, x, np.round(2 * x) / 2)

        f = x**2 - 10 * np.cos(2 * np.pi * x)

        return 10 * x.shape[0] + np.sum(f)


class Pathological(Benchmark):
    r"""Pathological's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-1}0.5 + \frac{sin^2(\sqrt{100x_i^2+x_{i+1}^2})-0.5}{1 + 0.001(x_i^2 - 2x_i x_{i+1} + x_{i+1}^2)^2}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("Pathological", -1, True, True, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Pathological benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(x.shape[0] - 1):
            f += 0.5 + (np.sin(np.sqrt(100 * x[i] ** 2 + x[i + 1] ** 2)) ** 2 - 0.5) / (
                1 + 0.001 * ((x[i] ** 2 - 2 * x[i] * x[i + 1] + x[i + 1] ** 2) ** 2)
            )

        return f


class Periodic(Benchmark):
    r"""Periodic's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = 1 + \sum_{i=1}^{n}sin^2(x_i) - 0.1e^{\sum_{i=1}^{n}x_i^2}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0.9 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("Periodic", -1, True, False, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Periodic benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 1 + np.sum(np.sin(x) ** 2) - 0.1 * np.exp(np.sum(x))

        return f


class Perm0DBeta(Benchmark):
    r"""Perm 0, D, Beta's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}(\sum_{j=1}^{n} (j + 10)(x_j^i - \frac{1}{j^i}))^2

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-n, n] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, \frac{1}{2}, \ldots, \frac{1}{n})`.

    """

    _defaults = ("Perm0DBeta", -1, True, True, True, False, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Perm0DBeta benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                f += ((j + 1 + 10) * (x[j] ** (i + 1) - (1 / (j + 1) ** (i + 1)))) ** 2

        return f


class PermDBeta(Benchmark):
    r"""Perm D, Beta's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}(\sum_{j=1}^{n} (j^i + 10)((\frac{x_j}{j})^i - 1))^2

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-n, n] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 2, \ldots, n)`.

    """

    _defaults = ("PermDBeta", -1, True, True, True, False, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the PermDBeta benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                f += (((j + 1) ** (i + 1) + 10) * ((x[j] / (j + 1)) ** (i + 1) - 1)) ** 2

        return f


class PowellSingular2(Benchmark):
    r"""Powell's Singular 2nd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-2}(x_{i-1}+10x_i)^2 + 5(x_{i+1} - x_{i+2})^2 + (x_i - 2x_{i+1})^4 + 10(x_{i-1} - x_{i+2})^4

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-4, 5] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("PowellSingular2", -1, True, True, False, False, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the PowellSingular2 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(x.shape[0] - 2):
            f += (
                (x[i - 1] + 10 * x[i]) ** 2
                + 5 * (x[i + 1] - x[i + 2]) ** 2
                + (x[i] - 2 * x[i + 1]) ** 4
                + 10 * (x[i - 1] - x[i + 2]) ** 4
            )

        return f


class PowellSum(Benchmark):
    r"""Powell's Sum benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}|x_i|^{i+1}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-1, 1] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("PowellSum", -1, True, True, False, False, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the PowellSum benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(x.shape[0]):
            f += np.fabs(x[i]) ** (i + 2)

        return f


class Qing(Benchmark):
    r"""Qing's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}(x_i^2 - i)^2

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-500, 500] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid x_i^* = (\pm \sqrt{i}, \pm \sqrt{i}, \ldots, \pm \sqrt{i})`.

    """

    _defaults = ("Qing", -1, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Qing benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(x.shape[0]):
            f += (x[i] ** 2 - (i + 1)) ** 2

        return f


class Quartic(Benchmark):
    r"""Quartic's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}ix_i^4 + rand()

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-1.28, 1.28] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 + rand() \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("Quartic", -1, True, False, True, False, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Quartic benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(x.shape[0]):
            f += (i + 1) * (x[i] ** 4)

        return f + np.random.uniform()


class Quintic(Benchmark):
    r"""Quintic's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}|x_i^5 - 3x_i^4 + 4x_i^3 + 2x_i^2 - 10x_i - 4|

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (-1 or 2, -1 or 2, \ldots, -1 or 2)`.

    """

    _defaults = ("Quintic", -1, True, True, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Quintic benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = np.fabs(x**5 - 3 * x**4 + 4 * x**3 + 2 * x**2 - 10 * x - 4)

        return np.sum(f)


class Rana(Benchmark):
    r"""Rana's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-2}(x_{i+1} + 1)cos(t_2)sin(t_1) + x_i cos(t_1)sin(t_2)

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-500, 500] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("Rana", -1, True, True, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Rana benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(x.shape[0] - 2):
            t1 = np.sqrt(np.fabs(x[i + 1] + x[i] + 1))

            t2 = np.sqrt(np.fabs(x[i + 1] - x[i] + 1))

            f += (x[i + 1] + 1) * np.cos(t2) * np.sin(t1) + x[i] * np.cos(t1) * np.sin(t2)

        return f


class Rastrigin(Benchmark):
    r"""Rastrigin's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = 10n + \sum_{i=1}^{n}(x_i^2 - 10cos(2 \pi x_i))

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5.12, 5.12] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("Rastrigin", -1, True, True, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Rastrigin benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = x**2 - 10 * np.cos(2 * np.pi * x)

        return 10 * x.shape[0] + np.sum(f)


class Ridge(Benchmark):
    r"""Ridge's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = x_1 + (\sum_{i=2}^{n}x_i^2)^{0.5}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-\lambda, \lambda]^n \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -\lambda \mid \mathbf{x^*} = (-\lambda, 0, \ldots, 0)`.

    """

    _defaults = ("Ridge", -1, True, False, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Ridge benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = x[1:] ** 2

        return x[0] + np.sum(f) ** 0.5


class Rosenbrock(Benchmark):
    r"""Rosenbrock's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-1}[100(x_{i+1}-x_i^2)^2 + (x_i - 1)^2]

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-30, 30] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 1, \ldots, 1)`.

    """

    _defaults = ("Rosenbrock", -1, True, True, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Rosenbrock benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(x.shape[0] - 1):
            f += 100 * ((x[i + 1] - x[i] ** 2) ** 2) + ((x[i] - 1) ** 2)

        return f


class RotatedExpandedScafferF6(Benchmark):
    r"""Rotated Expanded Scaffer's F6 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) =  f(x_1, x_2) + f(x_2, x_3) + \ldots + f(x_n, x_1)

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("RotatedExpandedScafferF6", -1, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the RotatedExpandedScafferF6 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        def _scaffer(x, y):
            return 0.5 + (np.sin(np.sqrt(x**2 + y**2)) ** 2 - 0.5) / ((1 + 0.0001 * (x**2 + y**2)) ** 2)

        f = 0

        for i in range(x.shape[0]):
            if i == (x.shape[0] - 1):
                f += _scaffer(x[i], x[0])

            else:
                f += _scaffer(x[i], x[i + 1])

        return f


class RotatedHyperEllipsoid(Benchmark):
    r"""Rotated Hyper-Ellipsoid's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}\sum_{j=1}^{i}x_j^2

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-65.536, 65.536] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("RotatedHyperEllipsoid", -1, True, True, True, False, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the RotatedHyperEllipsoid benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(x.shape[0]):
            for j in range(i):
                f += x[j] ** 2

        return f


class Salomon(Benchmark):
    r"""Salomon's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = 1 - cos(2 \pi \sqrt{\sum_{i=1}^{n}x_i^2}) + 0.1\sqrt{\sum_{i=1}^{n}x_i^2}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("Salomon", -1, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Salomon benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 1 - np.cos(2 * np.pi * np.sqrt(np.sum(x**2))) + 0.1 * np.sqrt(np.sum(x**2))

        return f


class SchumerSteiglitz(Benchmark):
    r"""Schumer Steiglitz's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}x_i^4

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("SchumerSteiglitz", -1, True, True, True, False, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the SchumerSteiglitz benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = x**4

        return np.sum(f)


class Schwefel(Benchmark):
    r"""Schwefel's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = 418.9829n -\sum_{i=1}^{n} x_i sin(\sqrt{|x_i|})

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (420.9687, 420.9687, \ldots, 420.9687)`.

    """

    _defaults = ("Schwefel", -1, True, False, False, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Schwefel benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = x * np.sin(np.sqrt(np.fabs(x)))

        return 418.9829 * x.shape[0] - np.sum(f)


class Schwefel220(Benchmark):
    r"""Schwefel's 2.20 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}|x_i|

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("Schwefel220", -1, True, True, False, False, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Schwefel220 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = np.fabs(x)

        return np.sum(f)


class Schwefel221(Benchmark):
    r"""Schwefel's 2.21 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \max_{i=1, \ldots, n}|x_i|

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("Schwefel221", -1, True, True, False, False, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Schwefel221 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = np.fabs(x)

        return np.amax(f)


class Schwefel222(Benchmark):
    r"""Schwefel's 2.22 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}|x_i| + \prod_{i=1}^{n}|x_i|

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("Schwefel222", -1, True, True, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Schwefel222 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = np.fabs(x)

        return np.sum(f) + np.prod(f)


class Schwefel223(Benchmark):
    r"""Schwefel's 2.23 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}x_i^{10}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("Schwefel223", -1, True, True, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Schwefel223 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = x**10

        return np.sum(f)


class Schwefel225(Benchmark):
    r"""Schwefel's 2.25 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=2}^{n}(x_i - 1)^2 + (x_1 - x_i^2)^2

        Domain:
            The function is commonly evaluated using :math:`x_i \in [0, 10] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 1, \ldots, 1)`.

    """

    _defaults = ("Schwefel225", -1, True, True, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Schwefel225 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(1, x.shape[0]):
            f += (x[i] - 1) ** 2 + (x[0] - x[i] ** 2) ** 2

        return f


class Schwefel226(Benchmark):
    r"""Schwefel's 2.26 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = -\frac{1}{n} \sum_{i=1}^{n}x_i sin(\sqrt{|x_i|})

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-500, 500] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -418.983 \mid \mathbf{x^*} = (\pm[\pi (0.5+k)]^2, \pm[\pi (0.5+k)]^2, \ldots, \pm[\pi (0.5+k)]^2)`.

    """

    _defaults = ("Schwefel226", -1, True, True, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Schwefel226 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = x * np.sin(np.sqrt(np.fabs(x)))

        return -1 / x.shape[0] * np.sum(f)


class Shubert(Benchmark):
    r"""Shubert's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \prod_{i=1}^n \sum_{j=1}^{5}cos((j+1)x_i+j)

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -186.7309 \mid \mathbf{x^*} = \text{multiple solutions}`.

    """

    _defaults = ("Shubert", -1, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Shubert benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 1

        for i in range(x.shape[0]):
            for j in range(1, 6):
                f *= np.cos((j + 1) * x[i] + j)

        return f


class Shubert3(Benchmark):
    r"""Shubert's 3rd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^n \sum_{j=1}^{5}j sin((j+1)x_i+j)

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -29.6733337 \mid \mathbf{x^*} = (?, ?, \ldots, ?)`.

    """

    _defaults = ("Shubert3", -1, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Shubert3 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(x.shape[0]):
            for j in range(1, 6):
                f += j * np.sin((j + 1) * x[i] + j)

        return f


class Shubert4(Benchmark):
    r"""Shubert's 4th benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^n \sum_{j=1}^{5}j cos((j+1)x_i+j)

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -25.740858 \mid \mathbf{x^*} = (?, ?, \ldots, ?)`.

    """

    _defaults = ("Shubert4", -1, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Shubert4 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(x.shape[0]):
            for j in range(1, 6):
                f += j * np.cos((j + 1) * x[i] + j)

        return f


class SchafferF6(Benchmark):
    r"""Schaffer's F6 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-1}0.5 + \frac{sin^2(\sqrt{x_i^2+x_{i+1}^2})-0.5}{[1 + 0.001(x_i^2 + x_{i+1}^2)]^2}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("SchafferF6", -1, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the SchafferF6 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(x.shape[0] - 1):
            f += 0.5 + (np.sin(np.sqrt(x[i] ** 2 + x[i + 1] ** 2)) ** 2 - 0.5) / (
                (1 + 0.001 * (x[i] ** 2 + x[i + 1] ** 2)) ** 2
            )

        return f


class Sphere(Benchmark):
    r"""Sphere's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} x_i^2

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5.12, 5.12] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("Sphere", -1, True, True, True, False, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Sphere benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = x**2

        return np.sum(f)


class SphereWithNoise(Benchmark):
    r"""Sphere with Noise's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = (\sum_{i=1}^{n} x_i^2)(1 + 0.1|N(0,1)|)

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5.12, 5.12] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("SphereWithNoise", -1, True, True, True, False, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the SphereWithNoise benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = x**2

        noise = 1 + 0.1 * np.fabs(np.random.normal())

        return np.sum(f) * noise


class Step(Benchmark):
    r"""Step's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} ⌊x_i⌋

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("Step", -1, False, False, False, False, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Step benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = np.floor(np.fabs(x))

        return np.sum(f)


class Step2(Benchmark):
    r"""Step's 2nd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} ⌊x_i + 0.5⌋^2

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (-0.5, -0.5, \ldots, -0.5)`.

    """

    _defaults = ("Step2", -1, False, False, False, False, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Step2 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = np.floor(x + 0.5) ** 2

        return np.sum(f)


class Step3(Benchmark):
    r"""Step's 3rd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} ⌊x_i^2⌋

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("Step3", -1, False, False, False, False, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Step3 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = np.floor(x**2)

        return np.sum(f)


class StrechedVSineWave(Benchmark):
    r"""Streched V Sine Wave's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-1}(x_{i+1}^2 + x_i^2)^{0.25}[sin^2(50(x_{i+1}^2 + x_i^2)^{0.1})+0.1]

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("StrechedVSineWave", -1, True, True, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the StrechedVSineWave benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(x.shape[0] - 1):
            f += (((x[i + 1] ** 2) + (x[i] ** 2)) ** 0.25) * (
                (np.sin(50 * ((x[i + 1] ** 2) + (x[i] ** 2)) ** 0.1) ** 2) + 0.1
            )

        return f


class StyblinskiTang(Benchmark):
    r"""Styblinski-Tang's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \frac{1}{2}\sum_{i=1}^{n}(x_i^4 - 16x_i^2 + 5x_i)

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = −39.16599n \mid \mathbf{x^*} = (-2.903534, -2.903534, \ldots, -2.903534)`.

    """

    _defaults = ("StyblinskiTang", -1, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the StyblinskiTang benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 1 / 2 * np.sum(x**4 - 16 * x**2 + 5 * x)

        return f


class SumDifferentPowers(Benchmark):
    r"""Sum of Different Powers' benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}|x_i|^{i+1}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-1, 1] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("SumDifferentPowers", -1, True, True, True, False, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the SumDifferentPowers benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(x.shape[0]):
            f += np.fabs(x[i]) ** (i + 2)

        return f


class SumSquares(Benchmark):
    r"""Sum of Squares' benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}ix_i^2

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("SumSquares", -1, True, True, True, False, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the SumSquares benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(x.shape[0]):
            f += (i + 1) * (x[i] ** 2)

        return f


class Trid(Benchmark):
    r"""Trid's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}(x_i - 1)^2 - \sum_{i=2}^{n}x_i x_{i-1}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-n^2, n^2] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -\frac{n(n+4)(n-1)}{6} \mid x_i = i(n+1-i)`.

    """

    _defaults = ("Trid", -1, True, True, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Trid benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        term = 0

        for i in range(1, x.shape[0]):
            term += x[i] * x[i - 1]

        f = np.sum((x - 1) ** 2) - term

        return f


class Trigonometric1(Benchmark):
    r"""Trigonometric's 1st benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}[n - \sum_{j=1}^{n} cos(x_j) + i(1 - cos(x_i) - sin(x_i))]^2

        Domain:
            The function is commonly evaluated using :math:`x_i \in [0, \pi] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("Trigonometric1", -1, True, True, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Trigonometric1 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        n = x.shape[0]

        f = 0

        for i in range(n):
            partial = 0

            for j in range(n):
                partial += np.cos(x[j])

            f += (n - partial + i * (1 - np.cos(x[i] - np.sin(x[i])))) ** 2

        return f


class Trigonometric2(Benchmark):
    r"""Trigonometric's 2nd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = 1 + \sum_{i=1}^{n}8sin^2[7(x_i - 0.9)^2] + 6sin^2[14(x_1-0.9)^2] + (x_i-0.9)^2

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-500, 500] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 1 \mid \mathbf{x^*} = (0.9, 0.9, \ldots, 0.9)`.

    """

    _defaults = ("Trigonometric2", -1, True, True, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Trigonometric2 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(x.shape[0]):
            f += (
                8 * (np.sin(7 * (x[i] - 0.9) ** 2) ** 2)
                + 6 * (np.sin(14 * (x[0] - 0.9) ** 2) ** 2)
                + ((x[i] - 0.9) ** 2)
            )

        return 1 + f


class Wavy(Benchmark):
    r"""Wavy's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = 1 - \frac{1}{n} \sum_{i=1}^{n}cos(10x_i)e^{\frac{-x_i^2}{2}}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-\pi, \pi] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("Wavy", -1, True, True, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Wavy benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = np.cos(10 * x) * np.exp(-1 * (x**2) / 2)

        return 1 - (1 / x.shape[0]) * np.sum(f)


class Weierstrass(Benchmark):
    r"""Weierstrass's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (\sum_{k=0}^{20} [0.5^k cos(2\pi 3^k(x_i+0.5))]) - n \sum_{k=0}^{20}[0.5^k cos(2\pi 3^k 0.5)]

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-0.5, 0.5] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("Weierstrass", -1, True, True, True, True, False)

    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Weierstrass benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0
        partial_term = 0

        for i in range(x.shape[0]):
            for k in range(21):
                f += 0.5**k * np.cos(2 * np.pi * 3**k * (x[i] + 0.5))

        for k in range(21):
            partial_term += 0.5**k * np.cos(2 * np.pi * 3**k * 0.5)

        return f - x.shape[0] * partial_term


class XinSheYang(Benchmark):
    r"""Xin-She Yang's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}\epsilon_i|x_i|^i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("XinSheYang", -1, True, False, False, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the XinSheYang benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(x.shape[0]):
            f += np.random.uniform() * (np.fabs(x[i]) ** (i + 1))

        return f


class XinSheYang2(Benchmark):
    r"""Xin-She Yang's 2nd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = (\sum_{i=1}^{n}|x_i|)e^{-\sum_{i=1}^{n}sin(x_i^2)}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-2 \pi, 2 \pi] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("XinSheYang2", -1, True, False, False, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the XinSheYang2 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = np.sum(np.fabs(x)) * np.exp(-np.sum(np.sin(x**2)))

        return f


class XinSheYang3(Benchmark):
    r"""Xin-She Yang's 3rd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = e^{-\sum_{i=1}^{n}(\frac{x_i}{\beta})^{2m}} - 2e^{-\sum_{i=1}^{n}x_i^2} \prod_{i=1}^{n} cos^2(x_i)

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-2 \pi, 2 \pi] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -1 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("XinSheYang3", -1, True, False, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the XinSheYang3 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = np.exp(-np.sum((x / 15) ** 10)) - 2 * np.exp(-np.sum(x**2)) * np.prod(np.cos(x) ** 2)

        return f


class XinSheYang4(Benchmark):
    r"""Xin-She Yang's 4th benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = (\sum_{i=1}^{n} sin^2(x_i) - e^{-\sum_{i=1}^{n}x_i^2})e^{-\sum_{i=1}^{n}sin^2(\sqrt{|x_i|})}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -1 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("XinSheYang4", -1, True, False, False, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the XinSheYang4 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (np.sum(np.sin(x) ** 2) - np.exp(-np.sum(x**2))) * np.exp(-np.sum(np.sin(np.sqrt(np.fabs(x)) ** 2)))

        return f


class Zakharov(Benchmark):
    r"""Zakharov's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^n x_i^{2}+(\sum_{i=1}^n 0.5ix_i)^2 + (\sum_{i=1}^n 0.5ix_i)^4

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5, 10] \mid i = \{1, 2, \ldots, n\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, \ldots, 0)`.

    """

    _defaults = ("Zakharov", -1, True, True, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Zakharov benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        term = 0

        for i in range(x.shape[0]):
            term += 0.5 * i * x[i]

        f = np.sum(x) + (term**2) + (term**4)

        return f
