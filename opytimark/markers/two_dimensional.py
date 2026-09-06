# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from typing import Any

import numpy as np
from numpy.typing import NDArray

import opytimark.utils.constants as c
import opytimark.utils.decorator as d
from opytimark.core import Benchmark
from opytimark.typing import BenchmarkValue


class Ackley2(Benchmark):
    r"""Ackley's 2nd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = -200e^{-0.2\sqrt{x_1^2+x_2^2}}

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-32, 32], x_2 \in [-32, 32]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -200 \mid \mathbf{x^*} = (0, 0)`.

    """

    _defaults = ("Ackley2", 2, True, True, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Ackley2 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = -200 * np.exp(-0.2 * np.sqrt(x[0] ** 2 + x[1] ** 2))

        return f


class Ackley3(Benchmark):
    r"""Ackley's 3rd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = -200e^{-0.02\sqrt{x_1^2+x_2^2}} + 5e^{cos(3x_1) + sin(3x_2)}

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-32, 32], x_2 \in [-32, 32]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) \approx -195.62902823841935 \mid \mathbf{x^*} \approx (\pm 0.682584587365898, -0.36075325513719)`.

    """

    _defaults = ("Ackley3", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Ackley3 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = -200 * np.exp(-0.02 * np.sqrt(x[0] ** 2 + x[1] ** 2)) + 5 * np.exp(np.cos(3 * x[0]) + np.sin(3 * x[1]))

        return f


class Adjiman(Benchmark):
    r"""Adjiman's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = cos(x_1) sin(x_2) - \frac{x_1}{x_2^2+1}

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-1, 2], x_2 \in [-1, 1]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -2.02181 \mid \mathbf{x^*} = (2, 0.10578)`.

    """

    _defaults = ("Adjiman", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Adjiman benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = np.cos(x[0]) * np.sin(x[1]) - (x[0] / (x[1] ** 2 + 1))

        return f


class BartelsConn(Benchmark):
    r"""Bartels Conn's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = |x_1^2+x_2^2+x_1x_2| + |sin(x_1)| + |cos(x_2)|

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 1 \mid \mathbf{x^*} = (0, 0)`.

    """

    _defaults = ("BartelsConn", 2, True, False, False, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the BartelsConn benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = np.fabs(x[0] ** 2 + x[1] ** 2 + x[0] * x[1]) + np.fabs(np.sin(x[0])) + np.fabs(np.cos(x[1]))

        return f


class Beale(Benchmark):
    r"""Beale's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = (1.5-x_1+x_1x_2)^2 + (2.25-x_1+x_1x_2^2)^2 + (2.625-x_1+x_1x_2^3)^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-4.5, 4.5], x_2 \in [-4.5, 4.5]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (3, 0.5)`.

    """

    _defaults = ("Beale", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Beale benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (
            (1.5 - x[0] + x[0] * x[1]) ** 2
            + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
            + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
        )

        return f


class BiggsExponential2(Benchmark):
    r"""Biggs Exponential's 2nd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = \sum_{i=1}^{10}(e^{-t_ix_1} - 5e^{-t_ix_2} - y_i)^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [0, 20], x_2 \in [0, 20]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 10)`.

    """

    _defaults = ("BiggsExponential2", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the BiggsExponential2 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(1, 11):
            z = i / 10

            y = np.exp(-z) - 5 * np.exp(-10 * z)

            f += (np.exp(-z * x[0]) - 5 * np.exp(-z * x[1]) - y) ** 2

        return f


class Bird(Benchmark):
    r"""Bird's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = sin(x_1)e^{(1-cos(x_2))^2}+cos(x_2)e^{(1-sin(x_1))^2}+(x_1-x_2)^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-2\pi, 2\pi], x_2 \in [-2\pi, 2\pi]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -106.764537 \mid \mathbf{x^*} = (4.70104, 3.15294) ~ or ~(−1.58214, −3.13024)`.

    """

    _defaults = ("Bird", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Bird benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (
            np.sin(x[0]) * np.exp((1 - np.cos(x[1])) ** 2)
            + np.cos(x[1]) * np.exp((1 - np.sin(x[0])) ** 2)
            + (x[0] - x[1]) ** 2
        )

        return f


class Bohachevsky1(Benchmark):
    r"""Bohachevsky's 1st benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = x_1^2 + 2x_2^2 - 0.3cos(3\pi x_1) - 0.4cos(4\pi x_2) + 0.7

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-100, 100], x_2 \in [-100, 100]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0)`.

    """

    _defaults = ("Bohachevsky1", 2, True, True, True, False, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Bohachevsky1 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * np.cos(3 * np.pi * x[0]) - 0.4 * np.cos(4 * np.pi * x[1]) + 0.7

        return f


class Bohachevsky2(Benchmark):
    r"""Bohachevsky's 2nd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = x_1^2 + 2x_2^2 - 0.3cos(3\pi x_1)cos(4\pi x_2) + 0.7

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-100, 100], x_2 \in [-100, 100]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0)`.

    """

    _defaults = ("Bohachevsky2", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Bohachevsky2 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * np.cos(3 * np.pi * x[0]) * np.cos(4 * np.pi * x[1]) + 0.3

        return f


class Bohachevsky3(Benchmark):
    r"""Bohachevsky's 3rd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = x_1^2 + 2x_2^2 - 0.3cos(3\pi x_1 + 4\pi x_2) + 0.3

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-100, 100], x_2 \in [-100, 100]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0)`.

    """

    _defaults = ("Bohachevsky3", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Bohachevsky3 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * np.cos(3 * np.pi * x[0] + 4 * np.pi * x[1]) + 0.3

        return f


class Booth(Benchmark):
    r"""Booth's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = (x_1 + 2x_2 - 7)^2 + (2x_1 + x_2 - 5)^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 3)`.

    """

    _defaults = ("Booth", 2, True, False, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Booth benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

        return f


class BraninHoo(Benchmark):
    r"""Branin Hoo's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = (x_2 - \frac{5.1x_1^2}{4\pi^2} + \frac{5x_1}{\pi} - 6)^2 + 10(1 - \frac{1}{8\pi})cos(x_1) + 10

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-5, 10], x_2 \in [0, 15]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0.39788735775266204 \mid \mathbf{x^*} = (-\pi, 12.275) ~ or ~(\pi, 2.275) ~ or ~(3\pi, 2.425)`.

    """

    _defaults = ("BraninHoo", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the BraninHoo benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (
            (x[1] - ((5.1 * x[0] ** 2) / (4 * np.pi**2)) + ((5 * x[0]) / np.pi) - 6) ** 2
            + 10 * (1 - (1 / (8 * np.pi))) * np.cos(x[0])
            + 10
        )

        return f


class Brent(Benchmark):
    r"""Brent's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = (x_1 + 10)^2 + (x_2 + 10)^2 + e^{-x_1^2 - x_2^2}

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = e^{-200} \mid \mathbf{x^*} = (-10, -10)`.

    """

    _defaults = ("Brent", 2, True, True, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Brent benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (x[0] + 10) ** 2 + (x[1] + 10) ** 2 + np.exp(-(x[0] ** 2) - (x[1] ** 2))

        return f


class Bukin2(Benchmark):
    r"""Bukin's 2nd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = 100(x_2 - 0.01x_1^2 + 1) + 0.01(x_1 + 10)^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-15, -5], x_2 \in [-3, 3]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = # \mid \mathbf{x^*} = (-10, 0)`.

    """

    _defaults = ("Bukin2", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Bukin2 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 100 * (x[1] - 0.01 * x[0] ** 2 + 1) + 0.01 * (x[0] + 10) ** 2

        return f


class Bukin4(Benchmark):
    r"""Bukin's 4th benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = 100x_2^2 + 0.01|x_1 + 10|

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-15, -5], x_2 \in [-3, 3]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = # \mid \mathbf{x^*} = (-10, 0)`.

    """

    _defaults = ("Bukin4", 2, True, False, False, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Bukin4 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 100 * x[1] ** 2 + 0.01 * np.fabs(x[0] + 10)

        return f


class Bukin6(Benchmark):
    r"""Bukin's 6th benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = 100sqrt{|x_2-0.01x_1^2|} + 0.01|x_1+10|

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-15, -5], x_2 \in [-3, 3]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = # \mid \mathbf{x^*} = (-10, 1)`.

    """

    _defaults = ("Bukin6", 2, True, False, False, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Bukin6 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 100 * np.sqrt(np.fabs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * np.fabs(x[0] + 10)

        return f


class Camel3(Benchmark):
    r"""Camel's Three Hump benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = 2x_1^2 - 1.05x_1^4 + \frac{x_1^6}{6} + x_1x_2 + x_2^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-5, 5], x_2 \in [-5, 5]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0)`.

    """

    _defaults = ("Camel3", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Camel3 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 2 * x[0] ** 2 - 1.05 * x[0] ** 4 + x[0] ** 6 / 6 + x[0] * x[1] + x[1] ** 2

        return f


class Camel6(Benchmark):
    r"""Camel's Six Hump benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = (4 - 2.1x_1^2 + \frac{x_1^4}{3})x_1^2 + x_1x_2 + (4x_2^2 - 4)x_2^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-5, 5], x_2 \in [-5, 5]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -1.0316284229280819 \mid \mathbf{x^*} = (−0.0898, 0.7126) ~ or ~(0.0898,−0.7126)`.

    """

    _defaults = ("Camel6", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Camel6 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (4 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3) * x[0] ** 2 + x[0] * x[1] + (4 * x[1] ** 2 - 4) * x[1] ** 2

        return f


class ChenBird(Benchmark):
    r"""Chen Bird's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = -\frac{0.001}{0.001^2 + (x_1^2 + x_2^2 - 1)^2} - \frac{0.001}{0.001^2 + (x_1^2 + x_2^2 - 0.5)^2} - \frac{0.001}{0.001^2 + (x_1^2 - x_2^2)^2}

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 2000.0039999840003 \mid \mathbf{x^*} = (0.5, 0.5)`.

    """

    _defaults = ("ChenBird", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the ChenBird benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (
            -((0.001) / (0.001**2 + (x[0] ** 2 + x[1] ** 2 - 1) ** 2))
            - ((0.001) / (0.001**2 + (x[0] ** 2 + x[1] ** 2 - 0.5) ** 2))
            - ((0.001) / (0.001**2 + (x[0] ** 2 - x[1] ** 2) ** 2))
        )

        return f


class ChenV(Benchmark):
    r"""Chen V's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = \frac{0.001}{0.001^2 + (x_1 - 0.4x_2 - 0.1)^2} + \frac{0.001}{0.001^2 + (2x_1 + x_2 - 1.5)^2}

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 2000.0000000000002 \mid \mathbf{x^*} = (0.5, 0.5)`.

    """

    _defaults = ("ChenV", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the ChenV benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = ((0.001) / (0.001**2 + (x[0] - 0.4 * x[1] - 0.1) ** 2)) + (
            (0.001) / (0.001**2 + (2 * x[0] + x[1] - 1.5) ** 2)
        )

        return f


class Chichinadze(Benchmark):
    r"""Chichinadze's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = x_1^2 - 12x_1 + 11 + 10cos(\frac{\pi x_1}{2}) + 8sin(\frac{5\pi x_1}{2}) - \frac{1}{5}^{0.5} e^{-0.5(x_2-0.5)^2}

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-30, 30], x_2 \in [-30, 30]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = −43.3159 \mid \mathbf{x^*} = (5.90133, 0.5)`.

    """

    _defaults = ("Chichinadze", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Chichinadze benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (
            x[0] ** 2
            - 12 * x[0]
            + 11
            + 10 * np.cos((np.pi * x[0]) / 2)
            + 8 * np.sin((5 * np.pi * x[0]) / 2)
            - (1 / 5) ** 0.5 * np.exp(-0.5 * (x[1] - 0.5) ** 2)
        )

        return f


class CrossTray(Benchmark):
    r"""CrossTray's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = -0.0001(|sin(x_1)sin(x_2)e^{|100-\frac{sqrt{x_1^2 + x_2^2}}{\pi}|}| + 1)^0.1

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = −2.06261218 \mid \mathbf{x^*} = (\pm 1.349406685353340, \pm 1.349406608602084)`.

    """

    _defaults = ("CrossTray", 2, True, False, False, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the CrossTray benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (
            -0.0001
            * (
                np.fabs(np.sin(x[0]) * np.sin(x[1]) * np.exp(np.fabs(100 - (np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))))
                + 1
            )
            ** 0.1
        )

        return f


class Cube(Benchmark):
    r"""Cube's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = 100(x_2 - x_1^3)^2 + (1 - x_1)^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (-1, 1)`.

    """

    _defaults = ("Cube", 2, True, False, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Cube benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 100 * (x[1] - x[0] ** 3) ** 2 + (1 - x[0]) ** 2

        return f


class Damavandi(Benchmark):
    r"""Damavandi's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = (1 - |\frac{sin(\pi (x_1-2))sin(\pi (x_2-2))}{\pi^2 (x_1-2)(x2_2)}|^5)(2 + (x_1-7)^2 + 2(x_2-7)^2)

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [0, 14], x_2 \in [0, 14]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (2.00000001, 2.00000001)`.

    """

    _defaults = ("Damavandi", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Damavandi benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (
            1
            - np.fabs(
                (np.sin(np.pi * (x[0] - 2)) * np.sin(np.pi * (x[1] - 2)))
                / (np.pi**2 * (x[0] - 2) * (x[1] - 2) + c.EPSILON)
            )
            ** 5
        ) * (2 + (x[0] - 7) ** 2 + 2 * (x[1] - 7) ** 2)

        return f


class DeckkersAarts(Benchmark):
    r"""Deckkers Aarts' benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = 10^5x_1^2 + x_2^2 - (x_1^2 + x_2^2)^2 + 10^{-5}(x_1^2 + x_2^2)^4

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-20, 20], x_2 \in [-20, 20]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -24771.093749999996 \mid \mathbf{x^*} = (0, \pm 15)`.

    """

    _defaults = ("DeckkersAarts", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the DeckkersAarts benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 10**5 * x[0] ** 2 + x[1] ** 2 - (x[0] ** 2 + x[1] ** 2) ** 2 + 10**-5 * (x[0] ** 2 + x[1] ** 2) ** 4

        return f


class DropWave(Benchmark):
    r"""Drop Wave's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = - \frac{1 + cos(12\sqrt{x_1^2+x_2^2})}{0.5(x_1^2+x_2^2) + 2}

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-5.2, 5.2], x_2 \in [-5.2, 5.2]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -1 \mid \mathbf{x^*} = (0, 0)`.

    """

    _defaults = ("DropWave", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the DropWave benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = -(1 + np.cos(12 * np.sqrt(x[0] ** 2 + x[1] ** 2))) / (0.5 * (x[0] ** 2 + x[1] ** 2) + 2)

        return f


class Easom(Benchmark):
    r"""Easom's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = -cos(x_1)cos(x_2)e^{-(x_1-\pi)^2 -(x_2-\pi)^2}

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-100, 100], x_2 \in [-100, 100]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -1 \mid \mathbf{x^*} = (\pi, \pi)`.

    """

    _defaults = ("Easom", 2, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Easom benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi) ** 2) - (x[1] - np.pi) ** 2)

        return f


class ElAttarVidyasagarDutta(Benchmark):
    r"""El-Attar-Vidyasagar-Dutta's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = (x_1^2 + x_2 - 10)^2 + (x_1 + x_2^2 - 7)^2 + (x_1^2 + x_2^3 - 1)^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 1.7127803548622027 \mid \mathbf{x^*} = (3.4091868222, −2.1714330361)`.

    """

    _defaults = ("ElAttarVidyasagarDutta", 2, True, False, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the ElAttarVidyasagarDutta benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (x[0] ** 2 + x[1] - 10) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2 + (x[0] ** 2 + x[1] ** 3 - 1) ** 2

        return f


class EggCrate(Benchmark):
    r"""Egg Crate's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = x_1^2 + x_2^2 + 25(sin^2(x_1) + sin^2(x_2))

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-5, 5], x_2 \in [-5, 5]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0)`.

    """

    _defaults = ("EggCrate", 2, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the EggCrate benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = x[0] ** 2 + x[1] ** 2 + 25 * (np.sin(x[0]) ** 2 + np.sin(x[1]) ** 2)

        return f


class EggHolder(Benchmark):
    r"""Egg Holder's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = -(x_2 + 47)sin(\sqrt{|x_2+\frac{x_1}{2}+47|})-x_1 sin(\sqrt{|x_1-(x_2+47)|})

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-512, 512], x_2 \in [-512, 512]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -959.6406627106155 \mid \mathbf{x^*} = (512, 404.2319)`.

    """

    _defaults = ("EggHolder", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the EggHolder benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = -(x[1] + 47) * np.sin(np.sqrt(np.fabs(x[1] + (x[0] / 2) + 47))) - x[0] * np.sin(
            np.sqrt(np.fabs(x[0] - (x[1] + 47)))
        )

        return f


class FreudensteinRoth(Benchmark):
    r"""Freudenstein Roth's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = (x_1 - 13 + ((5 - x_2)x_2 - 2)x_2) ** 2 + (x_1 - 29 + ((x_2 + 1)x_2 - 14)x_2) ** 2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (5, 4)`.

    """

    _defaults = ("FreudensteinRoth", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the FreudensteinRoth benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (x[0] - 13 + ((5 - x[1]) * x[1] - 2) * x[1]) ** 2 + (x[0] - 29 + ((x[1] + 1) * x[1] - 14) * x[1]) ** 2

        return f


class Giunta(Benchmark):
    r"""Giunta's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = 0.6 + \sum_{i=1}^{2}[sin(\frac{16}{15}x_i-1) + sin^2(\frac{16}{15}x_i-1) + \frac{1}{50}sin(4(\frac{16}{15}x_i-1))]

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-1, 1], x_2 \in [-1, 1]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0.06447042053690571 \mid \mathbf{x^*} = (0.4673200277395354, 0.4673200169591304)`.

    """

    _defaults = ("Giunta", 2, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Giunta benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0.6

        for i in range(x.shape[0]):
            f += (
                np.sin(16 / 15 * x[i] - 1)
                + np.sin(16 / 15 * x[i] - 1) ** 2
                + (1 / 50) * np.sin(4 * (16 / 15 * x[i] - 1))
            )

        return f


class GoldsteinPrice(Benchmark):
    r"""Goldstein Price's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = [1 + (x_1 + x_2 + 1)^2 (19 - 14x_1 + 3x_1^2 - 14x_2 + 6x_1x_2 + 3x_2^2)] [30 + (2x_1 - 3x_2)^2 (18 - 32x_1 + 12x_1^2 + 48x_2 - 36x_1x_2 + 27x_2^2)]

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-2, 2], x_2 \in [-2, 2]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 3 \mid \mathbf{x^*} = (0, -1)`.

    """

    _defaults = ("GoldsteinPrice", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the GoldsteinPrice benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (
            1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)
        ) * (
            30
            + (2 * x[0] - 3 * x[1]) ** 2
            * (18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2)
        )

        return f


class Himmelblau(Benchmark):
    r"""Himmelblau's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-5, 5], x_2 \in [-5, 5]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (3, 2)`.

    """

    _defaults = ("Himmelblau", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Himmelblau benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

        return f


class HolderTable(Benchmark):
    r"""HolderTable's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = -|sin(x_1)cos(x_2)e^{|1 - \frac{\sqrt{x_1^2 + x_2^2}}{\pi}|}|

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -19.208502567767606 \mid \mathbf{x^*} = (\pm 8.05502, \pm 9.66459)`.

    """

    _defaults = ("HolderTable", 2, True, False, False, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the HolderTable benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = -np.fabs(np.sin(x[0]) * np.cos(x[1]) * np.exp(np.fabs(1 - (np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))))

        return f


class Hosaki(Benchmark):
    r"""Hosaki's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = (1 - 8x_1 + 7x_1^2 - \frac{7}{3x_1^3} + \frac{1}{4x_1^4})x_2^2e^{-x_2}

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [0, 5], x_2 \in [0, 6]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -2.345811576101292 \mid \mathbf{x^*} = (4, 2)`.

    """

    _defaults = ("Hosaki", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Hosaki benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (1 - 8 * x[0] + 7 * x[0] ** 2 - 7 / 3 * x[0] ** 3 + 1 / 4 * x[0] ** 4) * x[1] ** 2 * np.exp(-x[1])

        return f


class JennrichSampson(Benchmark):
    r"""Jennrich Sampson's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = \sum_{i=1}^{10}(2 + 2i - (e^{ix_1} + e^{ix_2}))^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-1, 1], x_2 \in [-1, 1]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 124.36218236181409 \mid \mathbf{x^*} = (0.257825, 0.257825)`.

    """

    _defaults = ("JennrichSampson", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the JennrichSampson benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(1, 11):
            f += (2 + 2 * i - (np.exp(i * x[0]) + np.exp(i * x[1]))) ** 2

        return f


class Keane(Benchmark):
    r"""Keane's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = \frac{sin^2(x_1-x_2)sin^2(x_1+x_2)}{sqrt{x_1^2+x_2^2}}

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [0, 10], x_2 \in [0, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0.6736675211468548 \mid \mathbf{x^*} = (1.393249070031784, 0) ~ or ~(0, 1.393249070031784)`.

    """

    _defaults = ("Keane", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Keane benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (np.sin(x[0] - x[1]) ** 2 * np.sin(x[0] + x[1]) ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2)

        return f


class Leon(Benchmark):
    r"""Leon's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = 100(x_2 - x_1^2)^2 + (1 - x_1)^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-1.2, 1.2], x_2 \in [-1.2, 1.2]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 1)`.

    """

    _defaults = ("Leon", 2, True, False, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Leon benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

        return f


class Levy13(Benchmark):
    r"""Levy's 13th benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = sin^2(3\pi x_1)+(x_1-1)^2(1+sin^2(3\pi x_2))+(x_2-1)^2(1+sin^2(2\pi x_2))

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 1)`.

    """

    _defaults = ("Levy13", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Levy13 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (
            np.sin(3 * np.pi * x[0]) ** 2
            + (x[0] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1]) ** 2)
            + (x[1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[1]) ** 2)
        )

        return f


class Matyas(Benchmark):
    r"""Matyas' benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = 0.26(x_1^2 + x_2^2) - 0.48x_1x_2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0)`.

    """

    _defaults = ("Matyas", 2, True, False, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Matyas benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]

        return f


class McCormick(Benchmark):
    r"""McCormick's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = sin(x_1 + x_2) + (x_1 - x_2)^2 - \frac{3}{2}x_1 + \frac{5}{2}x_2 + 1

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-1.5, 4], x_2 \in [-3, 3]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -1.9132228873800594 \mid \mathbf{x^*} = (−0.547, −1.547)`.

    """

    _defaults = ("McCormick", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the McCormick benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = np.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 3 / 2 * x[0] + 5 / 2 * x[1] + 1

        return f


class Mishra3(Benchmark):
    r"""Mishra's 3rd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = |cos(sqrt{|x_1^2+x_2|})|^{0.5} + 0.01(x_1+x_2)

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -0.18465133334298883 \mid \mathbf{x^*} = (−8.466613775046579, −9.998521308999999)`.

    """

    _defaults = ("Mishra3", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Mishra3 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = np.fabs(np.cos(np.sqrt(np.fabs(x[0] ** 2 + x[1])))) ** 0.5 + 0.01 * (x[0] + x[1])

        return f


class Mishra4(Benchmark):
    r"""Mishra's 4th benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = |sin(sqrt{|x_1^2+x_2|})|^{0.5} + 0.01(x_1+x_2)

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -0.1994069700888328 \mid \mathbf{x^*} = (−9.941127263635860, −9.999571661999983)`.

    """

    _defaults = ("Mishra4", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Mishra4 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = np.fabs(np.sin(np.sqrt(np.fabs(x[0] ** 2 + x[1])))) ** 0.5 + 0.01 * (x[0] + x[1])

        return f


class Mishra5(Benchmark):
    r"""Mishra's 5th benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = [sin^2(cos(x_1)+cos(x_2))^2 + cos^2(sin(x_1)+sin(x_2)) + x_1]^2 + 0.01x_1 + 0.1x_2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -1.019829519930943 \mid \mathbf{x^*} = (−1.986820662153768, −10)`.

    """

    _defaults = ("Mishra5", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Mishra5 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (
            (np.sin((np.cos(x[0]) + np.cos(x[1])) ** 2) ** 2 + np.cos((np.sin(x[0]) + np.sin(x[1])) ** 2) ** 2 + x[0])
            ** 2
            + 0.01 * x[0]
            + 0.1 * x[1]
        )

        return f


class Mishra6(Benchmark):
    r"""Mishra's 6th benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = -log[sin^2(cos(x_1)+cos(x_2))^2 - cos^2(sin(x_1)+sin(x_2)) + x_1]^2 + 0.1((x_1-1)^2 + (x_2-1)^2)

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -2.2839498384747587 \mid \mathbf{x^*} = (2.886307215440481, 1.823260331422321)`.

    """

    _defaults = ("Mishra6", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Mishra6 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = -np.log(
            (np.sin((np.cos(x[0]) + np.cos(x[1])) ** 2) ** 2 - np.cos((np.sin(x[0]) + np.sin(x[1])) ** 2) ** 2 + x[0])
            ** 2
        ) + 0.1 * ((x[0] - 1) ** 2 + (x[1] - 1) ** 2)

        return f


class Mishra8(Benchmark):
    r"""Mishra's 8th benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = 0.001(|g(x_1)| + |h(x_2)|)^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (2, -3)`.

    """

    _defaults = ("Mishra8", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Mishra8 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        g = (
            x[0] ** 10
            - 20 * x[0] ** 9
            + 180 * x[0] ** 8
            - 960 * x[0] ** 7
            + 3360 * x[0] ** 6
            - 8064 * x[0] ** 5
            + 13340 * x[0] ** 4
            - 15360 * x[0] ** 3
            + 11520 * x[0] ** 2
            - 5120 * x[0]
            + 2624
        )

        h = x[1] ** 4 + 12 * x[1] ** 3 + 54 * x[1] ** 2 + 108 * x[1] + 81

        f = 0.001 * (np.fabs(g) + np.fabs(h)) ** 2

        return f


class Parsopoulos(Benchmark):
    r"""Parsopoulos' benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = cos(x_1)^2 + sin(x_2)^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-5, 5], x_2 \in [-5, 5]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (k\frac{\pi}{2}, \lambda \pi)`.

    """

    _defaults = ("Parsopoulos", 2, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Parsopoulos benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = np.cos(x[0]) ** 2 + np.sin(x[1]) ** 2

        return f


class PenHolder(Benchmark):
    r"""Pen Holder's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = -e^{[cos(x_1)cos(x_2)e^{|1-[(x_1^2+x_2^2)]^{0.5} / \pi|}|]^{-1}}

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-11, 11], x_2 \in [-11, 11]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -0.9635348327265058 \mid \mathbf{x^*} = (\pm 9.646167671043401, \pm 9.646167671043401)`.

    """

    _defaults = ("PenHolder", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the PenHolder benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = -np.exp(
            -1 / np.fabs(np.cos(x[0]) * np.cos(x[1]) * np.exp(np.fabs(1 - (x[0] ** 2 + x[1] ** 2) ** 0.5 / np.pi)))
        )

        return f


class Periodic(Benchmark):
    r"""Periodic's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = 1 + sin^2(x_1) + sin^2(x_2) - 0.1e^{-(x_1^2+x_2^2)}

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0.9 \mid \mathbf{x^*} = (0, 0)`.

    """

    _defaults = ("Periodic", 2, True, False, True, True, True)

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

        f = 1 + np.sin(x[0]) ** 2 + np.sin(x[1]) ** 2 - 0.1 * np.exp(-(x[0] ** 2 + x[1] ** 2))

        return f


class Price1(Benchmark):
    r"""Price's 1st benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = (|x_1| - 5)^2 + (|x_2| - 5)^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (-5, -5) ~ or ~(-5, 5) ~ or ~(5, -5) ~ or ~(5, 5)`.

    """

    _defaults = ("Price1", 2, True, False, False, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Price1 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (np.fabs(x[0]) - 5) ** 2 + (np.fabs(x[1]) - 5) ** 2

        return f


class Price2(Benchmark):
    r"""Price's 2nd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = 1 + sin^2(x_1) + sin^2(x_2) - 0.1e^{-x_1^2-x_2^2}

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0.9 \mid \mathbf{x^*} = (0, 0)`.

    """

    _defaults = ("Price2", 2, True, False, False, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Price2 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 1 + np.sin(x[0]) ** 2 + np.sin(x[1]) ** 2 - 0.1 * np.exp(-x[0] ** 2 - x[1] ** 2)

        return f


class Price3(Benchmark):
    r"""Price's 3rd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = 100(x_2 - x_1^2)^2 + [6.4(x_2 - 0.5)^2 - x_1 - 0.6]^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-5, 5], x_2 \in [-5, 5]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0.341307503353524, 0.116490811845416) ~ or ~(1, 1)`.

    """

    _defaults = ("Price3", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Price3 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 100 * (x[1] - x[0] ** 2) ** 2 + (6.4 * (x[1] - 0.5) ** 2 - x[0] - 0.6) ** 2

        return f


class Price4(Benchmark):
    r"""Price's 4th benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = (2x_1^3x_2 - x_2^3)^2 + (6x_1 - x_2^2 + x_2)^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0) ~ or ~(2, 4) ~ or ~(1.464, -2.506)`.

    """

    _defaults = ("Price4", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Price4 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (2 * x[0] ** 3 * x[1] - x[1] ** 3) ** 2 + (6 * x[0] - x[1] ** 2 + x[1]) ** 2

        return f


class Quadratic(Benchmark):
    r"""Quadratic's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = -3803.84 - 138.08x_1 - 232.92x_2 + 128.08x_1^2 + 203.64x_2^2 + 182.25x_1x_2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -3873.7243 \mid \mathbf{x^*} = (0.19388, 0.48513)`.

    """

    _defaults = ("Quadratic", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Quadratic benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = -3803.84 - 138.08 * x[0] - 232.92 * x[1] + 128.08 * x[0] ** 2 + 203.64 * x[1] ** 2 + 182.25 * x[0] * x[1]

        return f


class RotatedEllipse1(Benchmark):
    r"""Rotated Ellipse's 1st benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = 7x_1^2 - 6\sqrt{3}x_1x_2 + 13x_2^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0)`.

    """

    _defaults = ("RotatedEllipse1", 2, True, False, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the RotatedEllipse1 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 7 * x[0] ** 2 - 6 * np.sqrt(3) * x[0] * x[1] + 13 * x[1] ** 2

        return f


class RotatedEllipse2(Benchmark):
    r"""Rotated Ellipse's 2nd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = x_1^2 - x_1x_2 + x_2^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0)`.

    """

    _defaults = ("RotatedEllipse2", 2, True, False, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the RotatedEllipse2 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = x[0] ** 2 - x[0] * x[1] + x[1] ** 2

        return f


class Rump(Benchmark):
    r"""Rump's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = (333.75 - x_1^2)x_2^6 + x_1^2(11x_1^2x_2^2 - 121x_2^4 - 2) + 5.5x_2^8 + \frac{x_1}{2x_2}

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0)`.

    """

    _defaults = ("Rump", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Rump benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (
            (333.75 - x[0] ** 2) * x[1] ** 6
            + x[0] ** 2 * (11 * x[0] ** 2 * x[1] ** 2 - 121 * x[1] ** 4 - 2)
            + 5.5 * x[1] ** 8
            + x[0] / (2 * x[1] + c.EPSILON)
        )

        return f


class Schaffer1(Benchmark):
    r"""Schaffer's 1st benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = 0.5 + \frac{sin^2(x_1^2 + x_2^2)^2 - 0.5}{1 + 0.001(x_1^2 + x_2^2)^2}

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-100, 100], x_2 \in [-100, 100]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0)`.

    """

    _defaults = ("Schaffer1", 2, True, False, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Schaffer1 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0.5 + (np.sin((x[0] ** 2 + x[1] ** 2) ** 2) ** 2 - 0.5) / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2) ** 2)

        return f


class Schaffer2(Benchmark):
    r"""Schaffer's 2nd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = 0.5 + \frac{sin^2(x_1^2 - x_2^2)^2 - 0.5}{1 + 0.001(x_1^2 + x_2^2)^2}

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-100, 100], x_2 \in [-100, 100]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0)`.

    """

    _defaults = ("Schaffer2", 2, True, False, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Schaffer2 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0.5 + (np.sin((x[0] ** 2 - x[1] ** 2) ** 2) ** 2 - 0.5) / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2) ** 2)

        return f


class Schaffer3(Benchmark):
    r"""Schaffer's 3rd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = 0.5 + \frac{sin^2(cos|x_1^2 + x_2^2|) - 0.5}{(1 + 0.001(x_1^2 + x_2^2))^2}

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-100, 100], x_2 \in [-100, 100]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0.0015668545260126288 \mid \mathbf{x^*} = (0, 1.253115)`.

    """

    _defaults = ("Schaffer3", 2, True, False, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Schaffer3 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (
            0.5
            + (np.sin(np.cos(np.fabs(x[0] ** 2 + x[1] ** 2))) ** 2 - 0.5) / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2
        )

        return f


class Schaffer4(Benchmark):
    r"""Schaffer's 4th benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = 0.5 + \frac{cos^2(sin(x_1^2 - x_2^2)) - 0.5}{1 + 0.001(x_1^2 + x_2^2)^2}

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-100, 100], x_2 \in [-100, 100]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0.29243850703298857 \mid \mathbf{x^*} = (0, 1.253115)`.

    """

    _defaults = ("Schaffer4", 2, True, False, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Schaffer4 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0.5 + (np.cos(np.sin(x[0] ** 2 - x[1] ** 2)) ** 2 - 0.5) / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2) ** 2)

        return f


class Schwefel26(Benchmark):
    r"""Schwefel's 2.6 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = \max(|x_1 + 2x_2 - 7|, |2x_1 + x_2 - 5|)

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-100, 100], x_2 \in [-100, 100]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 3)`.

    """

    _defaults = ("Schwefel26", 2, True, False, False, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Schwefel26 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = max(np.fabs(x[0] + 2 * x[1] - 7), np.fabs(2 * x[0] + x[1] - 5))

        return f


class Schwefel236(Benchmark):
    r"""Schwefel's 2.36 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = -x_1x_2(72 - 2x_1 - 2x_2)

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [0, 500], x_2 \in [0, 500]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -3456 \mid \mathbf{x^*} = (12, 12)`.

    """

    _defaults = ("Schwefel236", 2, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Schwefel236 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = -x[0] * x[1] * (72 - 2 * x[0] - 2 * x[1])

        return f


class Table1(Benchmark):
    r"""Table's 1st benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = -|cos(x_1)cos(x_2)e^{|1-(x_1+x_2)^{0.5}/\pi|}

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -26.920335555515848 \mid \mathbf{x^*} = (\pm 9.646168, \pm 9.646168)`.

    """

    _defaults = ("Table1", 2, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Table1 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = -np.fabs(np.cos(x[0]) * np.cos(x[1]) * np.exp(np.fabs(1 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi)))

        return f


class Table2(Benchmark):
    r"""Table's 2nd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = -|cos(x_1)cos(x_2)e^{|1-(x_1+x_2)^{0.5}/\pi|}

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -19.20850256788675 \mid \mathbf{x^*} = (\pm 8.055023472141116, \pm 9.664590028909654)`.

    """

    _defaults = ("Table2", 2, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Table2 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = -np.fabs(np.sin(x[0]) * np.cos(x[1]) * np.exp(np.fabs(1 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi)))

        return f


class Table3(Benchmark):
    r"""Table's 3rd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = -\frac{1}{30}e^{2|1 - \frac{sqrt{x_1^2+x_2^2}{\pi}}}cos^2(x_1)cos^2(x_2)

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -19.20850256788675 \mid \mathbf{x^*} = (\pm 9.646157266348881, \pm 9.646134286497169)`.

    """

    _defaults = ("Table3", 2, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Table3 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (
            -np.cos(x[0]) ** 2
            * np.cos(x[1]) ** 2
            * np.exp(2 * np.fabs(1 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))
            / 30
        )

        return f


class TesttubeHolder(Benchmark):
    r"""Testtube Holder's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = -4[(sin(x_1)cos(x_2)e^{|cos[(x_1^2+x_2^2)/200]|})]

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -10.872299901558 \mid \mathbf{x^*} = (\pm \frac{\pi}{2}, 0)`.

    """

    _defaults = ("TesttubeHolder", 2, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the TesttubeHolder benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = -4 * (np.sin(x[0]) * np.cos(x[1]) * np.exp(np.fabs(np.cos((x[0] ** 2 + x[1] ** 2) / 200))))

        return f


class Trecani(Benchmark):
    r"""Trecani's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = x_1^4 - 4x_1^3 + 4x_1 + x_2^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-5, 5], x_2 \in [-5, 5]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0) ~ or ~(-2, 0)`.

    """

    _defaults = ("Trecani", 2, True, False, True, False, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Trecani benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = x[0] ** 4 - 4 * x[0] ** 3 + 4 * x[0] + x[1] ** 2

        return f


class Trefethen(Benchmark):
    r"""Trefethen's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = e^{sin(50x_1)} + sin(60e^{x^2}) + sin(70sin(x_1)) + sin(sin(80x_2)) - sin(10(x_1 + x_2)) + \frac{1}{4}(x_1^2 + x_2^2)

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -3.3068686465567008 \mid \mathbf{x^*} = (−0.024403, 0.210612)`.

    """

    _defaults = ("Trefethen", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Trefethen benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (
            np.exp(np.sin(50 * x[0]))
            + np.sin(60 * np.exp(x[1]))
            + np.sin(70 * np.sin(x[0]))
            + np.sin(np.sin(80 * x[1]))
            - np.sin(10 * (x[0] + x[1]))
            + (1 / 4) * (x[0] ** 2 + x[1] ** 2)
        )

        return f


class VenterSobiezcczanskiSobieski(Benchmark):
    r"""Venter Sobiezcczanski-Sobieski's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = x_1^2 - 100cos(x_1)^2 - 100cos(\frac{x_1^2}{30}) + x_2^2 - 100cos(x_2)^2 - 100cos(\frac{x_2^2}{30})

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-50, 50], x_2 \in [-50, 50]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -400 \mid \mathbf{x^*} = (0, 0)`.

    """

    _defaults = ("VenterSobiezcczanskiSobieski", 2, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the VenterSobiezcczanskiSobieski benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (
            x[0] ** 2
            - 100 * np.cos(x[0]) ** 2
            - 100 * np.cos(x[0] ** 2 / 30)
            + x[1] ** 2
            - 100 * np.cos(x[1]) ** 2
            - 100 * np.cos(x[1] ** 2 / 30)
        )

        return f


class WayburnSeader1(Benchmark):
    r"""WayburnSeader's 1st benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = (x_1^6 + x_2^4 - 17)^2 + (2x_1 + x_2 - 4)^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 2) ~ or ~(1.596804153876933, 0.806391692246134)`.

    """

    _defaults = ("WayburnSeader1", 2, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the WayburnSeader1 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (x[0] ** 6 + x[1] ** 4 - 17) ** 2 + (2 * x[0] + x[1] - 4) ** 2

        return f


class WayburnSeader2(Benchmark):
    r"""WayburnSeader's 2nd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = [1.613 - 4(x_1 - 0.3125)^2 - 4(x_2 - 1.625)^2]^2 + (x_2 - 1)^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0.200138974728779, 1) ~ or ~(0.424861025271221, 1)`.

    """

    _defaults = ("WayburnSeader2", 2, True, False, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the WayburnSeader2 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (1.613 - 4 * (x[0] - 0.3125) ** 2 - 4 * (x[1] - 1.625) ** 2) ** 2 + (x[1] - 1) ** 2

        return f


class WayburnSeader3(Benchmark):
    r"""WayburnSeader's 3rd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = \frac{2}{3}x_1^3 - 8x_1^2 + 33x_1 - x_1x_2 + 5 + [(x_1 - 4)^2 + (x_2 - 5)^2 - 4]^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-500, 500], x_2 \in [-500, 500]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 19.105879794568 \mid \mathbf{x^*} = (5.146896745324582, 6.839589743000071)`.

    """

    _defaults = ("WayburnSeader3", 2, True, False, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the WayburnSeader3 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (
            (2 / 3) * x[0] ** 3
            - 8 * x[0] ** 2
            + 33 * x[0]
            - x[0] * x[1]
            + 5
            + ((x[0] - 4) ** 2 + (x[1] - 5) ** 2 - 4) ** 2
        )

        return f


class Zettl(Benchmark):
    r"""Zettl's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = (x_1^2 + x_2^2 - 2x_1)^2 + 0.25x_1

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-5, 10], x_2 \in [-5, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -0.0037912371501199 \mid \mathbf{x^*} = (−0.0299, 0)`.

    """

    _defaults = ("Zettl", 2, True, False, True, False, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Zettl benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (x[0] ** 2 + x[1] ** 2 - 2 * x[0]) ** 2 + 0.25 * x[0]

        return f


class Zirilli(Benchmark):
    r"""Zirilli's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2) = 0.25x_1^4 - 0.5x_1^2 + 0.1x_1 + 0.5x_2^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [-10, 10], x_2 \in [-10, 10]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -0.3523860365437344 \mid \mathbf{x^*} = (-1.0465, 0)`.

    """

    _defaults = ("Zirilli", 2, True, False, True, False, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Zirilli benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0.25 * x[0] ** 4 - 0.5 * x[0] ** 2 + 0.1 * x[0] + 0.5 * x[1] ** 2

        return f
