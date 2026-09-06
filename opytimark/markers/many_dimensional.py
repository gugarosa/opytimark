# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from typing import Any

import numpy as np
from numpy.typing import NDArray

import opytimark.utils.decorator as d
from opytimark.core import Benchmark
from opytimark.typing import BenchmarkValue


class BiggsExponential3(Benchmark):
    r"""Biggs Exponential's 3rd benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3) = \sum_{i=1}^{10}(e^{-t_ix_1} - x_3e^{-t_ix_2} - y_i)^2

        Domain:
            The function is commonly evaluated using :math:`x_i \in [0, 20] \mid i = \{1, 2, 3\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 10, 5)`.

    """

    _defaults = ("BiggsExponential3", 3, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the BiggsExponential3 benchmark.

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

            f += (np.exp(-z * x[0]) - x[2] * np.exp(-z * x[1]) - y) ** 2

        return f


class BiggsExponential4(Benchmark):
    r"""Biggs Exponential's 4th benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3, x_4) = \sum_{i=1}^{10}(x_3e^{-t_ix_1} - x_4e^{-t_ix_2} - y_i)^2

        Domain:
            The function is commonly evaluated using :math:`x_i \in [0, 20] \mid i = \{1, 2, 3, 4\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 10, 1, 5)`.

    """

    _defaults = ("BiggsExponential4", 4, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the BiggsExponential4 benchmark.

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

            f += (x[2] * np.exp(-z * x[0]) - x[3] * np.exp(-z * x[1]) - y) ** 2

        return f


class BiggsExponential5(Benchmark):
    r"""Biggs Exponential's 5th benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3, x_4, x_5) = \sum_{i=1}^{11}(x_3e^{-t_ix_1} - x_4e^{-t_ix_2} + 3e^{-t_ix_5} - y_i)^2

        Domain:
            The function is commonly evaluated using :math:`x_i \in [0, 20] \mid i = \{1, 2, 3, 4, 5\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 10, 1, 5, 4)`.

    """

    _defaults = ("BiggsExponential5", 5, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the BiggsExponential5 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(1, 12):
            z = i / 10

            y = np.exp(-z) - 5 * np.exp(-10 * z) + 3 * np.exp(-4 * z)

            f += (x[2] * np.exp(-z * x[0]) - x[3] * np.exp(-z * x[1]) + 3 * np.exp(-z * x[4]) - y) ** 2

        return f


class BiggsExponential6(Benchmark):
    r"""Biggs Exponential's 6th benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3, x_4, x_5, x_6) = \sum_{i=1}^{13}(x_3e^{-t_ix_1} - x_4e^{-t_ix_2} + x_6e^{-t_ix_5} - y_i)^2

        Domain:
            The function is commonly evaluated using :math:`x_i \in [0, 20] \mid i = \{1, 2, 3, 4, 5, 6\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 10, 1, 5, 4, 3)`.

    """

    _defaults = ("BiggsExponential6", 6, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the BiggsExponential6 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(1, 14):
            z = i / 10

            y = np.exp(-z) - 5 * np.exp(-10 * z) + 3 * np.exp(-4 * z)

            f += (x[2] * np.exp(-z * x[0]) - x[3] * np.exp(-z * x[1]) + x[5] * np.exp(-z * x[4]) - y) ** 2

        return f


class BoxBetts(Benchmark):
    r"""BoxBetts's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3) = \sum_{i=1}^{n}g(x)^2
        .. math:: g(x) = e^{-0.1(i+1)x_1} - e^{-0.1(i+1)x_2} - (e^{-0.1(i+1)} - e^{-(i+1)}*x_3)

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [0.9, 1.2], x_2 \in [9, 11.2], x_3 \in [0.9, 1.2]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 10, 1)`.

    """

    _defaults = ("BoxBetts", 3, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the BoxBetts benchmark.

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
                np.exp(-0.1 * (i + 2) * x[0])
                - np.exp(-0.1 * (i + 2) * x[1])
                - (np.exp(-0.1 * (i + 2)) - np.exp(-(i + 2)) * x[2])
            ) ** 2

        return f


class Colville(Benchmark):
    r"""Colville's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3, x_4) = 100(x_1 - x_2^2)^2 + (1 - x_1)^2 + 90(x_4 - x_3^2)^2 + (1 - x_3)^2 + 10.1((x_2-1)^2 + (x_4-1)^2) + 19.8(x_2-1)(x_4-1)

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, 3, 4\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 1, 1, 1)`.

    """

    _defaults = ("Colville", 4, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Colville benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (
            100 * (x[0] - x[1] ** 2) ** 2
            + (1 - x[0]) ** 2
            + 90 * (x[3] - x[2] ** 2) ** 2
            + (1 - x[2]) ** 2
            + 10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2)
            + 19.8 * (x[1] - 1) * (x[3] - 1)
        )

        return f


class GulfResearch(Benchmark):
    r"""GulfResearch's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3) = \sum_{i=1}^{99}[e^(-\frac{(u_i-x_2)^x_3}{x_1}) - 0.01i]^2

        Domain:
            The function is commonly evaluated using :math:`x_1 \in [0.1, 100], x_2 \in [0, 25.6], x_3 \in [0, 5]`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (50, 25, 1.5)`.

    """

    _defaults = ("GulfResearch", 3, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the GulfResearch benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(1, 100):
            u = 25 + (-50 * np.log(0.01 * i)) ** (1 / 1.5)

            f += (np.exp(-((u - x[1]) ** x[2]) / x[0]) - 0.01 * i) ** 2

        return f


class HelicalValley(Benchmark):
    r"""Helical Valley's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3) = 100[(x_3-10\theta)^2 + (\sqrt{x_1^2+x_2^2}-1)] + x_3^2

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, 3\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 0, 0)`.

    """

    _defaults = ("HelicalValley", 3, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the HelicalValley benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        if x[0] >= 0:
            theta = np.arctan(x[1] / x[0])

        else:
            theta = np.pi + np.arctan(x[1] / x[0])

        f = 100 * (x[2] - 10 * theta) ** 2 + (np.sqrt(x[0] ** 2 + x[1] ** 2) - 1) ** 2 + x[2] ** 2

        return f


class MieleCantrell(Benchmark):
    r"""MieleCantrell's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3, x_4) = (e^{-x_1} - x_2)^4 + 100(x_2 - x_3)^6 + (tan(x_3-x_4))^4 + x_1^8

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-1, 1] \mid i = \{1, 2, 3, 4\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 1, 1, 1)`.

    """

    _defaults = ("MieleCantrell", 4, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the MieleCantrell benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (np.exp(-x[0]) - x[1]) ** 4 + 100 * (x[1] - x[2]) ** 6 + (np.tan(x[2] - x[3])) ** 4 + x[0] ** 8

        return f


class Mishra9(Benchmark):
    r"""Mishra's 9th benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3) = (ab^2c + abc^2 + b^2 + (x_1 + x_2 - x_3)^2)^2

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, 3\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 2, 3)`.

    """

    _defaults = ("Mishra9", 3, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Mishra9 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        a = 2 * (x[0] ** 3) + 5 * x[0] * x[1] + 4 * x[2] - 2 * (x[0] ** 2) * x[2] - 18

        b = x[0] + (x[1] ** 2) * x[2] + x[0] * (x[2] ** 2) - 22

        c = 8 * (x[0] ** 2) + 2 * x[1] * x[2] + 2 * (x[1] ** 2) + 3 * (x[1] ** 3) - 52

        f = (a * (b**2) * c + a * b * (c**2) + (b**2) + (x[0] + x[1] - x[2]) ** 2) ** 2

        return f


class Paviani(Benchmark):
    r"""Paviani's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_10) = \sum_{i=1}^{10}[ln(x_i-2)^2 + ln(10-x_i)^2] - (\prod_{i=1}^{10}x_i)^{0.2}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [2.0001, 10] \mid i = \{1, 2, \ldots, 10\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -45.778452053828865 \mid \mathbf{x^*} = (9.351, 9.351, \ldots, 9.351)`.

    """

    _defaults = ("Paviani", 10, True, True, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Paviani benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        sum_term = 0

        prod_term = 1

        for i in range(x.shape[0]):
            sum_term += np.log(x[i] - 2) ** 2 + np.log(10 - x[i]) ** 2

            prod_term *= x[i]

        f = sum_term - prod_term**0.2

        return f


class SchmidtVetters(Benchmark):
    r"""Schmidt Vetters's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3) = \frac{1}{1 + (x_1-x_2)^2} + sin(\frac{\pi x_2+x_3}{2}) + e^{(\frac{x_1+x_2}{x_2}-2)^2}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [0, 2] \mid i = \{1, 2, 3\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 3 \mid \mathbf{x^*} = (0.78547, 0.78547, 0.78547)`.

    """

    _defaults = ("SchmidtVetters", 3, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the SchmidtVetters benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 1 / (1 + (x[0] - x[1]) ** 2) + np.sin((np.pi ** x[1] + x[2]) / 2) + np.exp((x[0] + x[1]) / x[1] - 2) ** 2

        return f


class Simpleton(Benchmark):
    r"""Simpleton's Problem benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10) = \frac{x_1 x_2 x_3 x_4 x_5}{x_6 x_7 x_8 x_9 x_10}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [1, 10] \mid i = \{1, 2, \ldots, 10\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 10^5 \mid \mathbf{x^*} = (10, 10, 10, 10, 10, 1, 1, 1, 1, 1)`.

    """

    _defaults = ("Simpleton", 10, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Simpleton benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (x[0] * x[1] * x[2] * x[3] * x[4]) / (x[5] * x[6] * x[7] * x[8] * x[9])

        return f


class Watson(Benchmark):
    r"""Watson's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3, x_4, x_5, x_6) = \sum_{i=0}^{29}{\sum_{j=0}^{4}((j-1)a_i^j x_{j+1}) - [\sum_{j=0}^{5}a_i^j x_{j+1}]^2 - 1}^2 + x_1^2

        Domain:
            The function is commonly evaluated using :math:`|x_i| \leq 10 \mid i = \{1, 2, 3, 4, 5, 6\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0.002288 \mid \mathbf{x^*} = (−0.0158, 1.012, −0.2329, 1.260, −1.513, 0.9928)`.

    """

    _defaults = ("Watson", 6, True, False, True, True, False)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Watson benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 0

        for i in range(30):
            outer_sum, inner_sum = 0, 0

            a = i / 29

            for j in range(2, 7):
                outer_sum += (j - 1) * (a ** (j - 2)) * x[j - 1]

            for j in range(1, 7):
                inner_sum += (a ** (j - 1)) * x[j - 1]

            f += (outer_sum - inner_sum**2 - 1) ** 2

        f += x[0] ** 2

        return f


class Wolfe(Benchmark):
    r"""Wolfe's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3) = \frac{4}{3}(x_1^2 + x_2^2 - x_1x_2)^{0.75} + x_3

        Domain:
            The function is commonly evaluated using :math:`x_i \in [0, 2] \mid i = \{1, 2, 3\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, 0)`.

    """

    _defaults = ("Wolfe", 3, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Wolfe benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = 4 / 3 * ((x[0] ** 2 + x[1] ** 2 - x[0] * x[1]) ** 0.75) + x[2]

        return f
