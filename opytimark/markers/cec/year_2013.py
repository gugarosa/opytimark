# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from typing import Any

import numpy as np
from numpy.typing import NDArray

import opytimark.markers.n_dimensional as n_dim
import opytimark.utils.constants as c
import opytimark.utils.decorator as d
import opytimark.utils.exception as e
from opytimark.core import CECBenchmark
from opytimark.typing import BenchmarkValue


def T_irregularity(x: NDArray[Any]) -> NDArray[Any]:
    """Introduce smooth local irregularities in the input.

    Args:
        x: An array holding the input to be transformed.

    Returns:
        The transformed input.

    """

    x_hat = np.where(x != 0, np.log(np.fabs(x + c.EPSILON)), 0)

    c_1 = np.where(x > 0, 10, 5.5)
    c_2 = np.where(x > 0, 7.9, 3.1)

    x_t = np.sign(x) * np.exp(x_hat + 0.049 * (np.sin(c_1 * x_hat) + np.sin(c_2 * x_hat)))

    return x_t


def T_asymmetry(x: NDArray[Any], beta: float) -> NDArray[Any]:
    """Break the symmetry of positive input coordinates.

    Args:
        x: An array holding the input to be transformed.
        beta: Exponential value used to produce the asymmetry.

    Returns:
        The transformed input.

    """

    x_t = np.array(x, dtype=np.result_type(x, np.float64), copy=True)
    positive = x_t > 0
    exponents = 1 + beta * np.linspace(0, 1, x_t.shape[0])[positive] * np.sqrt(x_t[positive])
    x_t[positive] **= exponents
    return x_t


def _diagonal_weights(D: int, alpha: float) -> NDArray[Any]:
    return alpha ** np.linspace(0, 0.5, D)


def T_diagonal(D: int, alpha: float) -> NDArray[Any]:
    """Construct the diagonal conditioning matrix.

    Args:
        D: Amount of dimensions.
        alpha: Exponential value used to produce the ill-conditioning.

    Returns:
        The transformed diagonal matrix.

    Notes:
        Its diagonal is ``alpha ** (i / (2 * (D - 1)))`` for zero-based ``i``.
        For one dimension, the matrix is the identity.

    """

    return np.diag(_diagonal_weights(D, alpha))


class F1(CECBenchmark):
    r"""Shifted Elliptic's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (10^6)^\frac{i-1}{n-1} z_i^2 \mid z_i = x_i - o_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F1", 1000, True, True, True, False, True)
    _year = "2013"
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

        z = T_irregularity(x - self.o[: x.shape[0]])
        return n_dim._elliptic(z)


class F2(CECBenchmark):
    r"""Shifted Rastrigin's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (z_i^2 - 10cos(2 \pi z_i) + 10) \mid z_i = x_i - o_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F2", 1000, True, True, True, True, True)
    _year = "2013"
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

        z = T_asymmetry(T_irregularity(x - self.o[: x.shape[0]]), 0.2) * _diagonal_weights(x.shape[0], 10)

        f = z**2 - 10 * np.cos(2 * np.pi * z) + 10

        return np.sum(f)


class F3(CECBenchmark):
    r"""Shifted Ackley's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = -20e^{-0.2\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2}}-e^{\frac{1}{n}\sum_{i=1}^{n}cos(2 \pi x_i)}+ 20 + e \mid z_i = x_i - o_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-32, 32] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F3", 1000, True, True, True, True, True)
    _year = "2013"
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

        z = T_asymmetry(T_irregularity(x - self.o[: x.shape[0]]), 0.2) * _diagonal_weights(x.shape[0], 10)

        inv = 1 / x.shape[0]

        term1 = -0.2 * np.sqrt(inv * np.sum(z**2))

        term2 = inv * np.sum(np.cos(2 * np.pi * z))

        f = 20 + np.e - np.exp(term2) - 20 * np.exp(term1)

        return f


class F4(CECBenchmark):
    r"""7-separable, 1-separable Shifted and Rotated Elliptic's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{|S|-1}w_i f_{elliptic}(z_i) + f_{elliptic}(z_{|S|})

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F4", 1000, True, True, True, False, False)
    _year = "2013"
    _auxiliary_data = ("o", "R25", "R50", "R100")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F4 benchmark.

        Args:
            args: Positional metadata overrides accepted by CECBenchmark.
            kwargs: Keyword metadata overrides accepted by CECBenchmark.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        super().__init__(*args, **kwargs)

        self.S = [50, 25, 25, 100, 50, 25, 25]
        self.W = [45.6996, 1.5646, 18465.3234, 0.011, 13.6259, 0.3015, 59.6078]
        self.f = n_dim.HighConditionedElliptic()

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

        D = x.shape[0]
        P = np.random.permutation(D)
        n = 0
        f = 0

        if D < 302:
            raise e.SizeError(f"`D` must be at least 302, but got {D}.")

        y = x - self.o[:D]
        y = y[P]

        for s, w in zip(self.S, self.W):
            if s == 25:
                z = np.matmul(self.R25, y[n : n + s])

            elif s == 50:
                z = np.matmul(self.R50, y[n : n + s])

            elif s == 100:
                z = np.matmul(self.R100, y[n : n + s])

            f += w * self.f(T_irregularity(z))

            n += s

        z = y[n:]

        f += self.f(T_irregularity(z))

        return f


class F5(CECBenchmark):
    r"""7-separable, 1-separable Shifted and Rotated Rastrigin's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{|S|-1}w_i f_{rastrigin}(z_i) + f_{rastrigin}(z_{|S|})

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F5", 1000, True, True, True, True, False)
    _year = "2013"
    _auxiliary_data = ("o", "R25", "R50", "R100")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F5 benchmark.

        Args:
            args: Positional metadata overrides accepted by CECBenchmark.
            kwargs: Keyword metadata overrides accepted by CECBenchmark.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        super().__init__(*args, **kwargs)

        self.S = [50, 25, 25, 100, 50, 25, 25]
        self.W = [0.1807, 9081.1379, 24.2718, 1.863e-06, 17698.0807, 0.0002, 0.0152]
        self.f = n_dim.Rastrigin()

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

        D = x.shape[0]
        P = np.random.permutation(D)
        n = 0
        f = 0

        if D < 302:
            raise e.SizeError(f"`D` must be at least 302, but got {D}.")

        y = x - self.o[:D]
        y = y[P]

        for s, w in zip(self.S, self.W):
            if s == 25:
                z = np.matmul(self.R25, y[n : n + s])

            elif s == 50:
                z = np.matmul(self.R50, y[n : n + s])

            elif s == 100:
                z = np.matmul(self.R100, y[n : n + s])

            z = T_asymmetry(T_irregularity(z), 0.2) * _diagonal_weights(z.shape[0], 10)

            f += w * self.f(z)

            n += s

        z = y[n:]

        z = T_asymmetry(T_irregularity(z), 0.2) * _diagonal_weights(z.shape[0], 10)

        f += self.f(z)

        return f


class F6(CECBenchmark):
    r"""7-separable, 1-separable Shifted and Rotated Ackley's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{|S|-1}w_i f_{ackley}(z_i) + f_{ackley}(z_{|S|})

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-32, 32] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F6", 1000, True, True, True, True, False)
    _year = "2013"
    _auxiliary_data = ("o", "R25", "R50", "R100")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F6 benchmark.

        Args:
            args: Positional metadata overrides accepted by CECBenchmark.
            kwargs: Keyword metadata overrides accepted by CECBenchmark.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        super().__init__(*args, **kwargs)

        self.S = [50, 25, 25, 100, 50, 25, 25]
        self.W = [0.0352, 5.3156e-05, 0.8707, 49513.742, 0.0831, 3.4764e-05, 282.2934]
        self.f = n_dim.Ackley1()

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

        D = x.shape[0]
        P = np.random.permutation(D)
        n = 0
        f = 0

        if D < 302:
            raise e.SizeError(f"`D` must be at least 302, but got {D}.")

        y = x - self.o[:D]
        y = y[P]

        for s, w in zip(self.S, self.W):
            if s == 25:
                z = np.matmul(self.R25, y[n : n + s])

            elif s == 50:
                z = np.matmul(self.R50, y[n : n + s])

            elif s == 100:
                z = np.matmul(self.R100, y[n : n + s])

            z = T_asymmetry(T_irregularity(z), 0.2) * _diagonal_weights(z.shape[0], 10)

            f += w * self.f(z)

            n += s

        z = y[n:]

        z = T_asymmetry(T_irregularity(z), 0.2) * _diagonal_weights(z.shape[0], 10)

        f += self.f(z)

        return f


class F7(CECBenchmark):
    r"""7-separable, 1-separable Shifted Schwefel's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{|S|-1}w_i f_{schwefel}(z_i) + f_{sphere}(z_{|S|})

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F7", 1000, True, True, True, True, False)
    _year = "2013"
    _auxiliary_data = ("o", "R25", "R50", "R100")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F7 benchmark.

        Args:
            args: Positional metadata overrides accepted by CECBenchmark.
            kwargs: Keyword metadata overrides accepted by CECBenchmark.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        super().__init__(*args, **kwargs)

        self.S = [50, 25, 25, 100, 50, 25, 25]
        self.W = [679.9025, 0.9321, 2122.8501, 0.506, 434.5961, 33389.6244, 2.5692]
        self.f_1 = n_dim.RotatedHyperEllipsoid()
        self.f_2 = n_dim.Sphere()

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

        D = x.shape[0]
        P = np.random.permutation(D)
        n = 0
        f = 0

        if D < 302:
            raise e.SizeError(f"`D` must be at least 302, but got {D}.")

        y = x - self.o[:D]
        y = y[P]

        for s, w in zip(self.S, self.W):
            if s == 25:
                z = np.matmul(self.R25, y[n : n + s])

            elif s == 50:
                z = np.matmul(self.R50, y[n : n + s])

            elif s == 100:
                z = np.matmul(self.R100, y[n : n + s])

            z = T_asymmetry(T_irregularity(z), 0.2)

            f += w * self.f_1(z)

            n += s

        z = y[n:]

        z = T_asymmetry(T_irregularity(z), 0.2)

        f += self.f_2(z)

        return f


class F8(CECBenchmark):
    r"""20-nonseparable Shifted and Rotated Elliptic's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{|S|}w_i f_{elliptic}(z_i)

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F8", 1000, True, True, True, False, False)
    _year = "2013"
    _auxiliary_data = ("o", "R25", "R50", "R100")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F8 benchmark.

        Args:
            args: Positional metadata overrides accepted by CECBenchmark.
            kwargs: Keyword metadata overrides accepted by CECBenchmark.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        super().__init__(*args, **kwargs)

        self.S = [
            50,
            50,
            25,
            25,
            100,
            100,
            25,
            25,
            50,
            25,
            100,
            25,
            100,
            50,
            25,
            25,
            25,
            100,
            50,
            25,
        ]
        self.W = [
            4.6303,
            0.6864,
            1143756360.0887,
            2.0077,
            789.3671,
            16.3332,
            6.0749,
            0.0646,
            0.0756,
            35.6725,
            7.9725e-06,
            10.7822,
            4.1999e-06,
            0.0019,
            0.0016,
            686.7975,
            0.1571,
            0.0441,
            0.3543,
            0.006,
        ]
        self.f = n_dim.HighConditionedElliptic()

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F8 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        D = x.shape[0]
        P = np.random.permutation(D)
        n = 0
        f = 0

        y = x - self.o[:D]
        y = y[P]

        for s, w in zip(self.S, self.W):
            if s == 25:
                z = np.matmul(self.R25, y[n : n + s])

            elif s == 50:
                z = np.matmul(self.R50, y[n : n + s])

            elif s == 100:
                z = np.matmul(self.R100, y[n : n + s])

            f += w * self.f(T_irregularity(z))

            n += s

        return f


class F9(CECBenchmark):
    r"""20-nonseparable Shifted and Rotated Rastrigin's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{|S|}w_i f_{rastrigin}(z_i)

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F9", 1000, True, True, True, True, False)
    _year = "2013"
    _auxiliary_data = ("o", "R25", "R50", "R100")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F9 benchmark.

        Args:
            args: Positional metadata overrides accepted by CECBenchmark.
            kwargs: Keyword metadata overrides accepted by CECBenchmark.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        super().__init__(*args, **kwargs)

        self.S = [
            50,
            50,
            25,
            25,
            100,
            100,
            25,
            25,
            50,
            25,
            100,
            25,
            100,
            50,
            25,
            25,
            25,
            100,
            50,
            25,
        ]
        self.W = [
            1756.9969,
            570.7338,
            3.3559,
            1.0364,
            62822.2923,
            1.7315,
            0.0898,
            0.0008,
            1403745.6363,
            8716.2083,
            0.0033,
            1.3495,
            0.0047,
            5089.9133,
            12.6664,
            0.0003,
            0.24,
            3.9643,
            0.0014,
            0.0052,
        ]
        self.f = n_dim.Rastrigin()

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F9 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        D = x.shape[0]
        P = np.random.permutation(D)
        n = 0
        f = 0

        y = x - self.o[:D]
        y = y[P]

        for s, w in zip(self.S, self.W):
            if s == 25:
                z = np.matmul(self.R25, y[n : n + s])

            elif s == 50:
                z = np.matmul(self.R50, y[n : n + s])

            elif s == 100:
                z = np.matmul(self.R100, y[n : n + s])

            z = T_asymmetry(T_irregularity(z), 0.2) * _diagonal_weights(z.shape[0], 10)

            f += w * self.f(z)

            n += s

        return f


class F10(CECBenchmark):
    r"""20-nonseparable Shifted and Rotated Ackley's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{|S|}w_i f_{ackley}(z_i)

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-32, 32] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F10", 1000, True, True, True, True, False)
    _year = "2013"
    _auxiliary_data = ("o", "R25", "R50", "R100")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F10 benchmark.

        Args:
            args: Positional metadata overrides accepted by CECBenchmark.
            kwargs: Keyword metadata overrides accepted by CECBenchmark.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        super().__init__(*args, **kwargs)

        self.S = [
            50,
            50,
            25,
            25,
            100,
            100,
            25,
            25,
            50,
            25,
            100,
            25,
            100,
            50,
            25,
            25,
            25,
            100,
            50,
            25,
        ]
        self.W = [
            0.3127,
            15.1277,
            2323.355,
            0.0008,
            11.4208,
            3.5541,
            29.9873,
            0.9981,
            1.6151,
            1.5128,
            0.6084,
            4464853.6323,
            6.8076e-05,
            0.1363,
            0.0007,
            59885.1276,
            1.8523,
            24.7834,
            0.5431,
            39.2404,
        ]
        self.f = n_dim.Ackley1()

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F10 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        D = x.shape[0]
        P = np.random.permutation(D)
        n = 0
        f = 0

        y = x - self.o[:D]
        y = y[P]

        for s, w in zip(self.S, self.W):
            if s == 25:
                z = np.matmul(self.R25, y[n : n + s])

            elif s == 50:
                z = np.matmul(self.R50, y[n : n + s])

            elif s == 100:
                z = np.matmul(self.R100, y[n : n + s])

            z = T_asymmetry(T_irregularity(z), 0.2) * _diagonal_weights(z.shape[0], 10)

            f += w * self.f(z)

            n += s

        return f


class F11(CECBenchmark):
    r"""20-nonseparable Shifted and Rotated Schwefel's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{|S|}w_i f_{schwefel}(z_i)

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F11", 1000, True, True, True, False, False)
    _year = "2013"
    _auxiliary_data = ("o", "R25", "R50", "R100")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F11 benchmark.

        Args:
            args: Positional metadata overrides accepted by CECBenchmark.
            kwargs: Keyword metadata overrides accepted by CECBenchmark.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        super().__init__(*args, **kwargs)

        self.S = [
            50,
            50,
            25,
            25,
            100,
            100,
            25,
            25,
            50,
            25,
            100,
            25,
            100,
            50,
            25,
            25,
            25,
            100,
            50,
            25,
        ]
        self.W = [
            0.0161,
            0.1286,
            0.0012,
            0.3492,
            3.9887,
            7.4469,
            2.6138,
            1.8601e-05,
            0.0779,
            4946500.0392,
            907.5677,
            1245.4389,
            0.0001,
            0.0025,
            0.0122,
            0.2253,
            16011.6801,
            4.1528,
            4208.6086,
            8.983e-06,
        ]
        self.f = n_dim.RotatedHyperEllipsoid()

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F11 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        D = x.shape[0]
        P = np.random.permutation(D)
        n = 0
        f = 0

        y = x - self.o[:D]
        y = y[P]

        for s, w in zip(self.S, self.W):
            if s == 25:
                z = np.matmul(self.R25, y[n : n + s])

            elif s == 50:
                z = np.matmul(self.R50, y[n : n + s])

            elif s == 100:
                z = np.matmul(self.R100, y[n : n + s])

            z = T_asymmetry(T_irregularity(z), 0.2)

            f += w * self.f(z)

            n += s

        return f


class F12(CECBenchmark):
    r"""Shifted Rosenbrock's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-1} (100(z_i^2-z_{i+1})^2 + (z_i - 1)^2) \mid z_i = x_i - o_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o} + 1`.

    """

    _defaults = ("F12", 1000, True, True, True, True, True)
    _year = "2013"
    _auxiliary_data = ("o",)

    @d.check_less_equal_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F12 benchmark.

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

        return f


class F13(CECBenchmark):
    r"""Shifted Schwefel's with Conforming Overlapping benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{|S|}w_i f_{schwefel}(z_i)

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F13", 905, True, True, True, False, False)
    _year = "2013"
    _auxiliary_data = ("o", "R25", "R50", "R100")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F13 benchmark.

        Args:
            args: Positional metadata overrides accepted by CECBenchmark.
            kwargs: Keyword metadata overrides accepted by CECBenchmark.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        super().__init__(*args, **kwargs)

        self.S = [
            50,
            50,
            25,
            25,
            100,
            100,
            25,
            25,
            50,
            25,
            100,
            25,
            100,
            50,
            25,
            25,
            25,
            100,
            50,
            25,
        ]
        self.W = [
            0.4353,
            0.0099,
            0.0542,
            29.3627,
            11490.3303,
            24.1283,
            3.4511,
            2.3264,
            0.0017,
            0.0253,
            19.9959,
            0.0003,
            0.0013,
            0.0387,
            88.8945,
            57901.3138,
            0.0084,
            0.0736,
            0.6883,
            119314.8936,
        ]
        self.C = np.cumsum(self.S)
        self.f = n_dim.RotatedHyperEllipsoid()

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F13 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        D = x.shape[0]
        P = np.random.permutation(D)
        m = 5
        f = 0

        y = x - self.o[:D]
        y = y[P]

        for i, (s, w) in enumerate(zip(self.S, self.W)):
            if i == 0:
                start_n = 0

            else:
                start_n = self.C[i - 1] - i * m

            end_n = self.C[i] - i * m

            if s == 25:
                z = np.matmul(self.R25, y[start_n:end_n])

            elif s == 50:
                z = np.matmul(self.R50, y[start_n:end_n])

            elif s == 100:
                z = np.matmul(self.R100, y[start_n:end_n])

            z = T_asymmetry(T_irregularity(z), 0.2)

            f += w * self.f(z)

        return f


class F14(CECBenchmark):
    r"""Shifted Schwefel's with Conflicting Overlapping benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{|S|}w_i f_{schwefel}(z_i)

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F14", 1000, True, True, True, False, False)
    _year = "2013"
    _auxiliary_data = ("o", "R25", "R50", "R100")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F14 benchmark.

        Args:
            args: Positional metadata overrides accepted by CECBenchmark.
            kwargs: Keyword metadata overrides accepted by CECBenchmark.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        super().__init__(*args, **kwargs)

        self.S = [
            50,
            50,
            25,
            25,
            100,
            100,
            25,
            25,
            50,
            25,
            100,
            25,
            100,
            50,
            25,
            25,
            25,
            100,
            50,
            25,
        ]
        self.W = [
            0.4753,
            498729.4349,
            328.1032,
            0.3231,
            136.4562,
            9.0255,
            0.0924,
            0.0001,
            0.0093,
            299.679,
            4.9395,
            81.3641,
            0.6544,
            11.6119,
            2860774.3201,
            8.5835e-05,
            23.5695,
            0.0481,
            1.4318,
            12.1697,
        ]
        self.C = np.cumsum(self.S)
        self.f = n_dim.RotatedHyperEllipsoid()

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F14 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        D = x.shape[0]
        P = np.random.permutation(D)
        m = 5
        f = 0

        x = x[P]

        for i, (s, w) in enumerate(zip(self.S, self.W)):
            if i == 0:
                start_n = 0
                start_shift = 0

            else:
                start_n = self.C[i - 1] - i * m
                start_shift = self.C[i - 1]

            end_n = self.C[i] - i * m
            end_shift = self.C[i]

            y = x[start_n:end_n] - self.o[start_shift:end_shift]

            if s == 25:
                z = np.matmul(self.R25, y)

            elif s == 50:
                z = np.matmul(self.R50, y)

            elif s == 100:
                z = np.matmul(self.R100, y)

            z = T_asymmetry(T_irregularity(z), 0.2)

            f += w * self.f(z)

        return f


class F15(CECBenchmark):
    r"""Shifted Schwefel's Problem 1.2 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}\sum_{j=1}^{i}z_j^2 \mid z_i = T_{asy}^{0.2}(T_{osz}(x_i - o_i))

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F15", 1000, True, True, True, False, False)
    _year = "2013"
    _auxiliary_data = ("o",)

    @d.check_less_equal_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F15 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        z = T_asymmetry(T_irregularity(x - self.o[: x.shape[0]]), 0.2)

        f = 0

        for i in range(x.shape[0]):
            for j in range(i):
                f += z[j] ** 2

        return f
