# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from typing import Any

import numpy as np
from numpy.typing import NDArray

import opytimark.markers.n_dimensional as n_dim
import opytimark.utils.decorator as d
from opytimark.core import CECBenchmark, CECCompositeBenchmark
from opytimark.typing import BenchmarkValue


def _composite_arguments(
    args: tuple[Any, ...], kwargs: dict[str, Any], default_bias: Any
) -> tuple[tuple[Any, ...], dict[str, Any], Any]:
    kwargs = kwargs.copy()

    # Keep the public positional bias slot when forwarding shared metadata
    if len(args) > 3:
        if "bias" in kwargs:
            raise TypeError("`bias` was supplied both positionally and by keyword.")
        bias = args[3]
        args = args[:3] + args[4:]
    else:
        bias = kwargs.pop("bias", default_bias)

    return args, kwargs, bias


class F1(CECBenchmark):
    r"""Shifted Sphere's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} z_i^2 - 450 \mid z_i = x_i - o_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -450 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F1", 100, True, True, True, False, True)
    _year = "2005"
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
    r"""Shifted Schwefel's 1.2 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (\sum_{j=1}^i z_j)^2 - 450 \mid z_i = x_i - o_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -450 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F2", 100, True, True, True, False, False)
    _year = "2005"
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

        f = 0

        for i in range(x.shape[0]):
            partial = 0

            for j in range(i):
                partial += z[j]

            f += partial**2

        return f - 450


class F3(CECBenchmark):
    r"""Shifted Rotated High Conditioned Elliptic's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (10^6)^\frac{i-1}{n-1} z_i^2 - 450 \mid z_i = (x_i - o_i) * M_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \in \{2, 10, 30, 50\}`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -450 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F3", -1, True, True, True, False, False)
    _year = "2005"
    _auxiliary_data = ("o", "M2", "M10", "M30", "M50")

    @d.check_exact_dimension_and_auxiliary_matrix
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F3 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        z = np.matmul(x - self.o[: x.shape[0]], self.M)
        return n_dim._elliptic(z) - 450


class F4(CECBenchmark):
    r"""Shifted Schwefel's 1.2 with Noise in Fitness benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (\sum_{j=1}^i z_j)^2 * (1 + 0.4|N(0,1)|) - 450 \mid z_i = x_i - o_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -450 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F4", 100, True, True, True, False, False)
    _year = "2005"
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

        f = 0

        for i in range(x.shape[0]):
            partial = 0

            for j in range(i):
                partial += z[j]

            f += partial**2

        noise = np.random.uniform()

        return f * (1 + 0.4 * noise) - 450


class F5(CECBenchmark):
    r"""Schwefel's Problem 2.6 with Global Optimum on Bounds benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \max{|A_i x - B_i|} - 310 \mid B_i = A_i - o_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -310 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F5", 100, True, True, True, False, False)
    _year = "2005"
    _auxiliary_data = ("o", "A")

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

        shift_1 = int(x.shape[0] / 4)
        shift_2 = int(3 * x.shape[0] / 4)

        self.o[:shift_1] = -100
        self.o[shift_2:] = 100

        A = self.A[: x.shape[0], : x.shape[0]]

        B = np.matmul(A, self.o[: x.shape[0]])

        f = np.max(np.fabs(np.matmul(A, x) - B))

        return f - 310


class F6(CECBenchmark):
    r"""Shifted Rosenbrock's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-1} (100(z_i^2-z_{i+1})^2 + (z_i - 1)^2) + 390 \mid z_i = x_i - o_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -390 \mid \mathbf{x^*} = \mathbf{o} + 1`.

    """

    _defaults = ("F6", 100, True, True, True, True, False)
    _year = "2005"
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

        f = 0

        for i in range(x.shape[0] - 1):
            f += 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2

        return f + 390


class F7(CECBenchmark):
    r"""Shifted Rotated Griewank's without Bounds benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = 1 + \sum_{i=1}^{n}\frac{x_i^2}{4000} - \prod cos(\frac{x_i}{\sqrt{i}}) - 180 \mid z_i = (x_i - o_i) * M_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [0, 600] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -180 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F7", -1, True, True, True, True, False)
    _year = "2005"
    _auxiliary_data = ("o", "M2", "M10", "M30", "M50")

    @d.check_exact_dimension_and_auxiliary_matrix
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F7 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        z = np.matmul(x - self.o[: x.shape[0]], self.M)

        term1, term2 = 0, 1

        for i in range(x.shape[0]):
            term1 += (z[i] ** 2) / 4000

            term2 *= np.cos(z[i] / np.sqrt(i + 1))

        f = 1 + term1 - term2

        return f - 180


class F8(CECBenchmark):
    r"""Shifted Rotated Ackley's with Global Optimum on Bounds benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = -20e^{-0.2\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2}}-e^{\frac{1}{n}\sum_{i=1}^{n}cos(2 \pi x_i)}+ 20 + e - 140 \mid z_i = (x_i - o_i) * M_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-32, 32] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -140 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F8", -1, True, True, True, True, False)
    _year = "2005"
    _auxiliary_data = ("o", "M2", "M10", "M30", "M50")

    @d.check_exact_dimension_and_auxiliary_matrix
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F8 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        shift = int(x.shape[0] / 2)

        for j in range(shift):
            self.o[2 * j] = -32 * self.o[2 * j + 1]

        z = np.matmul(x - self.o[: x.shape[0]], self.M)

        inv = 1 / x.shape[0]

        term1 = -0.2 * np.sqrt(inv * np.sum(z**2))

        term2 = inv * np.sum(np.cos(2 * np.pi * z))

        f = 20 + np.e - np.exp(term2) - 20 * np.exp(term1)

        return np.sum(f) - 140


class F9(CECBenchmark):
    r"""Shifted Rastrigin's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (z_i^2 - 10cos(2 \pi z_i) + 10) - 330 \mid z_i = x_i - o_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -330 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F9", 100, True, True, True, True, True)
    _year = "2005"
    _auxiliary_data = ("o",)

    @d.check_less_equal_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F9 benchmark.

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


class F10(CECBenchmark):
    r"""Shifted Rotated Rastrigin's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (z_i^2 - 10cos(2 \pi z_i) + 10) - 330 \mid z_i = (x_i - o_i) * M_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -330 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F10", 100, True, True, True, True, False)
    _year = "2005"
    _auxiliary_data = ("o", "M2", "M10", "M30", "M50")

    @d.check_exact_dimension_and_auxiliary_matrix
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F10 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        z = np.matmul(x - self.o[: x.shape[0]], self.M)

        f = z**2 - 10 * np.cos(2 * np.pi * z) + 10

        return np.sum(f) - 330


class F11(CECBenchmark):
    r"""Shifted Rotated Weierstrass's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (\sum_{k=0}^{20} [0.5^k cos(2\pi 3^k(z_i+0.5))]) - n \sum_{k=0}^{20}[0.5^k cos(2\pi 3^k 0.5)] \mid z_i = (x_i - o_i) * M_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-0.5, 0.5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 90 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F11", 100, True, True, True, True, False)
    _year = "2005"
    _auxiliary_data = ("o", "M2", "M10", "M30", "M50")

    @d.check_exact_dimension_and_auxiliary_matrix
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F11 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        z = np.matmul(x - self.o[: x.shape[0]], self.M)

        f = 0

        for i in range(x.shape[0]):
            for k in range(21):
                f += 0.5**k * np.cos(2 * np.pi * 3**k * (z[i] + 0.5))

        for k in range(21):
            f -= x.shape[0] * (0.5**k * np.cos(2 * np.pi * 3**k * 0.5))

        return f + 90


class F12(CECBenchmark):
    r"""Schwefel's Problem 2.13 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (A_i - B_i)^2 - 460 \mid A_i = \sum_{j=1}^{n} a_{ij} sin(\alpha_j) + b_{ij} cos(\alpha_j), A_i = \sum_{j=1}^{n} a_{ij} sin(x_j) + b_{ij} cos(x_j)

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-\pi, \pi] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -460 \mid \mathbf{x^*} = \mathbf{\alpha}`.

    """

    _defaults = ("F12", 100, True, True, True, True, False)
    _year = "2005"
    _auxiliary_data = ("alpha", "a", "b")

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

        alpha = self.alpha[: x.shape[0]]
        a = self.a[: x.shape[0], : x.shape[0]]
        b = self.b[: x.shape[0], : x.shape[0]]

        A = a * np.sin(alpha) + b * np.cos(alpha)
        B = a * np.sin(x) + b * np.cos(x)

        f = (A - B) ** 2

        return np.sum(f) - 460


class F13(CECBenchmark):
    r"""Shifted Expanded Griewank's plus Rosenbrock's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) =  f(x_1, x_2) + f(x_2, x_3) + \ldots + f(x_n, x_1) - 130 \mid z_i = x_i - o_i + 1

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-3, 1] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -130 \mid \mathbf{x^*} = \mathbf{\alpha}`.

    """

    _defaults = ("F13", 100, True, True, True, True, False)
    _year = "2005"
    _auxiliary_data = ("o",)

    @d.check_less_equal_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F13 benchmark.

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

        z = x - self.o[: x.shape[0]] + 1

        f = 0

        for i in range(x.shape[0]):
            if i == (x.shape[0] - 1):
                f += _griewank(_rosenbrock(z[i], z[0]))

            else:
                f += _griewank(_rosenbrock(z[i], z[i + 1]))

        return f - 130


class F14(CECBenchmark):
    r"""Shifted Rotated Expanded Scaffer's F6 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) =  f(x_1, x_2) + f(x_2, x_3) + \ldots + f(x_n, x_1) - 300 \mid z_i = x_i - o_i + 1

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = -300 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F14", 100, True, True, True, True, False)
    _year = "2005"
    _auxiliary_data = ("o", "M2", "M10", "M30", "M50")

    @d.check_exact_dimension_and_auxiliary_matrix
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F14 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        def _scaffer(x, y):
            return 0.5 + (np.sin(np.sqrt(x**2 + y**2)) ** 2 - 0.5) / ((1 + 0.0001 * (x**2 + y**2)) ** 2)

        z = np.matmul(x - self.o[: x.shape[0]], self.M)

        f = 0

        for i in range(x.shape[0]):
            if i == (x.shape[0] - 1):
                f += _scaffer(z[i], z[0])

            else:
                f += _scaffer(z[i], z[i + 1])

        return f - 300


class F15(CECCompositeBenchmark):
    r"""Hybrid Composition 1 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}{w_i \ast [f_i'((\mathbf{x}-\mathbf{o_i})/ \lambda_i \ast \mathbf{M_i}) + bias_i]} + f_{bias}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 120 \mid \mathbf{x^*} = \mathbf{o_1}`.

    """

    _defaults = ("F15", 100, True, True, True, True, True)
    _year = "2005"
    _auxiliary_data = ("o", "M2", "M10", "M30", "M50")
    _bias = 120

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F15 benchmark.

        Args:
            args: Positional metadata overrides with bias as the fourth argument.
            kwargs: Keyword metadata and bias overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        sigma = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        l = (1, 1, 10, 10, 5 / 60, 5 / 60, 5 / 32, 5 / 32, 5 / 100, 5 / 100)
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

        args, kwargs, bias = _composite_arguments(args, kwargs, self._bias)
        CECBenchmark.__init__(self, *args, **kwargs)

        self._initialize_composition(sigma, l, functions, bias)


class F16(CECCompositeBenchmark):
    r"""Rotated Hybrid Composition 1 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}{w_i \ast [f_i'((\mathbf{x}-\mathbf{o_i})/ \lambda_i \ast \mathbf{M_i}) + bias_i]} + f_{bias}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 120 \mid \mathbf{x^*} = \mathbf{o_1}`.

    """

    _defaults = ("F16", 100, True, True, True, True, False)
    _year = "2005"
    _auxiliary_data = ("o", "M2", "M10", "M30", "M50")
    _bias = 120

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F16 benchmark.

        Args:
            args: Positional metadata overrides with bias as the fourth argument.
            kwargs: Keyword metadata and bias overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        sigma = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        l = (1, 1, 10, 10, 5 / 60, 5 / 60, 5 / 32, 5 / 32, 5 / 100, 5 / 100)
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

        args, kwargs, bias = _composite_arguments(args, kwargs, self._bias)
        CECBenchmark.__init__(self, *args, **kwargs)

        self._initialize_composition(sigma, l, functions, bias)


class F17(CECCompositeBenchmark):
    r"""Rotated Hybrid Composition 1 with Noise benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}{w_i \ast [f_i'((\mathbf{x}-\mathbf{o_i})/ \lambda_i \ast \mathbf{M_i}) + bias_i]} + f_{bias}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 120 \mid \mathbf{x^*} = \mathbf{o_1}`.

    """

    _defaults = ("F17", 100, True, True, True, True, False)
    _year = "2005"
    _auxiliary_data = ("o", "M2", "M10", "M30", "M50")
    _bias = 120

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F17 benchmark.

        Args:
            args: Positional metadata overrides with bias as the fourth argument.
            kwargs: Keyword metadata and bias overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        sigma = (1, 2, 1.5, 1.5, 1, 1, 1.5, 1.5, 2, 2)
        l = (5 / 16, 5 / 32, 2, 1, 1 / 10, 1 / 20, 20, 10, 1 / 6, 5 / 60)
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

        args, kwargs, bias = _composite_arguments(args, kwargs, self._bias)
        CECBenchmark.__init__(self, *args, **kwargs)

        self._initialize_composition(sigma, l, functions, bias)

    @d.check_exact_dimension_and_auxiliary_matrix
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F17 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        g = self._evaluate_composition(x)
        return g * (1 + 0.2 * np.fabs(np.random.normal())) + self.bias


class F18(CECCompositeBenchmark):
    r"""Rotated Hybrid Composition 2 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}{w_i \ast [f_i'((\mathbf{x}-\mathbf{o_i})/ \lambda_i \ast \mathbf{M_i}) + bias_i]} + f_{bias}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 10 \mid \mathbf{x^*} = \mathbf{o_1}`.

    """

    _defaults = ("F18", 100, True, True, True, True, False)
    _year = "2005"
    _auxiliary_data = ("o", "M2", "M10", "M30", "M50")
    _bias = 10

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F18 benchmark.

        Args:
            args: Positional metadata overrides with bias as the fourth argument.
            kwargs: Keyword metadata and bias overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        sigma = (1, 2, 1.5, 1.5, 1, 1, 1.5, 1.5, 2, 2)
        l = (5 / 16, 5 / 32, 2, 1, 1 / 10, 1 / 20, 20, 10, 1 / 6, 5 / 60)
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

        args, kwargs, bias = _composite_arguments(args, kwargs, self._bias)
        CECBenchmark.__init__(self, *args, **kwargs)

        self._initialize_composition(sigma, l, functions, bias)


class F19(CECCompositeBenchmark):
    r"""Rotated Hybrid Composition 2 with Narrow Basin Global Optimum benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}{w_i \ast [f_i'((\mathbf{x}-\mathbf{o_i})/ \lambda_i \ast \mathbf{M_i}) + bias_i]} + f_{bias}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 10 \mid \mathbf{x^*} = \mathbf{o_1}`.

    """

    _defaults = ("F19", 100, True, True, True, True, False)
    _year = "2005"
    _auxiliary_data = ("o", "M2", "M10", "M30", "M50")
    _bias = 10

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F19 benchmark.

        Args:
            args: Positional metadata overrides with bias as the fourth argument.
            kwargs: Keyword metadata and bias overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        sigma = (0.1, 2, 1.5, 1.5, 1, 1, 1.5, 1.5, 2, 2)
        l = (0.1 * 5 / 32, 5 / 32, 2, 1, 1 / 10, 1 / 20, 20, 10, 1 / 6, 5 / 60)
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

        args, kwargs, bias = _composite_arguments(args, kwargs, self._bias)
        CECBenchmark.__init__(self, *args, **kwargs)

        self._initialize_composition(sigma, l, functions, bias)


class F20(CECCompositeBenchmark):
    r"""Rotated Hybrid Composition 2 with Global Optimum on the Bounds benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}{w_i \ast [f_i'((\mathbf{x}-\mathbf{o_i})/ \lambda_i \ast \mathbf{M_i}) + bias_i]} + f_{bias}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 10 \mid \mathbf{x^*} = \mathbf{o_1}`.

    """

    _defaults = ("F20", 100, True, True, True, True, False)
    _year = "2005"
    _auxiliary_data = ("o", "M2", "M10", "M30", "M50")
    _bias = 10

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F20 benchmark.

        Args:
            args: Positional metadata overrides with bias as the fourth argument.
            kwargs: Keyword metadata and bias overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        sigma = (0.1, 2, 1.5, 1.5, 1, 1, 1.5, 1.5, 2, 2)
        l = (0.1 * 5 / 32, 5 / 32, 2, 1, 1 / 10, 1 / 20, 20, 10, 1 / 6, 5 / 60)
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

        args, kwargs, bias = _composite_arguments(args, kwargs, self._bias)
        CECBenchmark.__init__(self, *args, **kwargs)

        self._initialize_composition(sigma, l, functions, bias)

    @d.check_exact_dimension_and_auxiliary_matrix
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F20 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        for j in range(x.shape[0] // 2):
            self.o[0][2 * j + 1] = 5

        return self._evaluate_composition(x) + self.bias


class F21(CECCompositeBenchmark):
    r"""Rotated Hybrid Composition 3 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}{w_i \ast [f_i'((\mathbf{x}-\mathbf{o_i})/ \lambda_i \ast \mathbf{M_i}) + bias_i]} + f_{bias}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 360 \mid \mathbf{x^*} = \mathbf{o_1}`.

    """

    _defaults = ("F21", 100, True, True, True, True, False)
    _year = "2005"
    _auxiliary_data = ("o", "M2", "M10", "M30", "M50")
    _bias = 360

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F21 benchmark.

        Args:
            args: Positional metadata overrides with bias as the fourth argument.
            kwargs: Keyword metadata and bias overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        sigma = (1, 1, 1, 1, 1, 2, 2, 2, 2, 2)
        l = (1 / 4, 5 / 100, 5, 1, 5, 1, 50, 10, 1 / 8, 5 / 200)
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

        args, kwargs, bias = _composite_arguments(args, kwargs, self._bias)
        CECBenchmark.__init__(self, *args, **kwargs)

        self._initialize_composition(sigma, l, functions, bias)


class F22(CECCompositeBenchmark):
    r"""Rotated Hybrid Composition 3 with High Condition Number Matrix benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}{w_i \ast [f_i'((\mathbf{x}-\mathbf{o_i})/ \lambda_i \ast \mathbf{M_i}) + bias_i]} + f_{bias}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 360 \mid \mathbf{x^*} = \mathbf{o_1}`.

    """

    _defaults = ("F22", 100, True, True, True, True, False)
    _year = "2005"
    _auxiliary_data = ("o", "M2", "M10", "M30", "M50")
    _bias = 360

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F22 benchmark.

        Args:
            args: Positional metadata overrides with bias as the fourth argument.
            kwargs: Keyword metadata and bias overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        sigma = (1, 1, 1, 1, 1, 2, 2, 2, 2, 2)
        l = (1 / 4, 5 / 100, 5, 1, 5, 1, 50, 10, 1 / 8, 5 / 200)
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

        args, kwargs, bias = _composite_arguments(args, kwargs, self._bias)
        CECBenchmark.__init__(self, *args, **kwargs)

        self._initialize_composition(sigma, l, functions, bias)


class F23(CECCompositeBenchmark):
    r"""Non-Continuous Rotated Hybrid Composition 3 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}{w_i \ast [f_i'((\mathbf{x}-\mathbf{o_i})/ \lambda_i \ast \mathbf{M_i}) + bias_i]} + f_{bias}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) \approx 360 \mid \mathbf{x^*} = \mathbf{o_1}`.

    """

    _defaults = ("F23", 100, False, True, False, True, False)
    _year = "2005"
    _auxiliary_data = ("o", "M2", "M10", "M30", "M50")
    _bias = 360

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F23 benchmark.

        Args:
            args: Positional metadata overrides with bias as the fourth argument.
            kwargs: Keyword metadata and bias overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        sigma = (1, 1, 1, 1, 1, 2, 2, 2, 2, 2)
        l = (1 / 4, 5 / 100, 5, 1, 5, 1, 50, 10, 1 / 8, 5 / 200)
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

        args, kwargs, bias = _composite_arguments(args, kwargs, self._bias)
        CECBenchmark.__init__(self, *args, **kwargs)

        self._initialize_composition(sigma, l, functions, bias)

    @d.check_exact_dimension_and_auxiliary_matrix
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F23 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        D = x.shape[0]
        x = np.where(np.fabs(x - self.o[0][:D]) < 0.5, x, np.round(2 * x) / 2)
        return self._evaluate_composition(x) + self.bias


class F24(CECCompositeBenchmark):
    r"""Rotated Hybrid Composition 4 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}{w_i \ast [f_i'((\mathbf{x}-\mathbf{o_i})/ \lambda_i \ast \mathbf{M_i}) + bias_i]} + f_{bias}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 260 \mid \mathbf{x^*} = \mathbf{o_1}`.

    """

    _defaults = ("F24", 100, True, True, True, True, False)
    _year = "2005"
    _auxiliary_data = ("o", "M2", "M10", "M30", "M50")
    _bias = 260

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F24 benchmark.

        Args:
            args: Positional metadata overrides with bias as the fourth argument.
            kwargs: Keyword metadata and bias overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        sigma = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
        l = (10, 1 / 4, 1, 5 / 32, 1, 5 / 100, 1 / 10, 1, 5 / 100, 5 / 100)
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

        args, kwargs, bias = _composite_arguments(args, kwargs, self._bias)
        CECBenchmark.__init__(self, *args, **kwargs)

        self._initialize_composition(sigma, l, functions, bias)


class F25(CECCompositeBenchmark):
    r"""Rotated Hybrid Composition 4 without Bounds benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}{w_i \ast [f_i'((\mathbf{x}-\mathbf{o_i})/ \lambda_i \ast \mathbf{M_i}) + bias_i]} + f_{bias}

        Domain:
            The function is commonly evaluated using :math:`x_i \in [?, ?] \mid i = \{1, 2, \ldots, n\}, n \leq 100`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 260 \mid \mathbf{x^*} = \mathbf{o_1}`.

    """

    _defaults = ("F25", 100, True, True, True, True, False)
    _year = "2005"
    _auxiliary_data = ("o", "M2", "M10", "M30", "M50")
    _bias = 260

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F25 benchmark.

        Args:
            args: Positional metadata overrides with bias as the fourth argument.
            kwargs: Keyword metadata and bias overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        sigma = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
        l = (10, 1 / 4, 1, 5 / 32, 1, 5 / 100, 1 / 10, 1, 5 / 100, 5 / 100)
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

        args, kwargs, bias = _composite_arguments(args, kwargs, self._bias)
        CECBenchmark.__init__(self, *args, **kwargs)

        self._initialize_composition(sigma, l, functions, bias)
