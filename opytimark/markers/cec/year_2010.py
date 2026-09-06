# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from typing import Any

import numpy as np
from numpy.typing import NDArray

import opytimark.markers.n_dimensional as n_dim
import opytimark.utils.decorator as d
import opytimark.utils.exception as e
from opytimark.core import CECBenchmark
from opytimark.logging import get_logger
from opytimark.typing import BenchmarkValue

logger = get_logger(__name__)


def _group_arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any], Any]:
    kwargs = kwargs.copy()

    # Keep the public positional group-size slot when forwarding shared metadata
    if len(args) > 4:
        if "group_size" in kwargs:
            raise TypeError("`group_size` was supplied both positionally and by keyword.")
        group_size = args[4]
        args = args[:4] + args[5:]
    else:
        group_size = kwargs.pop("group_size", 50)

    return args, kwargs, group_size


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
    _year = "2010"
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
    _year = "2010"
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
    _year = "2010"
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

        inv = 1 / x.shape[0]

        term1 = -0.2 * np.sqrt(inv * np.sum(z**2))

        term2 = inv * np.sum(np.cos(2 * np.pi * z))

        f = 20 + np.e - np.exp(term2) - 20 * np.exp(term1)

        return np.sum(f)


class F4(CECBenchmark):
    r"""Single-group Shifted and m-rotated Elliptic's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = f_{rot\_elliptic}[z(P_1:P_m)] * 10^6 + f_{elliptic}[z(P_{m+1}:P_n)] \mid z_i = x_i - o_i, z_i = (x_i - o_i) \ast M_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F4", 1000, True, True, True, False, False)
    _year = "2010"
    _auxiliary_data = ("o", "M")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F4 benchmark.

        Args:
            args: Positional metadata overrides with group_size as the fifth argument.
            kwargs: Keyword metadata and group_size overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        args, kwargs, group_size = _group_arguments(args, kwargs)
        super().__init__(*args, **kwargs)

        self.m = group_size
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

        if self.m >= D:
            raise e.SizeError(f"`group_size` must be less than {D}, but got {self.m!r}.")

        p = np.random.permutation(D)
        p_1 = p[: self.m]
        p_2 = p[self.m :]

        s = x - self.o[:D]

        z_rot = np.dot(s[p_1], self.M[: self.m, : self.m])
        z = s[p_2]

        return self.f(z_rot) * 1e6 + self.f(z)


class F5(CECBenchmark):
    r"""Single-group Shifted and m-rotated Rastrigin's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = f_{rot\_rastrigin}[z(P_1:P_m)] * 10^6 + f_{rastrigin}[z(P_{m+1}:P_n)] \mid z_i = x_i - o_i, z_i = (x_i - o_i) \ast M_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F5", 1000, True, True, True, True, False)
    _year = "2010"
    _auxiliary_data = ("o", "M")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F5 benchmark.

        Args:
            args: Positional metadata overrides with group_size as the fifth argument.
            kwargs: Keyword metadata and group_size overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        args, kwargs, group_size = _group_arguments(args, kwargs)
        super().__init__(*args, **kwargs)

        self.m = group_size
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

        if self.m >= D:
            raise e.SizeError(f"`group_size` must be less than {D}, but got {self.m!r}.")

        p = np.random.permutation(D)
        p_1 = p[: self.m]
        p_2 = p[self.m :]

        s = x - self.o[:D]

        z_rot = np.dot(s[p_1], self.M[: self.m, : self.m])
        z = s[p_2]

        return self.f(z_rot) * 1e6 + self.f(z)


class F6(CECBenchmark):
    r"""Single-group Shifted and m-rotated Ackley's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = f_{rot\_ackley}[z(P_1:P_m)] * 10^6 + f_{ackley}[z(P_{m+1}:P_n)] \mid z_i = x_i - o_i, z_i = (x_i - o_i) \ast M_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-32, 32] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F6", 1000, True, True, True, True, False)
    _year = "2010"
    _auxiliary_data = ("o", "M")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F6 benchmark.

        Args:
            args: Positional metadata overrides with group_size as the fifth argument.
            kwargs: Keyword metadata and group_size overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        args, kwargs, group_size = _group_arguments(args, kwargs)
        super().__init__(*args, **kwargs)

        self.m = group_size
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

        if self.m >= D:
            raise e.SizeError(f"`group_size` must be less than {D}, but got {self.m!r}.")

        p = np.random.permutation(D)
        p_1 = p[: self.m]
        p_2 = p[self.m :]

        s = x - self.o[:D]

        z_rot = np.dot(s[p_1], self.M[: self.m, : self.m])
        z = s[p_2]

        return self.f(z_rot) * 1e6 + self.f(z)


class F7(CECBenchmark):
    r"""Single-group Shifted and m-rotated Schwefel's Problem 1.2 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = f_{schwefel}[z(P_1:P_m)] * 10^6 + f_{sphere}[z(P_{m+1}:P_n)] \mid z_i = x_i - o_i, z_i = (x_i - o_i) \ast M_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F7", 1000, True, True, True, False, False)
    _year = "2010"
    _auxiliary_data = ("o",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F7 benchmark.

        Args:
            args: Positional metadata overrides with group_size as the fifth argument.
            kwargs: Keyword metadata and group_size overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        args, kwargs, group_size = _group_arguments(args, kwargs)
        super().__init__(*args, **kwargs)

        self.m = group_size
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

        if self.m >= D:
            raise e.SizeError(f"`group_size` must be less than {D}, but got {self.m!r}.")

        p = np.random.permutation(D)
        p_1 = p[: self.m]
        p_2 = p[self.m :]

        s = x - self.o[:D]

        z_1 = s[p_1]
        z_2 = s[p_2]

        return self.f_1(z_1) * 1e6 + self.f_2(z_2)


class F8(CECBenchmark):
    r"""Single-group Shifted and m-rotated Rosenbrock's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = f_{rosenbrock}[z(P_1:P_m)] * 10^6 + f_{sphere}[z(P_{m+1}:P_n)] \mid z_i = x_i - o_i, z_i = (x_i - o_i) \ast M_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F8", 1000, True, True, True, True, False)
    _year = "2010"
    _auxiliary_data = ("o",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F8 benchmark.

        Args:
            args: Positional metadata overrides with group_size as the fifth argument.
            kwargs: Keyword metadata and group_size overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        args, kwargs, group_size = _group_arguments(args, kwargs)
        super().__init__(*args, **kwargs)

        self.m = group_size
        self.f_1 = n_dim.Rosenbrock()
        self.f_2 = n_dim.Sphere()

    @d.check_less_equal_dimension
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

        if self.m >= D:
            raise e.SizeError(f"`group_size` must be less than {D}, but got {self.m!r}.")

        p = np.random.permutation(D)
        p_1 = p[: self.m]
        p_2 = p[self.m :]

        s = x - self.o[:D]

        z_1 = s[p_1]
        z_2 = s[p_2]

        return self.f_1(z_1) * 1e6 + self.f_2(z_2)


class F9(CECBenchmark):
    r"""D/2m-group Shifted and m-rotated Elliptic's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{k=1}^{\frac{n}{2m}} f_{rot\_elliptic}[z(P_{(k-1)*m+1}:P_{k*m})] * 10^6 + f_{elliptic}[z(P_{\frac{n}{2}+1}:P_n)] \mid z_i = x_i - o_i, z_i = (x_i - o_i) \ast M_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F9", 1000, True, True, True, False, False)
    _year = "2010"
    _auxiliary_data = ("o", "M")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F9 benchmark.

        Args:
            args: Positional metadata overrides with group_size as the fifth argument.
            kwargs: Keyword metadata and group_size overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        args, kwargs, group_size = _group_arguments(args, kwargs)
        super().__init__(*args, **kwargs)

        self.m = group_size
        self.f = n_dim.HighConditionedElliptic()

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

        D = x.shape[0]
        f = 0
        n_groups = int(D / (2 * self.m))

        if self.m >= D:
            raise e.SizeError(f"`group_size` must be less than {D}, but got {self.m!r}.")

        p = np.random.permutation(D)

        s = x - self.o[:D]

        for i in range(n_groups):
            p_1 = p[i * self.m : (i + 1) * self.m]
            z_rot = np.dot(s[p_1], self.M[: self.m, : self.m])

            f += self.f(z_rot)

        p_2 = p[int(D / 2) :]
        z = s[p_2]

        f += self.f(z)

        return f


class F10(CECBenchmark):
    r"""D/2m-group Shifted and m-rotated Rastrigin's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{k=1}^{\frac{n}{2m}} f_{rot\_rastrigin}[z(P_{(k-1)*m+1}:P_{k*m})] * 10^6 + f_{rastrigin}[z(P_{\frac{n}{2}+1}:P_n)] \mid z_i = x_i - o_i, z_i = (x_i - o_i) \ast M_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F10", 1000, True, True, True, True, False)
    _year = "2010"
    _auxiliary_data = ("o", "M")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F10 benchmark.

        Args:
            args: Positional metadata overrides with group_size as the fifth argument.
            kwargs: Keyword metadata and group_size overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        args, kwargs, group_size = _group_arguments(args, kwargs)
        super().__init__(*args, **kwargs)

        self.m = group_size
        self.f = n_dim.Rastrigin()

    @d.check_less_equal_dimension
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
        f = 0
        n_groups = int(D / (2 * self.m))

        if self.m >= D:
            raise e.SizeError(f"`group_size` must be less than {D}, but got {self.m!r}.")

        p = np.random.permutation(D)

        s = x - self.o[:D]

        for i in range(n_groups):
            p_1 = p[i * self.m : (i + 1) * self.m]
            z_rot = np.dot(s[p_1], self.M[: self.m, : self.m])

            f += self.f(z_rot)

        p_2 = p[int(D / 2) :]
        z = s[p_2]

        f += self.f(z)

        return f


class F11(CECBenchmark):
    r"""D/2m-group Shifted and m-rotated Ackley's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{k=1}^{\frac{n}{2m}} f_{rot\_ackley}[z(P_{(k-1)*m+1}:P_{k*m})] * 10^6 + f_{ackley}[z(P_{\frac{n}{2}+1}:P_n)] \mid z_i = x_i - o_i, z_i = (x_i - o_i) \ast M_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-32, 32] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F11", 1000, True, True, True, True, False)
    _year = "2010"
    _auxiliary_data = ("o", "M")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F11 benchmark.

        Args:
            args: Positional metadata overrides with group_size as the fifth argument.
            kwargs: Keyword metadata and group_size overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        args, kwargs, group_size = _group_arguments(args, kwargs)
        super().__init__(*args, **kwargs)

        self.m = group_size
        self.f = n_dim.Ackley1()

    @d.check_less_equal_dimension
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
        f = 0
        n_groups = int(D / (2 * self.m))

        if self.m >= D:
            raise e.SizeError(f"`group_size` must be less than {D}, but got {self.m!r}.")

        p = np.random.permutation(D)

        s = x - self.o[:D]

        for i in range(n_groups):
            p_1 = p[i * self.m : (i + 1) * self.m]
            z_rot = np.dot(s[p_1], self.M[: self.m, : self.m])

            f += self.f(z_rot)

        p_2 = p[int(D / 2) :]
        z = s[p_2]

        f += self.f(z)

        return f


class F12(CECBenchmark):
    r"""D/2m-group Shifted and m-rotated Schwefel's Problem 1.2 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{k=1}^{\frac{n}{2m}} f_{schwefel}[z(P_{(k-1)*m+1}:P_{k*m})] * 10^6 + f_{sphere}[z(P_{\frac{n}{2}+1}:P_n)] \mid z_i = x_i - o_i, z_i = (x_i - o_i) \ast M_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F12", 1000, True, True, True, False, False)
    _year = "2010"
    _auxiliary_data = ("o",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F12 benchmark.

        Args:
            args: Positional metadata overrides with group_size as the fifth argument.
            kwargs: Keyword metadata and group_size overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        args, kwargs, group_size = _group_arguments(args, kwargs)
        super().__init__(*args, **kwargs)

        self.m = group_size
        self.f_1 = n_dim.RotatedHyperEllipsoid()
        self.f_2 = n_dim.Sphere()

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

        D = x.shape[0]
        f = 0
        n_groups = int(D / (2 * self.m))

        if self.m >= D:
            raise e.SizeError(f"`group_size` must be less than {D}, but got {self.m!r}.")

        p = np.random.permutation(D)

        s = x - self.o[:D]

        for i in range(n_groups):
            p_1 = p[i * self.m : (i + 1) * self.m]
            z_1 = s[p_1]

            f += self.f_1(z_1)

        p_2 = p[int(D / 2) :]
        z_2 = s[p_2]

        f += self.f_2(z_2)

        return f


class F13(CECBenchmark):
    r"""D/2m-group Shifted and m-rotated Rosenbrock benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{k=1}^{\frac{n}{2m}} f_{rosenbrock}[z(P_{(k-1)*m+1}:P_{k*m})] * 10^6 + f_{sphere}[z(P_{\frac{n}{2}+1}:P_n)] \mid z_i = x_i - o_i, z_i = (x_i - o_i) \ast M_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F13", 1000, True, True, True, True, False)
    _year = "2010"
    _auxiliary_data = ("o",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F13 benchmark.

        Args:
            args: Positional metadata overrides with group_size as the fifth argument.
            kwargs: Keyword metadata and group_size overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        args, kwargs, group_size = _group_arguments(args, kwargs)
        super().__init__(*args, **kwargs)

        self.m = group_size
        self.f_1 = n_dim.Rosenbrock()
        self.f_2 = n_dim.Sphere()

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

        D = x.shape[0]
        f = 0
        n_groups = int(D / (2 * self.m))

        if self.m >= D:
            raise e.SizeError(f"`group_size` must be less than {D}, but got {self.m!r}.")

        p = np.random.permutation(D)

        s = x - self.o[:D]

        for i in range(n_groups):
            p_1 = p[i * self.m : (i + 1) * self.m]
            z_1 = s[p_1]

            f += self.f_1(z_1)

        p_2 = p[int(D / 2) :]
        z_2 = s[p_2]

        f += self.f_2(z_2)

        return f


class F14(CECBenchmark):
    r"""D/m-group Shifted and m-rotated Elliptic's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{k=1}^{\frac{n}{m}} f_{rot\_elliptic}[z(P_{(k-1)*m+1}:P_{k*m})] \mid z_i = (x_i - o_i) \ast M_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F14", 1000, True, True, True, False, False)
    _year = "2010"
    _auxiliary_data = ("o", "M")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F14 benchmark.

        Args:
            args: Positional metadata overrides with group_size as the fifth argument.
            kwargs: Keyword metadata and group_size overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        args, kwargs, group_size = _group_arguments(args, kwargs)
        super().__init__(*args, **kwargs)

        self.m = group_size
        self.f = n_dim.HighConditionedElliptic()

    @d.check_less_equal_dimension
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
        f = 0
        n_groups = int(D / (self.m))

        if self.m >= D:
            raise e.SizeError(f"`group_size` must be less than {D}, but got {self.m!r}.")

        P = np.random.permutation(D)

        s = x - self.o[:D]

        for i in range(n_groups):
            p = P[i * self.m : (i + 1) * self.m]
            z = np.dot(s[p], self.M[: self.m, : self.m])

            f += self.f(z)

        return f


class F15(CECBenchmark):
    r"""D/m-group Shifted and m-rotated Rastrigin's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{k=1}^{\frac{n}{m}} f_{rot\_elliptic}[z(P_{(k-1)*m+1}:P_{k*m})] \mid z_i = (x_i - o_i) \ast M_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F15", 1000, True, True, True, True, False)
    _year = "2010"
    _auxiliary_data = ("o", "M")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F15 benchmark.

        Args:
            args: Positional metadata overrides with group_size as the fifth argument.
            kwargs: Keyword metadata and group_size overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        args, kwargs, group_size = _group_arguments(args, kwargs)
        super().__init__(*args, **kwargs)

        self.m = group_size
        self.f = n_dim.Rastrigin()

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

        D = x.shape[0]
        f = 0
        n_groups = int(D / (self.m))

        if self.m >= D:
            raise e.SizeError(f"`group_size` must be less than {D}, but got {self.m!r}.")

        P = np.random.permutation(D)

        s = x - self.o[:D]

        for i in range(n_groups):
            p = P[i * self.m : (i + 1) * self.m]
            z = np.dot(s[p], self.M[: self.m, : self.m])

            f += self.f(z)

        return f


class F16(CECBenchmark):
    r"""D/m-group Shifted and m-rotated Ackley's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{k=1}^{\frac{n}{m}} f_{rot\_elliptic}[z(P_{(k-1)*m+1}:P_{k*m})] \mid z_i = (x_i - o_i) \ast M_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-32, 32] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F16", 1000, True, True, True, True, False)
    _year = "2010"
    _auxiliary_data = ("o", "M")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F16 benchmark.

        Args:
            args: Positional metadata overrides with group_size as the fifth argument.
            kwargs: Keyword metadata and group_size overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        args, kwargs, group_size = _group_arguments(args, kwargs)
        super().__init__(*args, **kwargs)

        self.m = group_size
        self.f = n_dim.Ackley1()

    @d.check_less_equal_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F16 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        D = x.shape[0]
        f = 0
        n_groups = int(D / (self.m))

        if self.m >= D:
            raise e.SizeError(f"`group_size` must be less than {D}, but got {self.m!r}.")

        P = np.random.permutation(D)

        s = x - self.o[:D]

        for i in range(n_groups):
            p = P[i * self.m : (i + 1) * self.m]
            z = np.dot(s[p], self.M[: self.m, : self.m])

            f += self.f(z)

        return f


class F17(CECBenchmark):
    r"""D/m-group Shifted Schwefel's Problem 1.2 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{k=1}^{\frac{n}{m}} f_{rot\_elliptic}[z(P_{(k-1)*m+1}:P_{k*m})] \mid z_i = x_i - o_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F17", 1000, True, True, True, False, False)
    _year = "2010"
    _auxiliary_data = ("o",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F17 benchmark.

        Args:
            args: Positional metadata overrides with group_size as the fifth argument.
            kwargs: Keyword metadata and group_size overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        args, kwargs, group_size = _group_arguments(args, kwargs)
        super().__init__(*args, **kwargs)

        self.m = group_size
        self.f = n_dim.RotatedHyperEllipsoid()

    @d.check_less_equal_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F17 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        D = x.shape[0]
        f = 0
        n_groups = int(D / (self.m))

        if self.m >= D:
            raise e.SizeError(f"`group_size` must be less than {D}, but got {self.m!r}.")

        P = np.random.permutation(D)

        s = x - self.o[:D]

        for i in range(n_groups):
            p = P[i * self.m : (i + 1) * self.m]
            z = s[p]

            f += self.f(z)

        return f


class F18(CECBenchmark):
    r"""D/m-group Shifted Rosenbrock's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{k=1}^{\frac{n}{m}} f_{rot\_elliptic}[z(P_{(k-1)*m+1}:P_{k*m})] \mid z_i = x_i - o_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o} + 1`.

    """

    _defaults = ("F18", 1000, True, True, True, True, False)
    _year = "2010"
    _auxiliary_data = ("o",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the F18 benchmark.

        Args:
            args: Positional metadata overrides with group_size as the fifth argument.
            kwargs: Keyword metadata and group_size overrides.

        Raises:
            TypeError: A metadata value has an unsupported type.
            ValueError: The declared dimension is invalid.
            OSError: Required auxiliary data cannot be loaded.

        """

        args, kwargs, group_size = _group_arguments(args, kwargs)
        super().__init__(*args, **kwargs)

        self.m = group_size
        self.f = n_dim.Rosenbrock()

    @d.check_less_equal_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F18 benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        D = x.shape[0]
        f = 0
        n_groups = int(D / (self.m))

        if self.m >= D:
            raise e.SizeError(f"`group_size` must be less than {D}, but got {self.m!r}.")

        P = np.random.permutation(D)

        s = x - self.o[:D]

        logger.debug("Shifted coordinates %s with permutation %s", s, P)

        for i in range(n_groups):
            p = P[i * self.m : (i + 1) * self.m]
            z = s[p]

            f += self.f(z)

        return f


class F19(CECBenchmark):
    r"""Shifted Schwefel's Problem 1.2 benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}\sum_{j=1}^{i}z_j^2 \mid z_i = x_i - o_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    _defaults = ("F19", 1000, True, True, True, False, False)
    _year = "2010"
    _auxiliary_data = ("o",)

    @d.check_less_equal_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F19 benchmark.

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
            for j in range(i):
                f += z[j] ** 2

        return f


class F20(CECBenchmark):
    r"""Shifted Rosenbrock's benchmark.

    Notes:
        .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-1} (100(z_i^2-z_{i+1})^2 + (z_i - 1)^2) \mid z_i = x_i - o_i

        Domain:
            The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

        Global Minima:
            :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o} + 1`.

    """

    _defaults = ("F20", 1000, True, True, True, True, False)
    _year = "2010"
    _auxiliary_data = ("o",)

    @d.check_less_equal_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the F20 benchmark.

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
