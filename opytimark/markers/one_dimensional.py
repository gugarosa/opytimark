# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from typing import Any

import numpy as np
from numpy.typing import NDArray

import opytimark.utils.constants as c
import opytimark.utils.decorator as d
from opytimark.core import Benchmark
from opytimark.typing import BenchmarkValue


class Forrester(Benchmark):
    r"""Forrester's benchmark.

    Notes:
        .. math:: f(x) = (6x - 2)^2 sin(12x - 4)

        Domain:
            The function is commonly evaluated using :math:`x \in [0, 1]`.

        Global Minima:
            :math:`f(x^*) \approx -5.9932767166446155 \mid x^* \approx (0.75)`.

    """

    _defaults = ("Forrester", 1, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the Forrester benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = (6 * x[0] - 2) ** 2 * np.sin(12 * x[0] - 4)

        return f


class GramacyLee(Benchmark):
    r"""Gramacy & Lee's benchmark.

    Notes:
        .. math:: f(x) = \frac{sin(10 \pi x)}{2x} + (x - 1)^4

        Domain:
            The function is commonly evaluated using :math:`x \in [-0.5, 2.5]`.

        Global Minima:
            :math:`f(x^*) = -0.8690111349894997 \mid x^* = (0.548563444114526)`.

    """

    _defaults = ("GramacyLee", 1, True, False, True, False, True)

    @d.check_exact_dimension
    def __call__(self, x: NDArray[Any]) -> BenchmarkValue:
        """Evaluate the GramacyLee benchmark.

        Args:
            x: Numeric coordinates supplied as a vector or single-column array.

        Returns:
            Computed benchmark value with its existing Python or NumPy type.

        Raises:
            SizeError: The input or group dimensions are unsupported.

        """

        f = np.sin(10 * np.pi * x[0]) / (2 * x[0] + c.EPSILON) + ((x[0] - 1) ** 4)

        return f
