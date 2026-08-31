"""One-dimensional benchmarking functions."""

import numpy as np

import opytimark.utils.constants as c
import opytimark.utils.decorator as d
from opytimark.core import Benchmark


class Forrester(Benchmark):
    r"""Forrester class implements the Forrester's benchmarking function.

    .. math:: f(x) = (6x - 2)^2 sin(12x - 4)

    Domain:
        The function is commonly evaluated using :math:`x \in [0, 1]`.

    Global Minima:
        :math:`f(x^*) \\approx -5.9932767166446155 \mid x^* \\approx (0.75)`.

    """

    _defaults = ("Forrester", 1, True, False, True, True, True)

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:

        # Calculating the Forrester's function
        f = (6 * x[0] - 2) ** 2 * np.sin(12 * x[0] - 4)

        return f


class GramacyLee(Benchmark):
    r"""GramacyLee class implements the Gramacy & Lee's benchmarking function.

    .. math:: f(x) = \\frac{sin(10 \\pi x)}{2x} + (x - 1)^4

    Domain:
        The function is commonly evaluated using :math:`x \in [-0.5, 2.5]`.

    Global Minima:
        :math:`f(x^*) = -0.8690111349894997 \mid x^* = (0.548563444114526)`.

    """

    _defaults = ("GramacyLee", 1, True, False, True, False, True)

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:

        # Calculating the Gramacy & Lee's function
        f = np.sin(10 * np.pi * x[0]) / (2 * x[0] + c.EPSILON) + ((x[0] - 1) ** 4)

        return f
