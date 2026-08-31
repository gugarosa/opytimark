"""Base classes for CEC benchmark functions."""

import numpy as np

import opytimark.utils.decorator as d
import opytimark.utils.exception as e
import opytimark.utils.loader as ld
from opytimark.core.benchmark import _MISSING, Benchmark


class CECBenchmark(Benchmark):
    """Base class for CEC benchmarks backed by auxiliary data."""

    _year = ""
    _auxiliary_data = ()

    def __init__(
        self,
        name=_MISSING,
        year=_MISSING,
        auxiliary_data=_MISSING,
        dims=_MISSING,
        continuous=_MISSING,
        convex=_MISSING,
        differentiable=_MISSING,
        multimodal=_MISSING,
        separable=_MISSING,
    ):
        super().__init__(
            name,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )
        self.year = self._year if year is _MISSING else year
        data = self._auxiliary_data if auxiliary_data is _MISSING else auxiliary_data
        self._load_auxiliary_data(self.name, self.year, data)

    @property
    def year(self):
        return self._year_value

    @year.setter
    def year(self, year):
        if not isinstance(year, str):
            raise e.TypeError("`year` should be a string")
        self._year_value = year

    def _load_auxiliary_data(self, name, year, data):
        for variable in data:
            setattr(self, variable, ld.load_cec_auxiliary(f"{name}_{variable}", year))


class CECCompositeBenchmark(CECBenchmark):
    """Base class for CEC composite benchmarks."""

    _bias = 1

    def __init__(
        self,
        name=_MISSING,
        year=_MISSING,
        auxiliary_data=_MISSING,
        sigma=(),
        l=(),
        functions=(),
        bias=_MISSING,
        dims=_MISSING,
        continuous=_MISSING,
        convex=_MISSING,
        differentiable=_MISSING,
        multimodal=_MISSING,
        separable=_MISSING,
    ):
        super().__init__(
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
        self._initialize_composition(
            sigma,
            l,
            functions,
            self._bias if bias is _MISSING else bias,
        )

    def _initialize_composition(self, sigma, scale, functions, bias):
        self.sigma = sigma
        self.l = scale
        self.f = functions
        self.bias = bias
        self.C = 2000
        self.f_bias = (0, 100, 200, 300, 400, 500, 600, 700, 800, 900)

    @d.check_exact_dimension_and_auxiliary_matrix
    def __call__(self, x):
        dimension = x.shape[0]
        weights = np.zeros(len(self.f))
        maxima = np.zeros(len(self.f))
        fitness = np.zeros(len(self.f))
        reference = 5 * np.ones(dimension)

        for index, function in enumerate(self.f):
            start = index * dimension
            end = start + dimension
            shifted = x - self.o[index][:dimension]
            weights[index] = np.exp(
                -np.sum(shifted**2) / (2 * dimension * self.sigma[index] ** 2)
            )
            maxima[index] = function(
                np.matmul(reference / self.l[index], self.M[start:end])
            )
            fitness[index] = (
                self.C
                * function(np.matmul(shifted / self.l[index], self.M[start:end]))
                / maxima[index]
            )

        weight_sum = np.sum(weights)
        maximum = np.max(weights)
        weights[weights != maximum] *= 1 - maximum**10
        weights /= weight_sum

        return np.matmul(weights, fitness + self.f_bias) + self.bias
