"""Base classes for CEC benchmark functions."""

import numpy as np

from opytimark.core.benchmark import Benchmark
from opytimark.utils.decorator import check_exact_dimension_and_auxiliary_matrix
from opytimark.utils.loader import load_cec_auxiliary


class CECBenchmark(Benchmark):
    """Base class for CEC benchmarks backed by bundled auxiliary data."""

    year = ""
    auxiliary_data = ()

    def __init__(self):
        name = type(self).__name__
        for variable in self.auxiliary_data:
            setattr(
                self,
                variable,
                load_cec_auxiliary(f"{name}_{variable}", self.year),
            )


class CECCompositeBenchmark(CECBenchmark):
    """Base class for CEC composite benchmarks."""

    C = 2000
    f_bias = tuple(range(0, 1000, 100))
    bias = 1

    def __init__(self, sigma, scale, functions):
        super().__init__()
        self.sigma = sigma
        self.l = scale
        self.f = functions

    @check_exact_dimension_and_auxiliary_matrix
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

        maximum = np.max(weights)
        weights[weights != maximum] *= 1 - maximum**10
        weights /= np.sum(weights)

        return np.matmul(weights, fitness + self.f_bias) + self.bias
