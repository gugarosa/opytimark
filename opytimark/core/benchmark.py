"""Base class for benchmark functions."""


class Benchmark:
    """Base class for callable benchmark functions."""

    dims = 1

    def __call__(self, x):
        raise NotImplementedError
