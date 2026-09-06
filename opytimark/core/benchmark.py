# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from numpy.typing import ArrayLike

import opytimark.utils.exception as e
from opytimark.typing import BenchmarkValue


class _Missing:
    def __repr__(self) -> str:
        return "<default>"


_MISSING = _Missing()


def _validate_type(name: str, value: object, expected_type: type) -> None:
    if not isinstance(value, expected_type):
        raise e.TypeError(
            f"`{name}` must be of type {expected_type.__name__}, but got {value!r} ({type(value).__name__})."
        )


class Benchmark:
    """
    Base class for callable benchmarks with validated, mutable metadata.

    """

    _defaults: tuple[str, int, bool, bool, bool, bool, bool] = ("Benchmark", 1, False, False, False, False, False)

    def __init__(
        self,
        name: str | _Missing = _MISSING,
        dims: int | _Missing = _MISSING,
        continuous: bool | _Missing = _MISSING,
        convex: bool | _Missing = _MISSING,
        differentiable: bool | _Missing = _MISSING,
        multimodal: bool | _Missing = _MISSING,
        separable: bool | _Missing = _MISSING,
    ) -> None:
        """Initialize benchmark metadata using the subclass defaults for omitted arguments.

        Args:
            name: Benchmark identifier used in diagnostics and CEC data-file selection.
            dims: Dimension metadata interpreted by the benchmark's input validator.
            continuous: Whether the benchmark is continuous.
            convex: Whether the benchmark is convex.
            differentiable: Whether the benchmark is differentiable.
            multimodal: Whether the benchmark has multiple modes.
            separable: Whether coordinates can be optimized independently.

        Raises:
            TypeError: A metadata value has the wrong type.
            ValueError: The dimension count is below -1 or equal to zero.

        Notes:
            Omitted values come from the subclass's `_defaults` tuple.
            Explicit values, including None, pass through property validation.
            Validators may require an exact size, impose an upper bound, or select supported CEC rotations.

        """

        supplied = (
            name,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )

        for field, value, default in zip(
            (
                "name",
                "dims",
                "continuous",
                "convex",
                "differentiable",
                "multimodal",
                "separable",
            ),
            supplied,
            self._defaults,
        ):
            setattr(self, field, default if value is _MISSING else value)

    @property
    def name(self) -> str:
        """Return the benchmark identifier.

        Returns:
            Validated benchmark name.

        """

        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """Set the benchmark identifier.

        Args:
            name: Benchmark name used in diagnostics.

        Raises:
            TypeError: The name is not a string.

        """

        _validate_type("name", name, str)

        self._name = name

    @property
    def dims(self) -> int:
        """Return the benchmark's dimension metadata.

        Returns:
            Positive dimension metadata or -1 for a variable-dimensional benchmark.

        """

        return self._dims

    @dims.setter
    def dims(self, dims: int) -> None:
        """Set the benchmark's dimension metadata.

        Args:
            dims: Positive dimension metadata or -1 for a variable-dimensional benchmark.

        Raises:
            TypeError: The dimension count is not an integer.
            ValueError: The dimension count is below -1 or equal to zero.

        """

        _validate_type("dims", dims, int)
        if not (dims >= -1 and dims != 0):
            raise e.ValueError(f"`dims` must be -1 or a positive integer, but got {dims!r}.")

        self._dims = dims

    @property
    def continuous(self) -> bool:
        """Return whether the benchmark is continuous.

        Returns:
            True when the benchmark is marked continuous.

        """

        return self._continuous

    @continuous.setter
    def continuous(self, continuous: bool) -> None:
        """Set the benchmark's continuity flag.

        Args:
            continuous: Whether the benchmark is continuous.

        Raises:
            TypeError: The continuity flag is not a boolean.

        """

        _validate_type("continuous", continuous, bool)

        self._continuous = continuous

    @property
    def convex(self) -> bool:
        """Return whether the benchmark is convex.

        Returns:
            True when the benchmark is marked convex.

        """

        return self._convex

    @convex.setter
    def convex(self, convex: bool) -> None:
        """Set the benchmark's convexity flag.

        Args:
            convex: Whether the benchmark is convex.

        Raises:
            TypeError: The convexity flag is not a boolean.

        """

        _validate_type("convex", convex, bool)

        self._convex = convex

    @property
    def differentiable(self) -> bool:
        """Return whether the benchmark is differentiable.

        Returns:
            True when the benchmark is marked differentiable.

        """

        return self._differentiable

    @differentiable.setter
    def differentiable(self, differentiable: bool) -> None:
        """Set the benchmark's differentiability flag.

        Args:
            differentiable: Whether the benchmark is differentiable.

        Raises:
            TypeError: The differentiability flag is not a boolean.

        """

        _validate_type("differentiable", differentiable, bool)

        self._differentiable = differentiable

    @property
    def multimodal(self) -> bool:
        """Return whether the benchmark has multiple modes.

        Returns:
            True when the benchmark is marked multimodal.

        """

        return self._multimodal

    @multimodal.setter
    def multimodal(self, multimodal: bool) -> None:
        """Set the benchmark's multimodality flag.

        Args:
            multimodal: Whether the benchmark has multiple modes.

        Raises:
            TypeError: The multimodality flag is not a boolean.

        """

        _validate_type("multimodal", multimodal, bool)

        self._multimodal = multimodal

    @property
    def separable(self) -> bool:
        """Return whether coordinates can be optimized independently.

        Returns:
            True when the benchmark is marked separable.

        """

        return self._separable

    @separable.setter
    def separable(self, separable: bool) -> None:
        """Set the benchmark's separability flag.

        Args:
            separable: Whether coordinates can be optimized independently.

        Raises:
            TypeError: The separability flag is not a boolean.

        """

        _validate_type("separable", separable, bool)

        self._separable = separable

    def __call__(self, x: ArrayLike) -> BenchmarkValue:
        """Evaluate the benchmark through a concrete implementation.

        Args:
            x: Numeric coordinates supplied to the benchmark.

        Returns:
            Benchmark value produced by a subclass implementation.

        Raises:
            NotImplementedError: The base benchmark has no numerical implementation.

        """

        raise NotImplementedError("`Benchmark.__call__` must be implemented by a subclass.")
