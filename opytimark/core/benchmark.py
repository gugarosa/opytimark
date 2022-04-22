"""Benchmark-based class.
"""

from typing import Optional

import numpy as np

import opytimark.utils.exception as e


class Benchmark:
    """A Benchmark class is the root of any benchmarking function.

    It is composed by several properties that defines the traits of a function,
    as well as a non-implemented __call__ method.

    """

    def __init__(
        self,
        name: Optional[str] = "Benchmark",
        dims: Optional[int] = 1,
        continuous: Optional[bool] = False,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = False,
        multimodal: Optional[bool] = False,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        # Name of the function
        self.name = name

        # Number of allowed dimensions
        self.dims = dims

        # Continuous
        self.continuous = continuous

        # Convexity
        self.convex = convex

        # Differentiability
        self.differentiable = differentiable

        # Modality
        self.multimodal = multimodal

        # Separability
        self.separable = separable

    @property
    def name(self) -> str:
        """Name of the function."""

        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if not isinstance(name, str):
            raise e.TypeError("`name` should be a string")

        self._name = name

    @property
    def dims(self) -> int:
        """Number of allowed dimensions."""

        return self._dims

    @dims.setter
    def dims(self, dims: int) -> None:
        if not isinstance(dims, int):
            raise e.TypeError("`dims` should be a integer")
        if dims < -1 or dims == 0:
            raise e.ValueError("`dims` should be >= -1 and different than 0")

        self._dims = dims

    @property
    def continuous(self) -> bool:
        """Whether function is continuous or not."""

        return self._continuous

    @continuous.setter
    def continuous(self, continuous: bool) -> None:
        if not isinstance(continuous, bool):
            raise e.TypeError("`continuous` should be a boolean")

        self._continuous = continuous

    @property
    def convex(self) -> bool:
        """Whether function is convex or not."""

        return self._convex

    @convex.setter
    def convex(self, convex: bool) -> None:
        if not isinstance(convex, bool):
            raise e.TypeError("`convex` should be a boolean")

        self._convex = convex

    @property
    def differentiable(self) -> bool:
        """Whether function is differentiable or not."""

        return self._differentiable

    @differentiable.setter
    def differentiable(self, differentiable: bool) -> None:
        if not isinstance(differentiable, bool):
            raise e.TypeError("`differentiable` should be a boolean")

        self._differentiable = differentiable

    @property
    def multimodal(self) -> bool:
        """Whether function is multimodal or not."""

        return self._multimodal

    @multimodal.setter
    def multimodal(self, multimodal: bool) -> None:
        if not isinstance(multimodal, bool):
            raise e.TypeError("`multimodal` should be a boolean")

        self._multimodal = multimodal

    @property
    def separable(self) -> bool:
        """Whether function is separable or not."""

        return self._separable

    @separable.setter
    def separable(self, separable: bool) -> None:
        if not isinstance(separable, bool):
            raise e.TypeError("`separable` should be a boolean")

        self._separable = separable

    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Note that it needs to be implemented in every child class as it is the
        one to hold the benchmarking function logic.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        raise NotImplementedError
