"""Benchmark-based class.
"""

import opytimark.utils.exception as e


class Benchmark:
    """A Benchmark class is the root of any benchmarking function.

    It is composed by several properties that defines the traits of a function,
    as well as a non-implemented __call__ method.

    """

    def __init__(self, name='Benchmark', dims=1, continuous=False, convex=False,
                 differentiable=False, multimodal=False, separable=False):
        """Initialization method.

        Args:
            name (str): Name of the function.
            dims (int): Number of allowed dimensions.
            continuous (bool): Whether the function is continuous.
            convex (bool): Whether the function is convex.
            differentiable (bool): Whether the function is differentiable.
            multimodal (bool): Whether the function is multimodal.
            separable (bool): Whether the function is separable.

        """

        # Name of the function
        self.name = name

        # Number of allowed dimensions
        self.dims = dims

        # Continual
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
    def name(self):
        """str: Name of the function.

        """

        return self._name

    @name.setter
    def name(self, name):
        if not isinstance(name, str):
            raise e.TypeError('`name` should be a string')

        self._name = name

    @property
    def dims(self):
        """int: Number of allowed dimensions

        """

        return self._dims

    @dims.setter
    def dims(self, dims):
        if not isinstance(dims, int):
            raise e.TypeError('`dims` should be a integer')
        if (dims < -1 or dims == 0):
            raise e.ValueError('`dims` should be >= -1 and different than 0')

        self._dims = dims

    @property
    def continuous(self):
        """bool: Whether function is continuous or not.

        """

        return self._continuous

    @continuous.setter
    def continuous(self, continuous):
        if not isinstance(continuous, bool):
            raise e.TypeError('`continuous` should be a boolean')

        self._continuous = continuous

    @property
    def convex(self):
        """bool: Whether function is convex or not.

        """

        return self._convex

    @convex.setter
    def convex(self, convex):
        if not isinstance(convex, bool):
            raise e.TypeError('`convex` should be a boolean')

        self._convex = convex

    @property
    def differentiable(self):
        """bool: Whether function is differentiable or not.

        """

        return self._differentiable

    @differentiable.setter
    def differentiable(self, differentiable):
        if not isinstance(differentiable, bool):
            raise e.TypeError('`differentiable` should be a boolean')

        self._differentiable = differentiable

    @property
    def multimodal(self):
        """bool: Whether function is multimodal or not.

        """

        return self._multimodal

    @multimodal.setter
    def multimodal(self, multimodal):
        if not isinstance(multimodal, bool):
            raise e.TypeError('`multimodal` should be a boolean')

        self._multimodal = multimodal

    @property
    def separable(self):
        """bool: Whether function is separable or not.

        """

        return self._separable

    @separable.setter
    def separable(self, separable):
        if not isinstance(separable, bool):
            raise e.TypeError('`separable` should be a boolean')

        self._separable = separable

    def __call__(self, x):
        """This method returns the function's output when the class is called.

        Note that it needs to be implemented in every child class as it is the
        one to hold the benchmarking function logic.

        Args:
            x (np.array): An input array for calculating the function's output.

        Returns:
            The benchmarking function output `f(x)`.

        """

        raise NotImplementedError
