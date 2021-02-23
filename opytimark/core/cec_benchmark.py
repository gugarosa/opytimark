"""CEC Benchmark-based class.
"""

import opytimark.utils.exception as e
import opytimark.utils.loader as l


class CECBenchmark:
    """A CECBenchmark class is the root of CEC-based benchmarking function.

    It is composed by several properties that defines the traits of a function,
    as well as a non-implemented __call__ method.

    """

    def __init__(self, name, year, auxiliary_data=(), dims=1, continuous=False, convex=False,
                 differentiable=False, multimodal=False, separable=False):
        """Initialization method.

        Args:
            name (str): Name of the function.
            year (str): Year of the function.
            auxiliary_data (tuple): Auxiliary variables to be externally loaded.
            dims (int): Number of allowed dimensions.
            continuous (bool): Whether the function is continuous.
            convex (bool): Whether the function is convex.
            differentiable (bool): Whether the function is differentiable.
            multimodal (bool): Whether the function is multimodal.
            separable (bool): Whether the function is separable.

        """

        # Name of the function
        self.name = name

        # Year of the function
        self.year = year

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

        # Loads the auxiliary data
        self._load_auxiliary_data(name, year, auxiliary_data)

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
    def year(self):
        """str: Year of the function.

        """

        return self._year

    @year.setter
    def year(self, year):
        if not isinstance(year, str):
            raise e.TypeError('`year` should be a string')

        self._year = year

    @property
    def dims(self):
        """int: Number of allowed dimensions.

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

    def _load_auxiliary_data(self, name, year, data):
        """Loads auxiliary data from a set of files.

        Args:
            name (str): Name of the function.
            year (str): Year of the function.
            data (list): List holding the variables to be loaded.

        """

        # Iterates over the `data` list
        for d in data:
            # Constructs the data file
            # Note that it will always be NAME_VARIABLE
            data_file = f'{name}_{d}'

            # Loads the data to a temporary variable
            tmp = l.load_cec_auxiliary(data_file, year)

            # Sets the temporary variable as a property
            setattr(self, d, tmp)


class CECCompositeBenchmark:
    """A CECCompositeBenchmark class is the root of CEC-based composite benchmarking function.

    It is composed by several properties that defines the traits of a function,
    as well as a non-implemented __call__ method.

    """

    def __init__(self, name, year, dims=1, continuous=False, convex=False,
                 differentiable=False, multimodal=False, separable=False):
        """Initialization method.

        Args:
            name (str): Name of the function.
            year (str): Year of the function.
            dims (int): Number of allowed dimensions.
            continuous (bool): Whether the function is continuous.
            convex (bool): Whether the function is convex.
            differentiable (bool): Whether the function is differentiable.
            multimodal (bool): Whether the function is multimodal.
            separable (bool): Whether the function is separable.

        """

        # Overrides the parent class
        super(CECCompositeBenchmark).__init__(name, year, dims, continuous,
                                              convex, differentiable, multimodal, separable)

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

    