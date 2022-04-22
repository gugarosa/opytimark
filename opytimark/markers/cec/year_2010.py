"""CEC2010 benchmarking functions.
"""

from typing import Optional, Tuple

import numpy as np

import opytimark.markers.n_dimensional as n_dim
import opytimark.utils.decorator as d
import opytimark.utils.exception as e
from opytimark.core import CECBenchmark

# Fixes Numpy's random seed
np.random.seed(0)


class F1(CECBenchmark):
    """F1 class implements the Shifted Elliptic's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (10^6)^\\frac{i-1}{n-1} z_i^2 \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F1",
        year: Optional[str] = "2010",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o"),
        dims: Optional[int] = 1000,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = False,
        separable: Optional[bool] = True,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F1, self).__init__(
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

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions and an equally-spaced interval between 0 and D-1
        D = x.shape[0]
        dims = np.linspace(1, D, D) - 1

        # Re-calculates the input
        z = x - self.o[:D]

        # Calculating the Shifted Elliptic's function
        z = 10e6 ** (dims / (D - 1)) * z**2

        return np.sum(z)


class F2(CECBenchmark):
    """F2 class implements the Shifted Rastrigin's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} (z_i^2 - 10cos(2 \\pi z_i) + 10) \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F2",
        year: Optional[str] = "2010",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o"),
        dims: Optional[int] = 1000,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = True,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F2, self).__init__(
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

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Re-calculates the input
        z = x - self.o[: x.shape[0]]

        # Calculating the Shifted Rastrigin's function
        f = z**2 - 10 * np.cos(2 * np.pi * z) + 10

        return np.sum(f)


class F3(CECBenchmark):
    """F3 class implements the Shifted Ackley's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = -20e^{-0.2\sqrt{\\frac{1}{n}\sum_{i=1}^{n}x_i^2}}-e^{\\frac{1}{n}\sum_{i=1}^{n}cos(2 \\pi x_i)}+ 20 + e \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-32, 32] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F3",
        year: Optional[str] = "2010",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o"),
        dims: Optional[int] = 1000,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = True,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F3, self).__init__(
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

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Re-calculates the input
        z = x - self.o[: x.shape[0]]

        # Calculating the 1 / n term
        inv = 1 / x.shape[0]

        # Calculating first term
        term1 = -0.2 * np.sqrt(inv * np.sum(z**2))

        # Calculating second term
        term2 = inv * np.sum(np.cos(2 * np.pi * z))

        # Calculating Shifted Ackley's function
        f = 20 + np.e - np.exp(term2) - 20 * np.exp(term1)

        return np.sum(f)


class F4(CECBenchmark):
    """F4 class implements the Single-group Shifted and m-rotated Elliptic's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = f_{rot\_elliptic}[z(P_1:P_m)] * 10^6 + f_{elliptic}[z(P_{m+1}:P_n)] \mid z_i = x_i - o_i, z_i = (x_i - o_i) \\ast M_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F4",
        year: Optional[str] = "2010",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M"),
        dims: Optional[int] = 1000,
        group_size: Optional[int] = 50,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = False,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            group_size: Size of function's group, i.e., `m` variable.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F4, self).__init__(
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

        # Defines the size of the group and benchmarking function
        self.m = group_size
        self.f = n_dim.HighConditionedElliptic()

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions
        D = x.shape[0]

        # If group size is bigger or equal to number of dimensions
        if self.m >= D:
            # Raises an error
            raise e.SizeError(
                "`group_size` should be smaller than number of input dimensions"
            )

        # Calculates an array of permutations and defines both groups' indexes
        p = np.random.permutation(D)
        p_1 = p[: self.m]
        p_2 = p[self.m :]

        # Shifts the input data
        s = x - self.o[:D]

        # Re-calculates both groups' inputs
        z_rot = np.dot(s[p_1], self.M[: self.m][: self.m])
        z = s[p_2]

        return self.f(z_rot) * 10e6 + self.f(z)


class F5(CECBenchmark):
    """F5 class implements the Single-group Shifted and m-rotated Rastrigin's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = f_{rot\_rastrigin}[z(P_1:P_m)] * 10^6 + f_{rastrigin}[z(P_{m+1}:P_n)] \mid z_i = x_i - o_i, z_i = (x_i - o_i) \\ast M_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F5",
        year: Optional[str] = "2010",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M"),
        dims: Optional[int] = 1000,
        group_size: Optional[int] = 50,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            group_size: Size of function's group, i.e., `m` variable.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F5, self).__init__(
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

        # Defines the size of the group and benchmarking function
        self.m = group_size
        self.f = n_dim.Rastrigin()

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions
        D = x.shape[0]

        # If group size is bigger or equal to number of dimensions
        if self.m >= D:
            # Raises an error
            raise e.SizeError(
                "`group_size` should be smaller than number of input dimensions"
            )

        # Calculates an array of permutations and defines both groups' indexes
        p = np.random.permutation(D)
        p_1 = p[: self.m]
        p_2 = p[self.m :]

        # Shifts the input data
        s = x - self.o[:D]

        # Re-calculates both groups' inputs
        z_rot = np.dot(s[p_1], self.M[: self.m][: self.m])
        z = s[p_2]

        return self.f(z_rot) * 10e6 + self.f(z)


class F6(CECBenchmark):
    """F6 class implements the Single-group Shifted and m-rotated Ackley's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = f_{rot\_ackley}[z(P_1:P_m)] * 10^6 + f_{ackley}[z(P_{m+1}:P_n)] \mid z_i = x_i - o_i, z_i = (x_i - o_i) \\ast M_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-32, 32] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F6",
        year: Optional[str] = "2010",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M"),
        dims: Optional[int] = 1000,
        group_size: Optional[int] = 50,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            group_size: Size of function's group, i.e., `m` variable.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F6, self).__init__(
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

        # Defines the size of the group and benchmarking function
        self.m = group_size
        self.f = n_dim.Ackley1()

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions
        D = x.shape[0]

        # If group size is bigger or equal to number of dimensions
        if self.m >= D:
            # Raises an error
            raise e.SizeError(
                "`group_size` should be smaller than number of input dimensions"
            )

        # Calculates an array of permutations and defines both groups' indexes
        p = np.random.permutation(D)
        p_1 = p[: self.m]
        p_2 = p[self.m :]

        # Shifts the input data
        s = x - self.o[:D]

        # Re-calculates both groups' inputs
        z_rot = np.dot(s[p_1], self.M[: self.m][: self.m])
        z = s[p_2]

        return self.f(z_rot) * 10e6 + self.f(z)


class F7(CECBenchmark):
    """F7 class implements the Single-group Shifted and m-rotated Schwefel's Problem 1.2 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = f_{schwefel}[z(P_1:P_m)] * 10^6 + f_{sphere}[z(P_{m+1}:P_n)] \mid z_i = x_i - o_i, z_i = (x_i - o_i) \\ast M_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F7",
        year: Optional[str] = "2010",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o"),
        dims: Optional[int] = 1000,
        group_size: Optional[int] = 50,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = False,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            group_size: Size of function's group, i.e., `m` variable.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F7, self).__init__(
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

        # Defines the size of the group and benchmarking function
        self.m = group_size
        self.f_1 = n_dim.RotatedHyperEllipsoid()
        self.f_2 = n_dim.Sphere()

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions
        D = x.shape[0]

        # If group size is bigger or equal to number of dimensions
        if self.m >= D:
            # Raises an error
            raise e.SizeError(
                "`group_size` should be smaller than number of input dimensions"
            )

        # Calculates an array of permutations and defines both groups' indexes
        p = np.random.permutation(D)
        p_1 = p[: self.m]
        p_2 = p[self.m :]

        # Shifts the input data
        s = x - self.o[:D]

        # Re-calculates both groups' inputs
        z_1 = s[p_1]
        z_2 = s[p_2]

        return self.f_1(z_1) * 10e6 + self.f_2(z_2)


class F8(CECBenchmark):
    """F8 class implements the Single-group Shifted and m-rotated Rosenbrock's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = f_{rosenbrock}[z(P_1:P_m)] * 10^6 + f_{sphere}[z(P_{m+1}:P_n)] \mid z_i = x_i - o_i, z_i = (x_i - o_i) \\ast M_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F8",
        year: Optional[str] = "2010",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o"),
        dims: Optional[int] = 1000,
        group_size: Optional[int] = 50,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            group_size: Size of function's group, i.e., `m` variable.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F8, self).__init__(
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

        # Defines the size of the group and benchmarking function
        self.m = group_size
        self.f_1 = n_dim.Rosenbrock()
        self.f_2 = n_dim.Sphere()

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions
        D = x.shape[0]

        # If group size is bigger or equal to number of dimensions
        if self.m >= D:
            # Raises an error
            raise e.SizeError(
                "`group_size` should be smaller than number of input dimensions"
            )

        # Calculates an array of permutations and defines both groups' indexes
        p = np.random.permutation(D)
        p_1 = p[: self.m]
        p_2 = p[self.m :]

        # Shifts the input data
        s = x - self.o[:D]

        # Re-calculates both groups' inputs
        z_1 = s[p_1]
        z_2 = s[p_2]

        return self.f_1(z_1) * 10e6 + self.f_2(z_2)


class F9(CECBenchmark):
    """F9 class implements the D/2m-group Shifted and m-rotated Elliptic's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{k=1}^{\\frac{n}{2m}} f_{rot\_elliptic}[z(P_{(k-1)*m+1}:P_{k*m})] * 10^6 + f_{elliptic}[z(P_{\\frac{n}{2}+1}:P_n)] \mid z_i = x_i - o_i, z_i = (x_i - o_i) \\ast M_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F9",
        year: Optional[str] = "2010",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M"),
        dims: Optional[int] = 1000,
        group_size: Optional[int] = 50,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = False,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            group_size: Size of function's group, i.e., `m` variable.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F9, self).__init__(
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

        # Defines the size of the group and benchmarking function
        self.m = group_size
        self.f = n_dim.HighConditionedElliptic()

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions, instantiates the function and calculates the number of groups
        D = x.shape[0]
        f = 0
        n_groups = int(D / (2 * self.m))

        # If group size is bigger or equal to number of dimensions
        if self.m >= D:
            # Raises an error
            raise e.SizeError(
                "`group_size` should be smaller than number of input dimensions"
            )

        # Calculates an array of permutations
        p = np.random.permutation(D)

        # Shifts the input data
        s = x - self.o[:D]

        # Iterates through all groups
        for i in range(n_groups):
            # Re-calculates the first group input
            p_1 = p[i * self.m : (i + 1) * self.m]
            z_rot = np.dot(s[p_1], self.M[: self.m][: self.m])

            # Sums up the first group output
            f += self.f(z_rot)

        # Re-calculates the second group input
        p_2 = p[int(D / 2) :]
        z = s[p_2]

        # Sums up the second group output
        f += self.f(z)

        return f


class F10(CECBenchmark):
    """F10 class implements the D/2m-group Shifted and m-rotated Rastrigin's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{k=1}^{\\frac{n}{2m}} f_{rot\_rastrigin}[z(P_{(k-1)*m+1}:P_{k*m})] * 10^6 + f_{rastrigin}[z(P_{\\frac{n}{2}+1}:P_n)] \mid z_i = x_i - o_i, z_i = (x_i - o_i) \\ast M_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F10",
        year: Optional[str] = "2010",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M"),
        dims: Optional[int] = 1000,
        group_size: Optional[int] = 50,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            group_size: Size of function's group, i.e., `m` variable.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F10, self).__init__(
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

        # Defines the size of the group and benchmarking function
        self.m = group_size
        self.f = n_dim.Rastrigin()

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions, instantiates the function and calculates the number of groups
        D = x.shape[0]
        f = 0
        n_groups = int(D / (2 * self.m))

        # If group size is bigger or equal to number of dimensions
        if self.m >= D:
            # Raises an error
            raise e.SizeError(
                "`group_size` should be smaller than number of input dimensions"
            )

        # Calculates an array of permutations
        p = np.random.permutation(D)

        # Shifts the input data
        s = x - self.o[:D]

        # Iterates through all groups
        for i in range(n_groups):
            # Re-calculates the first group input
            p_1 = p[i * self.m : (i + 1) * self.m]
            z_rot = np.dot(s[p_1], self.M[: self.m][: self.m])

            # Sums up the first group output
            f += self.f(z_rot)

        # Re-calculates the second group input
        p_2 = p[int(D / 2) :]
        z = s[p_2]

        # Sums up the second group output
        f += self.f(z)

        return f


class F11(CECBenchmark):
    """F11 class implements the D/2m-group Shifted and m-rotated Ackley's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{k=1}^{\\frac{n}{2m}} f_{rot\_ackley}[z(P_{(k-1)*m+1}:P_{k*m})] * 10^6 + f_{ackley}[z(P_{\\frac{n}{2}+1}:P_n)] \mid z_i = x_i - o_i, z_i = (x_i - o_i) \\ast M_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-32, 32] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F11",
        year: Optional[str] = "2010",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M"),
        dims: Optional[int] = 1000,
        group_size: Optional[int] = 50,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            group_size: Size of function's group, i.e., `m` variable.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F11, self).__init__(
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

        # Defines the size of the group and benchmarking function
        self.m = group_size
        self.f = n_dim.Ackley1()

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions, instantiates the function and calculates the number of groups
        D = x.shape[0]
        f = 0
        n_groups = int(D / (2 * self.m))

        # If group size is bigger or equal to number of dimensions
        if self.m >= D:
            # Raises an error
            raise e.SizeError(
                "`group_size` should be smaller than number of input dimensions"
            )

        # Calculates an array of permutations
        p = np.random.permutation(D)

        # Shifts the input data
        s = x - self.o[:D]

        # Iterates through all groups
        for i in range(n_groups):
            # Re-calculates the first group input
            p_1 = p[i * self.m : (i + 1) * self.m]
            z_rot = np.dot(s[p_1], self.M[: self.m][: self.m])

            # Sums up the first group output
            f += self.f(z_rot)

        # Re-calculates the second group input
        p_2 = p[int(D / 2) :]
        z = s[p_2]

        # Sums up the second group output
        f += self.f(z)

        return f


class F12(CECBenchmark):
    """F12 class implements the D/2m-group Shifted and m-rotated Schwefel's Problem 1.2 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{k=1}^{\\frac{n}{2m}} f_{schwefel}[z(P_{(k-1)*m+1}:P_{k*m})] * 10^6 + f_{sphere}[z(P_{\\frac{n}{2}+1}:P_n)] \mid z_i = x_i - o_i, z_i = (x_i - o_i) \\ast M_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F12",
        year: Optional[str] = "2010",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o"),
        dims: Optional[int] = 1000,
        group_size: Optional[int] = 50,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = False,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            group_size: Size of function's group, i.e., `m` variable.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F12, self).__init__(
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

        # Defines the size of the group and benchmarking function
        self.m = group_size
        self.f_1 = n_dim.RotatedHyperEllipsoid()
        self.f_2 = n_dim.Sphere()

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions, instantiates the function and calculates the number of groups
        D = x.shape[0]
        f = 0
        n_groups = int(D / (2 * self.m))

        # If group size is bigger or equal to number of dimensions
        if self.m >= D:
            # Raises an error
            raise e.SizeError(
                "`group_size` should be smaller than number of input dimensions"
            )

        # Calculates an array of permutations
        p = np.random.permutation(D)

        # Shifts the input data
        s = x - self.o[:D]

        # Iterates through all groups
        for i in range(n_groups):
            # Re-calculates the first group input
            p_1 = p[i * self.m : (i + 1) * self.m]
            z_1 = s[p_1]

            # Sums up the first group output
            f += self.f_1(z_1)

        # Re-calculates the second group input
        p_2 = p[int(D / 2) :]
        z_2 = s[p_2]

        # Sums up the second group output
        f += self.f_2(z_2)

        return f


class F13(CECBenchmark):
    """F13 class implements the D/2m-group Shifted and m-rotated Rosenbrock benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{k=1}^{\\frac{n}{2m}} f_{rosenbrock}[z(P_{(k-1)*m+1}:P_{k*m})] * 10^6 + f_{sphere}[z(P_{\\frac{n}{2}+1}:P_n)] \mid z_i = x_i - o_i, z_i = (x_i - o_i) \\ast M_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F13",
        year: Optional[str] = "2010",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o"),
        dims: Optional[int] = 1000,
        group_size: Optional[int] = 50,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            group_size: Size of function's group, i.e., `m` variable.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F13, self).__init__(
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

        # Defines the size of the group and benchmarking function
        self.m = group_size
        self.f_1 = n_dim.Rosenbrock()
        self.f_2 = n_dim.Sphere()

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions, instantiates the function and calculates the number of groups
        D = x.shape[0]
        f = 0
        n_groups = int(D / (2 * self.m))

        # If group size is bigger or equal to number of dimensions
        if self.m >= D:
            # Raises an error
            raise e.SizeError(
                "`group_size` should be smaller than number of input dimensions"
            )

        # Calculates an array of permutations
        p = np.random.permutation(D)

        # Shifts the input data
        s = x - self.o[:D]

        # Iterates through all groups
        for i in range(n_groups):
            # Re-calculates the first group input
            p_1 = p[i * self.m : (i + 1) * self.m]
            z_1 = s[p_1]

            # Sums up the first group output
            f += self.f_1(z_1)

        # Re-calculates the second group input
        p_2 = p[int(D / 2) :]
        z_2 = s[p_2]

        # Sums up the second group output
        f += self.f_2(z_2)

        return f


class F14(CECBenchmark):
    """F14 class implements the D/m-group Shifted and m-rotated Elliptic's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{k=1}^{\\frac{n}{m}} f_{rot\_elliptic}[z(P_{(k-1)*m+1}:P_{k*m})] \mid z_i = (x_i - o_i) \\ast M_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F14",
        year: Optional[str] = "2010",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M"),
        dims: Optional[int] = 1000,
        group_size: Optional[int] = 50,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = False,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            group_size: Size of function's group, i.e., `m` variable.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F14, self).__init__(
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

        # Defines the size of the group and benchmarking function
        self.m = group_size
        self.f = n_dim.HighConditionedElliptic()

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions, instantiates the function and calculates the number of groups
        D = x.shape[0]
        f = 0
        n_groups = int(D / (self.m))

        # If group size is bigger or equal to number of dimensions
        if self.m >= D:
            # Raises an error
            raise e.SizeError(
                "`group_size` should be smaller than number of input dimensions"
            )

        # Calculates an array of permutations
        P = np.random.permutation(D)

        # Shifts the input data
        s = x - self.o[:D]

        # Iterates through all groups
        for i in range(n_groups):
            # Re-calculates the first group input
            p = P[i * self.m : (i + 1) * self.m]
            z = np.dot(s[p], self.M[: self.m][: self.m])

            # Sums up the group output
            f += self.f(z)

        return f


class F15(CECBenchmark):
    """F15 class implements the D/m-group Shifted and m-rotated Rastrigin's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{k=1}^{\\frac{n}{m}} f_{rot\_elliptic}[z(P_{(k-1)*m+1}:P_{k*m})] \mid z_i = (x_i - o_i) \\ast M_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F15",
        year: Optional[str] = "2010",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M"),
        dims: Optional[int] = 1000,
        group_size: Optional[int] = 50,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            group_size: Size of function's group, i.e., `m` variable.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F15, self).__init__(
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

        # Defines the size of the group and benchmarking function
        self.m = group_size
        self.f = n_dim.Rastrigin()

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions, instantiates the function and calculates the number of groups
        D = x.shape[0]
        f = 0
        n_groups = int(D / (self.m))

        # If group size is bigger or equal to number of dimensions
        if self.m >= D:
            # Raises an error
            raise e.SizeError(
                "`group_size` should be smaller than number of input dimensions"
            )

        # Calculates an array of permutations
        P = np.random.permutation(D)

        # Shifts the input data
        s = x - self.o[:D]

        # Iterates through all groups
        for i in range(n_groups):
            # Re-calculates the first group input
            p = P[i * self.m : (i + 1) * self.m]
            z = np.dot(s[p], self.M[: self.m][: self.m])

            # Sums up the group output
            f += self.f(z)

        return f


class F16(CECBenchmark):
    """F16 class implements the D/m-group Shifted and m-rotated Ackley's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{k=1}^{\\frac{n}{m}} f_{rot\_elliptic}[z(P_{(k-1)*m+1}:P_{k*m})] \mid z_i = (x_i - o_i) \\ast M_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-32, 32] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F16",
        year: Optional[str] = "2010",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "M"),
        dims: Optional[int] = 1000,
        group_size: Optional[int] = 50,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            group_size: Size of function's group, i.e., `m` variable.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F16, self).__init__(
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

        # Defines the size of the group and benchmarking function
        self.m = group_size
        self.f = n_dim.Ackley1()

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions, instantiates the function and calculates the number of groups
        D = x.shape[0]
        f = 0
        n_groups = int(D / (self.m))

        # If group size is bigger or equal to number of dimensions
        if self.m >= D:
            # Raises an error
            raise e.SizeError(
                "`group_size` should be smaller than number of input dimensions"
            )

        # Calculates an array of permutations
        P = np.random.permutation(D)

        # Shifts the input data
        s = x - self.o[:D]

        # Iterates through all groups
        for i in range(n_groups):
            # Re-calculates the first group input
            p = P[i * self.m : (i + 1) * self.m]
            z = np.dot(s[p], self.M[: self.m][: self.m])

            # Sums up the group output
            f += self.f(z)

        return f


class F17(CECBenchmark):
    """F17 class implements the D/m-group Shifted Schwefel's Problem 1.2 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{k=1}^{\\frac{n}{m}} f_{rot\_elliptic}[z(P_{(k-1)*m+1}:P_{k*m})] \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F17",
        year: Optional[str] = "2010",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o"),
        dims: Optional[int] = 1000,
        group_size: Optional[int] = 50,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = False,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            group_size: Size of function's group, i.e., `m` variable.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F17, self).__init__(
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

        # Defines the size of the group and benchmarking function
        self.m = group_size
        self.f = n_dim.RotatedHyperEllipsoid()

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions, instantiates the function and calculates the number of groups
        D = x.shape[0]
        f = 0
        n_groups = int(D / (self.m))

        # If group size is bigger or equal to number of dimensions
        if self.m >= D:
            # Raises an error
            raise e.SizeError(
                "`group_size` should be smaller than number of input dimensions"
            )

        # Calculates an array of permutations
        P = np.random.permutation(D)

        # Shifts the input data
        s = x - self.o[:D]

        # Iterates through all groups
        for i in range(n_groups):
            # Re-calculates the first group input
            p = P[i * self.m : (i + 1) * self.m]
            z = s[p]

            # Sums up the group output
            f += self.f(z)

        return f


class F18(CECBenchmark):
    """F18 class implements the D/m-group Shifted Rosenbrock's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{k=1}^{\\frac{n}{m}} f_{rot\_elliptic}[z(P_{(k-1)*m+1}:P_{k*m})] \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o} + 1`.

    """

    def __init__(
        self,
        name: Optional[str] = "F18",
        year: Optional[str] = "2010",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o"),
        dims: Optional[int] = 1000,
        group_size: Optional[int] = 50,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            group_size: Size of function's group, i.e., `m` variable.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F18, self).__init__(
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

        # Defines the size of the group and benchmarking function
        self.m = group_size
        self.f = n_dim.Rosenbrock()

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions, instantiates the function and calculates the number of groups
        D = x.shape[0]
        f = 0
        n_groups = int(D / (self.m))

        # If group size is bigger or equal to number of dimensions
        if self.m >= D:
            # Raises an error
            raise e.SizeError(
                "`group_size` should be smaller than number of input dimensions"
            )

        # Calculates an array of permutations
        P = np.random.permutation(D)

        # Shifts the input data
        s = x - self.o[:D]

        print(s, P)

        # Iterates through all groups
        for i in range(n_groups):
            # Re-calculates the first group input
            p = P[i * self.m : (i + 1) * self.m]
            z = s[p]

            # Sums up the group output
            f += self.f(z)

        return f


class F19(CECBenchmark):
    """F19 class implements the Shifted Schwefel's Problem 1.2 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}\sum_{j=1}^{i}z_j^2 \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F19",
        year: Optional[str] = "2010",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o"),
        dims: Optional[int] = 1000,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = False,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F19, self).__init__(
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

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Re-calculates the input
        z = x - self.o[: x.shape[0]]

        # Instantiating function
        f = 0

        # For every input dimension
        for i in range(x.shape[0]):
            # For `j` in `i` range
            for j in range(i):
                # Calculating the Schwefel's Problem 1.2 function
                f += z[j] ** 2

        return f


class F20(CECBenchmark):
    """F20 class implements the Shifted Rosenbrock's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-1} (100(z_i^2-z_{i+1})^2 + (z_i - 1)^2) \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o} + 1`.

    """

    def __init__(
        self,
        name: Optional[str] = "F20",
        year: Optional[str] = "2010",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o"),
        dims: Optional[int] = 1000,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            year: Year of the function.
            auxiliary_data: Auxiliary variables to be externally loaded.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(F20, self).__init__(
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

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Re-calculates the input
        z = x - self.o[: x.shape[0]]

        # Instantiating function
        f = 0

        # For every input dimension
        for i in range(x.shape[0] - 1):
            # Calculating the Shifted Rosenbrock's function
            f += 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2

        return f
