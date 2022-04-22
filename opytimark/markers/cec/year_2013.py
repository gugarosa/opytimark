"""CEC2013 benchmarking functions.
"""

import warnings
from typing import Optional, Tuple

import numpy as np

import opytimark.markers.n_dimensional as n_dim
import opytimark.utils.constants as c
import opytimark.utils.decorator as d
import opytimark.utils.exception as e
from opytimark.core import CECBenchmark

# Fixes Numpy's random seed
np.random.seed(0)


def T_irregularity(x: np.array) -> np.array:
    """Performs a transformation over the input to create smooth local irregularities.

    Args:
        x: An array holding the input to be transformed.

    Returns:
        (np.array): The transformed input.

    """

    # Defines the x_hat transformation
    x_hat = np.where(x != 0, np.log(np.fabs(x + c.EPSILON)), 0)

    # Defines both c_1 and c_2 transformations
    c_1 = np.where(x > 0, 10, 5.5)
    c_2 = np.where(x > 0, 7.9, 3.1)

    # Re-calculates the input
    x_t = np.sign(x) * np.exp(
        x_hat + 0.049 * (np.sin(c_1 * x_hat) + np.sin(c_2 * x_hat))
    )

    return x_t


def T_asymmetry(x: np.array, beta: float) -> np.array:
    """Performs a transformation over the input to break the symmetry of the symmetric functions.

    Args:
        x: An array holding the input to be transformed.
        beta: Exponential value used to produce the asymmetry.

    Returns:
        (np.array): The transformed input.

    """

    # Gathers the amount of dimensions and calculates an equally-spaced interval between 0 and D-1
    D = x.shape[0]
    dims = np.linspace(1, D, D) - 1

    # Activates the context manager for catching warnings
    with warnings.catch_warnings():
        # Ignores whenever the np.where raises an invalid square root value
        # This will ensure that no warnings will be raised when calculating the line below
        warnings.filterwarnings("ignore", r"invalid value encountered in sqrt")

        # Re-calculates the input
        x_t = np.where(x > 0, x ** (1 + beta * (dims / (D - 1)) * np.sqrt(x)), x)

    return x_t


def T_diagonal(D: int, alpha: float) -> np.array:
    """Creates a transformed diagonal matrix used to provide ill-conditioning.

    Args:
        D: Amount of dimensions.
        alpha: Exponential value used to produce the ill-conditioning.

    Returns:
        (np.array): The transformed diagonal matrix.

    """

    # Calculates an equally-spaced interval between 0 and D-1
    dims = np.linspace(1, D, D) - 1

    # Creates an empty matrix
    M = np.zeros((D, D))

    # Fill the diagonal matrix with the ill-condition
    np.fill_diagonal(M, alpha**0.5 * (dims / (D - 1)))

    return M


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
        year: Optional[str] = "2013",
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

        # Re-calculates the input using the proposed transform
        z = T_irregularity(x - self.o[: x.shape[0]])

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
        year: Optional[str] = "2013",
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

        # Re-calculates the input using the proposed transforms
        z = np.matmul(
            T_asymmetry(T_irregularity(x - self.o[: x.shape[0]]), 0.2),
            T_diagonal(x.shape[0], 10),
        )

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
        year: Optional[str] = "2013",
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

        # Re-calculates the input using the proposed transforms
        z = np.matmul(
            T_asymmetry(T_irregularity(x - self.o[: x.shape[0]]), 0.2),
            T_diagonal(x.shape[0], 10),
        )

        # Calculating the 1 / n term
        inv = 1 / x.shape[0]

        # Calculating first term
        term1 = -0.2 * np.sqrt(inv * np.sum(z**2))

        # Calculating second term
        term2 = inv * np.sum(np.cos(2 * np.pi * z))

        # Calculating Shifted Ackley's function
        f = 20 + np.e - np.exp(term2) - 20 * np.exp(term1)

        return f


class F4(CECBenchmark):
    """F4 class implements the 7-separable, 1-separable Shifted and Rotated Elliptic's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{|S|-1}w_i f_{elliptic}(z_i) + f_{elliptic}(z_{|S|})

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F4",
        year: Optional[str] = "2013",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "R25", "R50", "R100"),
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

        # Defines the subsets, weights and the benchmarking to be evaluated
        self.S = [50, 25, 25, 100, 50, 25, 25]
        self.W = [45.6996, 1.5646, 18465.3234, 0.0110, 13.6259, 0.3015, 59.6078]
        self.f = n_dim.HighConditionedElliptic()

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions, an array of permutations, a counter
        # and the function itself
        D = x.shape[0]
        P = np.random.permutation(D)
        n = 0
        f = 0

        # Checks if number of dimensions is valid
        if D < 302:
            # Raises an error
            raise e.SizeError("`D` should be greater than 302")

        # Re-calculates the input and permutes its input
        y = x - self.o[:D]
        y = y[P]

        # Iterates through every possible subset and weight
        for s, w in zip(self.S, self.W):
            # Checks if the subset has 25 features
            if s == 25:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R25, y[n : n + s])

            # Checks if the subset has 50 features
            elif s == 50:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R50, y[n : n + s])

            # Checks if the subset has 100 features
            elif s == 100:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R100, y[n : n + s])

            # Sums up the calculated fitness multiplied by its corresponding weight
            f += w * self.f(T_irregularity(z))

            # Also increments the dimension counter
            n += s

        # Lastly, gathers the remaining positions
        z = y[n:]

        # Calculates their fitness and sums up to produce the final result
        f += self.f(T_irregularity(z))

        return f


class F5(CECBenchmark):
    """F5 class implements the 7-separable, 1-separable Shifted and Rotated Rastrigin's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{|S|-1}w_i f_{rastrigin}(z_i) + f_{rastrigin}(z_{|S|})

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F5",
        year: Optional[str] = "2013",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "R25", "R50", "R100"),
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

        # Defines the subsets, weights and the benchmarking to be evaluated
        self.S = [50, 25, 25, 100, 50, 25, 25]
        self.W = [0.1807, 9081.1379, 24.2718, 1.8630e-06, 17698.0807, 0.0002, 0.0152]
        self.f = n_dim.Rastrigin()

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions, an array of permutations, a counter
        # and the function itself
        D = x.shape[0]
        P = np.random.permutation(D)
        n = 0
        f = 0

        # Checks if number of dimensions is valid
        if D < 302:
            # Raises an error
            raise e.SizeError("`D` should be greater than 302")

        # Re-calculates the input and permutes its input
        y = x - self.o[:D]
        y = y[P]

        # Iterates through every possible subset and weight
        for s, w in zip(self.S, self.W):
            # Checks if the subset has 25 features
            if s == 25:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R25, y[n : n + s])

            # Checks if the subset has 50 features
            elif s == 50:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R50, y[n : n + s])

            # Checks if the subset has 100 features
            elif s == 100:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R100, y[n : n + s])

            # Applies the irregulary, asymmetry and diagonal transforms
            z = np.matmul(
                T_asymmetry(T_irregularity(z), 0.2), T_diagonal(z.shape[0], 10)
            )

            # Sums up the calculated fitness multiplied by its corresponding weight
            f += w * self.f(z)

            # Also increments the dimension counter
            n += s

        # Lastly, gathers the remaining positions
        z = y[n:]

        # Applies the irregulary, asymmetry and diagonal transforms
        z = np.matmul(T_asymmetry(T_irregularity(z), 0.2), T_diagonal(z.shape[0], 10))

        # Calculates their fitness and sums up to produce the final result
        f += self.f(z)

        return f


class F6(CECBenchmark):
    """F6 class implements the 7-separable, 1-separable Shifted and Rotated Ackley's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{|S|-1}w_i f_{ackley}(z_i) + f_{ackley}(z_{|S|})

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-32, 32] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F6",
        year: Optional[str] = "2013",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "R25", "R50", "R100"),
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

        # Defines the subsets, weights and the benchmarking to be evaluated
        self.S = [50, 25, 25, 100, 50, 25, 25]
        self.W = [0.0352, 5.3156e-05, 0.8707, 49513.7420, 0.0831, 3.4764e-05, 282.2934]
        self.f = n_dim.Ackley1()

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions, an array of permutations, a counter
        # and the function itself
        D = x.shape[0]
        P = np.random.permutation(D)
        n = 0
        f = 0

        # Checks if number of dimensions is valid
        if D < 302:
            # Raises an error
            raise e.SizeError("`D` should be greater than 302")

        # Re-calculates the input and permutes its input
        y = x - self.o[:D]
        y = y[P]

        # Iterates through every possible subset and weight
        for s, w in zip(self.S, self.W):
            # Checks if the subset has 25 features
            if s == 25:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R25, y[n : n + s])

            # Checks if the subset has 50 features
            elif s == 50:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R50, y[n : n + s])

            # Checks if the subset has 100 features
            elif s == 100:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R100, y[n : n + s])

            # Applies the irregulary, asymmetry and diagonal transforms
            z = np.matmul(
                T_asymmetry(T_irregularity(z), 0.2), T_diagonal(z.shape[0], 10)
            )

            # Sums up the calculated fitness multiplied by its corresponding weight
            f += w * self.f(z)

            # Also increments the dimension counter
            n += s

        # Lastly, gathers the remaining positions
        z = y[n:]

        # Applies the irregulary, asymmetry and diagonal transforms
        z = np.matmul(T_asymmetry(T_irregularity(z), 0.2), T_diagonal(z.shape[0], 10))

        # Calculates their fitness and sums up to produce the final result
        f += self.f(z)

        return f


class F7(CECBenchmark):
    """F7 class implements the 7-separable, 1-separable Shifted Schwefel's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{|S|-1}w_i f_{schwefel}(z_i) + f_{sphere}(z_{|S|})

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F7",
        year: Optional[str] = "2013",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "R25", "R50", "R100"),
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

        # Defines the subsets, weights and the benchmarking to be evaluated
        self.S = [50, 25, 25, 100, 50, 25, 25]
        self.W = [679.9025, 0.9321, 2122.8501, 0.5060, 434.5961, 33389.6244, 2.5692]
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

        # Defines the number of dimensions, an array of permutations, a counter
        # and the function itself
        D = x.shape[0]
        P = np.random.permutation(D)
        n = 0
        f = 0

        # Checks if number of dimensions is valid
        if D < 302:
            # Raises an error
            raise e.SizeError("`D` should be greater than 302")

        # Re-calculates the input and permutes its input
        y = x - self.o[:D]
        y = y[P]

        # Iterates through every possible subset and weight
        for s, w in zip(self.S, self.W):
            # Checks if the subset has 25 features
            if s == 25:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R25, y[n : n + s])

            # Checks if the subset has 50 features
            elif s == 50:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R50, y[n : n + s])

            # Checks if the subset has 100 features
            elif s == 100:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R100, y[n : n + s])

            # Applies the irregulary and asymmetry transforms
            z = T_asymmetry(T_irregularity(z), 0.2)

            # Sums up the calculated fitness multiplied by its corresponding weight
            f += w * self.f_1(z)

            # Also increments the dimension counter
            n += s

        # Lastly, gathers the remaining positions
        z = y[n:]

        # Applies the irregulary and asymmetry transforms
        z = T_asymmetry(T_irregularity(z), 0.2)

        # Calculates their fitness and sums up to produce the final result
        f += self.f_2(z)

        return f


class F8(CECBenchmark):
    """F8 class implements the 20-nonseparable Shifted and Rotated Elliptic's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{|S|}w_i f_{elliptic}(z_i)

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F8",
        year: Optional[str] = "2013",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "R25", "R50", "R100"),
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

        # Defines the subsets, weights and the benchmarking to be evaluated
        self.S = [
            50,
            50,
            25,
            25,
            100,
            100,
            25,
            25,
            50,
            25,
            100,
            25,
            100,
            50,
            25,
            25,
            25,
            100,
            50,
            25,
        ]
        self.W = [
            4.6303,
            0.6864,
            1143756360.0887,
            2.0077,
            789.3671,
            16.3332,
            6.0749,
            0.0646,
            0.0756,
            35.6725,
            7.9725e-06,
            10.7822,
            4.1999e-06,
            0.0019,
            0.0016,
            686.7975,
            0.1571,
            0.0441,
            0.3543,
            0.0060,
        ]
        self.f = n_dim.HighConditionedElliptic()

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions, an array of permutations, a counter
        # and the function itself
        D = x.shape[0]
        P = np.random.permutation(D)
        n = 0
        f = 0

        # Re-calculates the input and permutes its input
        y = x - self.o[:D]
        y = y[P]

        # Iterates through every possible subset and weight
        for s, w in zip(self.S, self.W):
            # Checks if the subset has 25 features
            if s == 25:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R25, y[n : n + s])

            # Checks if the subset has 50 features
            elif s == 50:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R50, y[n : n + s])

            # Checks if the subset has 100 features
            elif s == 100:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R100, y[n : n + s])

            # Sums up the calculated fitness multiplied by its corresponding weight
            f += w * self.f(T_irregularity(z))

            # Also increments the dimension counter
            n += s

        return f


class F9(CECBenchmark):
    """F9 class implements the 20-nonseparable Shifted and Rotated Rastrigin's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{|S|}w_i f_{rastrigin}(z_i)

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-5, 5] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F9",
        year: Optional[str] = "2013",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "R25", "R50", "R100"),
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

        # Defines the subsets, weights and the benchmarking to be evaluated
        self.S = [
            50,
            50,
            25,
            25,
            100,
            100,
            25,
            25,
            50,
            25,
            100,
            25,
            100,
            50,
            25,
            25,
            25,
            100,
            50,
            25,
        ]
        self.W = [
            1756.9969,
            570.7338,
            3.3559,
            1.0364,
            62822.2923,
            1.7315,
            0.0898,
            0.0008,
            1403745.6363,
            8716.2083,
            0.0033,
            1.3495,
            0.0047,
            5089.9133,
            12.6664,
            0.0003,
            0.2400,
            3.9643,
            0.0014,
            0.0052,
        ]
        self.f = n_dim.Rastrigin()

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions, an array of permutations, a counter
        # and the function itself
        D = x.shape[0]
        P = np.random.permutation(D)
        n = 0
        f = 0

        # Re-calculates the input and permutes its input
        y = x - self.o[:D]
        y = y[P]

        # Iterates through every possible subset and weight
        for s, w in zip(self.S, self.W):
            # Checks if the subset has 25 features
            if s == 25:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R25, y[n : n + s])

            # Checks if the subset has 50 features
            elif s == 50:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R50, y[n : n + s])

            # Checks if the subset has 100 features
            elif s == 100:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R100, y[n : n + s])

            # Applies the irregulary, asymmetry and diagonal transforms
            z = np.matmul(
                T_asymmetry(T_irregularity(z), 0.2), T_diagonal(z.shape[0], 10)
            )

            # Sums up the calculated fitness multiplied by its corresponding weight
            f += w * self.f(z)

            # Also increments the dimension counter
            n += s

        return f


class F10(CECBenchmark):
    """F10 class implements the 20-nonseparable Shifted and Rotated Ackley's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{|S|}w_i f_{ackley}(z_i)

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-32, 32] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F10",
        year: Optional[str] = "2013",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "R25", "R50", "R100"),
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

        # Defines the subsets, weights and the benchmarking to be evaluated
        self.S = [
            50,
            50,
            25,
            25,
            100,
            100,
            25,
            25,
            50,
            25,
            100,
            25,
            100,
            50,
            25,
            25,
            25,
            100,
            50,
            25,
        ]
        self.W = [
            0.3127,
            15.1277,
            2323.3550,
            0.0008,
            11.4208,
            3.5541,
            29.9873,
            0.9981,
            1.6151,
            1.5128,
            0.6084,
            4464853.6323,
            6.8076e-05,
            0.1363,
            0.0007,
            59885.1276,
            1.8523,
            24.7834,
            0.5431,
            39.2404,
        ]
        self.f = n_dim.Ackley1()

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions, an array of permutations, a counter
        # and the function itself
        D = x.shape[0]
        P = np.random.permutation(D)
        n = 0
        f = 0

        # Re-calculates the input and permutes its input
        y = x - self.o[:D]
        y = y[P]

        # Iterates through every possible subset and weight
        for s, w in zip(self.S, self.W):
            # Checks if the subset has 25 features
            if s == 25:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R25, y[n : n + s])

            # Checks if the subset has 50 features
            elif s == 50:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R50, y[n : n + s])

            # Checks if the subset has 100 features
            elif s == 100:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R100, y[n : n + s])

            # Applies the irregulary, asymmetry and diagonal transforms
            z = np.matmul(
                T_asymmetry(T_irregularity(z), 0.2), T_diagonal(z.shape[0], 10)
            )

            # Sums up the calculated fitness multiplied by its corresponding weight
            f += w * self.f(z)

            # Also increments the dimension counter
            n += s

        return f


class F11(CECBenchmark):
    """F11 class implements the 20-nonseparable Shifted and Rotated Schwefel's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{|S|}w_i f_{schwefel}(z_i)

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F11",
        year: Optional[str] = "2013",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "R25", "R50", "R100"),
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

        # Defines the subsets, weights and the benchmarking to be evaluated
        self.S = [
            50,
            50,
            25,
            25,
            100,
            100,
            25,
            25,
            50,
            25,
            100,
            25,
            100,
            50,
            25,
            25,
            25,
            100,
            50,
            25,
        ]
        self.W = [
            0.0161,
            0.1286,
            0.0012,
            0.3492,
            3.9887,
            7.4469,
            2.6138,
            1.8601e-05,
            0.0779,
            4946500.0392,
            907.5677,
            1245.4389,
            0.0001,
            0.0025,
            0.0122,
            0.2253,
            16011.6801,
            4.1528,
            4208.6086,
            8.9830e-06,
        ]
        self.f = n_dim.RotatedHyperEllipsoid()

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions, an array of permutations, a counter
        # and the function itself
        D = x.shape[0]
        P = np.random.permutation(D)
        n = 0
        f = 0

        # Re-calculates the input and permutes its input
        y = x - self.o[:D]
        y = y[P]

        # Iterates through every possible subset and weight
        for s, w in zip(self.S, self.W):
            # Checks if the subset has 25 features
            if s == 25:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R25, y[n : n + s])

            # Checks if the subset has 50 features
            elif s == 50:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R50, y[n : n + s])

            # Checks if the subset has 100 features
            elif s == 100:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R100, y[n : n + s])

            # Applies the irregulary and asymmetry transforms
            z = T_asymmetry(T_irregularity(z), 0.2)

            # Sums up the calculated fitness multiplied by its corresponding weight
            f += w * self.f(z)

            # Also increments the dimension counter
            n += s

        return f


class F12(CECBenchmark):
    """F12 class implements the Shifted Rosenbrock's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n-1} (100(z_i^2-z_{i+1})^2 + (z_i - 1)^2) \mid z_i = x_i - o_i

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o} + 1`.

    """

    def __init__(
        self,
        name: Optional[str] = "F12",
        year: Optional[str] = "2013",
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


class F13(CECBenchmark):
    """F13 class implements the Shifted Schwefel's with Conforming Overlapping benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{|S|}w_i f_{schwefel}(z_i)

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F13",
        year: Optional[str] = "2013",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "R25", "R50", "R100"),
        dims=905,
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

        # Defines the subsets, weights, cumulative sum and the benchmarking to be evaluated
        self.S = [
            50,
            50,
            25,
            25,
            100,
            100,
            25,
            25,
            50,
            25,
            100,
            25,
            100,
            50,
            25,
            25,
            25,
            100,
            50,
            25,
        ]
        self.W = [
            0.4353,
            0.0099,
            0.0542,
            29.3627,
            11490.3303,
            24.1283,
            3.4511,
            2.3264,
            0.0017,
            0.0253,
            19.9959,
            0.0003,
            0.0013,
            0.0387,
            88.8945,
            57901.3138,
            0.0084,
            0.0736,
            0.6883,
            119314.8936,
        ]
        self.C = np.cumsum(self.S)
        self.f = n_dim.RotatedHyperEllipsoid()

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions, an array of permutations, an overlap size
        # and the function itself
        D = x.shape[0]
        P = np.random.permutation(D)
        m = 5
        f = 0

        # Re-calculates the input and permutes its input
        y = x - self.o[:D]
        y = y[P]

        # Iterates through every possible subset and weight
        for i, (s, w) in enumerate(zip(self.S, self.W)):
            # Checks if is the first iteration
            if i == 0:
                # If yes, defines the starting index as 0
                start_n = 0

            # If is not the first iteration
            else:
                # Calculates the starting index
                start_n = self.C[i - 1] - i * m

            # Calculates the ending index
            end_n = self.C[i] - i * m

            # Checks if the subset has 25 features
            if s == 25:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R25, y[start_n:end_n])

            # Checks if the subset has 50 features
            elif s == 50:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R50, y[start_n:end_n])

            # Checks if the subset has 100 features
            elif s == 100:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R100, y[start_n:end_n])

            # Applies the irregulary and asymmetry transforms
            z = T_asymmetry(T_irregularity(z), 0.2)

            # Sums up the calculated fitness multiplied by its corresponding weight
            f += w * self.f(z)

        return f


class F14(CECBenchmark):
    """F14 class implements the Shifted Schwefel's with Conflicting Overlapping benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{|S|}w_i f_{schwefel}(z_i)

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F14",
        year: Optional[str] = "2013",
        auxiliary_data: Optional[Tuple[str, ...]] = ("o", "R25", "R50", "R100"),
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

        # Defines the subsets, weights, cumulative sum and the benchmarking to be evaluated
        self.S = [
            50,
            50,
            25,
            25,
            100,
            100,
            25,
            25,
            50,
            25,
            100,
            25,
            100,
            50,
            25,
            25,
            25,
            100,
            50,
            25,
        ]
        self.W = [
            0.4753,
            498729.4349,
            328.1032,
            0.3231,
            136.4562,
            9.0255,
            0.0924,
            0.0001,
            0.0093,
            299.6790,
            4.9395,
            81.3641,
            0.6544,
            11.6119,
            2860774.3201,
            8.5835e-05,
            23.5695,
            0.0481,
            1.4318,
            12.1697,
        ]
        self.C = np.cumsum(self.S)
        self.f = n_dim.RotatedHyperEllipsoid()

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Defines the number of dimensions, an array of permutations, an overlap size
        # and the function itself
        D = x.shape[0]
        P = np.random.permutation(D)
        m = 5
        f = 0

        # Permutes the initial input
        x = x[P]

        # Iterates through every possible subset and weight
        for i, (s, w) in enumerate(zip(self.S, self.W)):
            # Checks if is the first iteration
            if i == 0:
                # If yes, defines both starting index and shift as 0
                start_n = 0
                start_shift = 0

            # If is not the first iteration
            else:
                # Calculates the starting index
                start_n = self.C[i - 1] - i * m
                start_shift = self.C[i - 1]

            # Calculates both ending index and shift
            end_n = self.C[i] - i * m
            end_shift = self.C[i]

            # Re-calculates the input
            y = x[start_n:end_n] - self.o[start_shift:end_shift]

            # Checks if the subset has 25 features
            if s == 25:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R25, y)

            # Checks if the subset has 50 features
            elif s == 50:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R50, y)

            # Checks if the subset has 100 features
            elif s == 100:
                # Rotates the input based on rotation matrix
                z = np.matmul(self.R100, y)

            # Applies the irregulary and asymmetry transforms
            z = T_asymmetry(T_irregularity(z), 0.2)

            # Sums up the calculated fitness multiplied by its corresponding weight
            f += w * self.f(z)

        return f


class F15(CECBenchmark):
    """F15 class implements the Shifted Schwefel's Problem 1.2 benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n}\sum_{j=1}^{i}z_j^2 \mid z_i = T_{asy}^{0.2}(T_{osz}(x_i - o_i))

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-100, 100] \mid i = \{1, 2, \ldots, n\}, n \leq 1000`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = \mathbf{o}`.

    """

    def __init__(
        self,
        name: Optional[str] = "F15",
        year: Optional[str] = "2013",
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

    @d.check_less_equal_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Re-calculates the input
        z = T_asymmetry(T_irregularity(x - self.o[: x.shape[0]]), 0.2)

        # Instantiating function
        f = 0

        # For every input dimension
        for i in range(x.shape[0]):
            # For `j` in `i` range
            for j in range(i):
                # Calculating the Schwefel's Problem 1.2 function
                f += z[j] ** 2

        return f
