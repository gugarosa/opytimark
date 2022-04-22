"""Many-dimensional benchmarking functions.
"""

from typing import Optional

import numpy as np

import opytimark.utils.decorator as d
from opytimark.core import Benchmark


class BiggsExponential3(Benchmark):
    """BiggsExponential3 class implements the Biggs Exponential's 3rd benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3) = \sum_{i=1}^{10}(e^{-t_ix_1} - x_3e^{-t_ix_2} - y_i)^2

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 20] \mid i = \{1, 2, 3\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 10, 5)`.

    """

    def __init__(
        self,
        name: Optional[str] = "BiggsExponential3",
        dims: Optional[int] = 3,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = True,
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

        super(BiggsExponential3, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Instantiating function
        f = 0

        # For `i` ranging from 1 to 10
        for i in range(1, 11):
            # Calculating `z`
            z = i / 10

            # Calculating partial `y`
            y = np.exp(-z) - 5 * np.exp(-10 * z)

            # Calculating Biggs Exponential's 3rd function
            f += (np.exp(-z * x[0]) - x[2] * np.exp(-z * x[1]) - y) ** 2

        return f


class BiggsExponential4(Benchmark):
    """BiggsExponential4 class implements the Biggs Exponential's 4th benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3, x_4) = \sum_{i=1}^{10}(x_3e^{-t_ix_1} - x_4e^{-t_ix_2} - y_i)^2

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 20] \mid i = \{1, 2, 3, 4\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 10, 1, 5)`.

    """

    def __init__(
        self,
        name: Optional[str] = "BiggsExponential4",
        dims: Optional[int] = 4,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = True,
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

        super(BiggsExponential4, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Instantiating function
        f = 0

        # For `i` ranging from 1 to 10
        for i in range(1, 11):
            # Calculating `z`
            z = i / 10

            # Calculating partial `y`
            y = np.exp(-z) - 5 * np.exp(-10 * z)

            # Calculating Biggs Exponential's 4th function
            f += (x[2] * np.exp(-z * x[0]) - x[3] * np.exp(-z * x[1]) - y) ** 2

        return f


class BiggsExponential5(Benchmark):
    """BiggsExponential5 class implements the Biggs Exponential's 5th benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3, x_4, x_5) = \sum_{i=1}^{11}(x_3e^{-t_ix_1} - x_4e^{-t_ix_2} + 3e^{-t_ix_5} - y_i)^2

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 20] \mid i = \{1, 2, 3, 4, 5\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 10, 1, 5, 4)`.

    """

    def __init__(
        self,
        name: Optional[str] = "BiggsExponential5",
        dims: Optional[int] = 5,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = True,
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

        super(BiggsExponential5, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Instantiating function
        f = 0

        # For `i` ranging from 1 to 11
        for i in range(1, 12):
            # Calculating `z`
            z = i / 10

            # Calculating partial `y`
            y = np.exp(-z) - 5 * np.exp(-10 * z) + 3 * np.exp(-4 * z)

            # Calculating Biggs Exponential's 5th function
            f += (
                x[2] * np.exp(-z * x[0])
                - x[3] * np.exp(-z * x[1])
                + 3 * np.exp(-z * x[4])
                - y
            ) ** 2

        return f


class BiggsExponential6(Benchmark):
    """BiggsExponential6 class implements the Biggs Exponential's 6th benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3, x_4, x_5, x_6) = \sum_{i=1}^{13}(x_3e^{-t_ix_1} - x_4e^{-t_ix_2} + x_6e^{-t_ix_5} - y_i)^2

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 20] \mid i = \{1, 2, 3, 4, 5, 6\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 10, 1, 5, 4, 3)`.

    """

    def __init__(
        self,
        name: Optional[str] = "BiggsExponential6",
        dims: Optional[int] = 6,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = True,
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

        super(BiggsExponential6, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Instantiating function
        f = 0

        # For `i` ranging from 1 to 13
        for i in range(1, 14):
            # Calculating `z`
            z = i / 10

            # Calculating partial `y`
            y = np.exp(-z) - 5 * np.exp(-10 * z) + 3 * np.exp(-4 * z)

            # Calculating Biggs Exponential's 6th function
            f += (
                x[2] * np.exp(-z * x[0])
                - x[3] * np.exp(-z * x[1])
                + x[5] * np.exp(-z * x[4])
                - y
            ) ** 2

        return f


class BoxBetts(Benchmark):
    """BoxBetts class implements the BoxBetts's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3) = \sum_{i=1}^{n}g(x)^2
    .. math:: g(x) = e^{-0.1(i+1)x_1} - e^{-0.1(i+1)x_2} - (e^{-0.1(i+1)} - e^{-(i+1)}*x_3)

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [0.9, 1.2], x_2 \in [9, 11.2], x_3 \in [0.9, 1.2]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 10, 1)`.

    """

    def __init__(
        self,
        name: Optional[str] = "BoxBetts",
        dims: Optional[int] = 3,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = True,
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

        super(BoxBetts, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Instantiating function
        f = 0

        # For every input dimension
        for i in range(x.shape[0]):
            # Calculating the BoxBetts's function
            f += (
                np.exp(-0.1 * (i + 2) * x[0])
                - np.exp(-0.1 * (i + 2) * x[1])
                - (np.exp(-0.1 * (i + 2)) - np.exp(-(i + 2)) * x[2])
            ) ** 2

        return f


class Colville(Benchmark):
    """Colville class implements the Colville's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3, x_4) = 100(x_1 - x_2^2)^2 + (1 - x_1)^2 + 90(x_4 - x_3^2)^2 + (1 - x_3)^2 + 10.1((x_2-1)^2 + (x_4-1)^2) + 19.8(x_2-1)(x_4-1)

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, 3, 4\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 1, 1, 1)`.

    """

    def __init__(
        self,
        name: Optional[str] = "Colville",
        dims: Optional[int] = 4,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = True,
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

        super(Colville, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Calculating the Colville's function
        f = (
            100 * (x[0] - x[1] ** 2) ** 2
            + (1 - x[0]) ** 2
            + 90 * (x[3] - x[2] ** 2) ** 2
            + (1 - x[2]) ** 2
            + 10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2)
            + 19.8 * (x[1] - 1) * (x[3] - 1)
        )

        return f


class GulfResearch(Benchmark):
    """GulfResearch class implements the GulfResearch's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3) = \sum_{i=1}^{99}[e^(-\\frac{(u_i-x_2)^x_3}{x_1}) - 0.01i]^2

    Domain:
        The function is commonly evaluated using :math:`x_1 \in [0.1, 100], x_2 \in [0, 25.6], x_3 \in [0, 5]`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (50, 25, 1.5)`.

    """

    def __init__(
        self,
        name: Optional[str] = "GulfResearch",
        dims: Optional[int] = 3,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = True,
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

        super(GulfResearch, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Instantiating function
        f = 0

        # For `i` ranging from 1 to 99
        for i in range(1, 100):
            # Calculating `u`
            u = 25 + (-50 * np.log(0.01 * i)) ** (1 / 1.5)

            # Calculating the GulfResearch's function
            f += (np.exp(-((u - x[1]) ** x[2]) / x[0]) - 0.01 * i) ** 2

        return f


class HelicalValley(Benchmark):
    """HelicalValley class implements the Helical Valley's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3) = 100[(x_3-10\\theta)^2 + (\\sqrt{x_1^2+x_2^2}-1)] + x_3^2

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, 3\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 0, 0)`.

    """

    def __init__(
        self,
        name: Optional[str] = "HelicalValley",
        dims: Optional[int] = 3,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = True,
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

        super(HelicalValley, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Checking whether x[0] is bigger or equal to zero
        if x[0] >= 0:
            # Calculating theta
            theta = np.arctan(x[1] / x[0])

        # Checking whether x[0] is smaller than zero
        else:
            # Calculating theta
            theta = np.pi + np.arctan(x[1] / x[0])

        # Calculating the Helical Valley's function
        f = (
            100 * (x[2] - 10 * theta) ** 2
            + (np.sqrt(x[0] ** 2 + x[1] ** 2) - 1) ** 2
            + x[2] ** 2
        )

        return f


class MieleCantrell(Benchmark):
    """MieleCantrell class implements the MieleCantrell's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3, x_4) = (e^{-x_1} - x_2)^4 + 100(x_2 - x_3)^6 + (tan(x_3-x_4))^4 + x_1^8

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-1, 1] \mid i = \{1, 2, 3, 4\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 1, 1, 1)`.

    """

    def __init__(
        self,
        name: Optional[str] = "MieleCantrell",
        dims: Optional[int] = 4,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = True,
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

        super(MieleCantrell, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Calculating the MieleCantrell's function
        f = (
            (np.exp(-x[0]) - x[1]) ** 4
            + 100 * (x[1] - x[2]) ** 6
            + (np.tan(x[2] - x[3])) ** 4
            + x[0] ** 8
        )

        return f


class Mishra9(Benchmark):
    """Mishra9 class implements the Mishra's 9th benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3) = (ab^2c + abc^2 + b^2 + (x_1 + x_2 - x_3)^2)^2

    Domain:
        The function is commonly evaluated using :math:`x_i \in [-10, 10] \mid i = \{1, 2, 3\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (1, 2, 3)`.

    """

    def __init__(
        self,
        name: Optional[str] = "Mishra9",
        dims: Optional[int] = 3,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
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

        super(Mishra9, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Calculating `a`
        a = 2 * (x[0] ** 3) + 5 * x[0] * x[1] + 4 * x[2] - 2 * (x[0] ** 2) * x[2] - 18

        # Calculating `b`
        b = x[0] + (x[1] ** 2) * x[2] + x[0] * (x[2] ** 2) - 22

        # Calculating `c`
        c = 8 * (x[0] ** 2) + 2 * x[1] * x[2] + 2 * (x[1] ** 2) + 3 * (x[1] ** 3) - 52

        # Calculating the Mishra's 9th function
        f = (
            a * (b**2) * c + a * b * (c**2) + (b**2) + (x[0] + x[1] - x[2]) ** 2
        ) ** 2

        return f


class Paviani(Benchmark):
    """Paviani class implements the Paviani's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, \ldots, x_10) = \sum_{i=1}^{10}[ln(x_i-2)^2 + ln(10-x_i)^2] - (\prod_{i=1}^{10}x_i)^{0.2}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [2.0001, 10] \mid i = \{1, 2, \ldots, 10\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = -45.778452053828865 \mid \mathbf{x^*} = (9.351, 9.351, \ldots, 9.351)`.

    """

    def __init__(
        self,
        name: Optional[str] = "Paviani",
        dims: Optional[int] = 10,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
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

        super(Paviani, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Instantiating summatory term
        sum_term = 0

        # Instantiating produtory term
        prod_term = 1

        # For every input dimension
        for i in range(x.shape[0]):
            # Calculating summatory term
            sum_term += np.log(x[i] - 2) ** 2 + np.log(10 - x[i]) ** 2

            # Calculating produtory term
            prod_term *= x[i]

        # Calculating the Paviani's function
        f = sum_term - prod_term**0.2

        return f


class SchmidtVetters(Benchmark):
    """SchmidtVetters class implements the Schmidt Vetters's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3) = \\frac{1}{1 + (x_1-x_2)^2} + sin(\\frac{\\pi x_2+x_3}{2}) + e^{(\\frac{x_1+x_2}{x_2}-2)^2}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 2] \mid i = \{1, 2, 3\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 3 \mid \mathbf{x^*} = (0.78547, 0.78547, 0.78547)`.

    """

    def __init__(
        self,
        name: Optional[str] = "SchmidtVetters",
        dims: Optional[int] = 3,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
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

        super(SchmidtVetters, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Calculating the Schmidt Vetters's function
        f = (
            1 / (1 + (x[0] - x[1]) ** 2)
            + np.sin((np.pi ** x[1] + x[2]) / 2)
            + np.exp((x[0] + x[1]) / x[1] - 2) ** 2
        )

        return f


class Simpleton(Benchmark):
    """Simpleton class implements the Simpleton's Problem benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10) = \\frac{x_1 x_2 x_3 x_4 x_5}{x_6 x_7 x_8 x_9 x_10}

    Domain:
        The function is commonly evaluated using :math:`x_i \in [1, 10] \mid i = \{1, 2, \ldots, 10\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 10^5 \mid \mathbf{x^*} = (10, 10, 10, 10, 10, 1, 1, 1, 1, 1)`.

    """

    def __init__(
        self,
        name: Optional[str] = "Simpleton",
        dims: Optional[int] = 10,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
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

        super(Simpleton, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Calculating the Simpleton's Problem's function
        f = (x[0] * x[1] * x[2] * x[3] * x[4]) / (x[5] * x[6] * x[7] * x[8] * x[9])

        return f


class Watson(Benchmark):
    """Watson class implements the Watson's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3, x_4, x_5, x_6) = \sum_{i=0}^{29}{\sum_{j=0}^{4}((j-1)a_i^j x_{j+1}) - [\sum_{j=0}^{5}a_i^j x_{j+1}]^2 - 1}^2 + x_1^2

    Domain:
        The function is commonly evaluated using :math:`|x_i| \leq 10 \mid i = \{1, 2, 3, 4, 5, 6\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0.002288 \mid \mathbf{x^*} = (−0.0158, 1.012, −0.2329, 1.260, −1.513, 0.9928)`.

    """

    def __init__(
        self,
        name: Optional[str] = "Watson",
        dims: Optional[int] = 6,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
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

        super(Watson, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Instantiating function
        f = 0

        # For `i` ranging from 0 to 29
        for i in range(30):
            # Instanciating outer and inner summatories
            outer_sum, inner_sum = 0, 0

            # Calculating `a`
            a = i / 29

            # For `j` ranging from 0 to 4
            for j in range(2, 7):
                # Calculating outer summatory
                outer_sum += (j - 1) * (a ** (j - 2)) * x[j - 1]

            # For `j` ranging from 0 to 5
            for j in range(1, 7):
                # Calculating inner summatory
                inner_sum += (a ** (j - 1)) * x[j - 1]

            # Calculating partial Watson's function
            f += (outer_sum - inner_sum**2 - 1) ** 2

        # Calculating final Watson's function
        f += x[0] ** 2

        return f


class Wolfe(Benchmark):
    """Wolfe class implements the Wolfe's benchmarking function.

    .. math:: f(\mathbf{x}) = f(x_1, x_2, x_3) = \\frac{4}{3}(x_1^2 + x_2^2 - x_1x_2)^{0.75} + x_3

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 2] \mid i = \{1, 2, 3\}`.

    Global Minima:
        :math:`f(\mathbf{x^*}) = 0 \mid \mathbf{x^*} = (0, 0, 0)`.

    """

    def __init__(
        self,
        name: Optional[str] = "Wolfe",
        dims: Optional[int] = 3,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = True,
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

        super(Wolfe, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        # Calculating the Wolfe's function
        f = 4 / 3 * ((x[0] ** 2 + x[1] ** 2 - x[0] * x[1]) ** 0.75) + x[2]

        return f
