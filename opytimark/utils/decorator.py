"""Decorators.
"""

from typing import Any

import numpy as np

import opytimark.utils.exception as e


def check_exact_dimension(f: callable) -> Any:
    """Checks whether the input dimension is exact to the demanded by the evaluated function.

    Args:
        f: Function to be checked.

    Returns:
        (Any): The function output or an error depending whether the check is valid.

    """

    def _check_exact_dimension(*args) -> Any:
        """Wraps the dimension checking in order to provide additional logic.

        Returns:
            (Any): The wrapped function output.

        """

        # Retrieving the object and the input from arguments
        obj, x = args[0], args[1]

        # Tries to squeeze the last dimension of `x` as it might be an array of (dim, 1)
        try:
            x = np.squeeze(x, axis=1)

        except ValueError:
            pass

        # If the function's number of dimensions is equal to `-1` (n-dimensional)
        if obj.dims == -1:
            if x.shape[0] == 0:
                raise e.SizeError(f"{obj.name} input should be n-dimensional")

            return f(obj, x)

        # If the input dimensions is different from function's allowed dimensions
        if x.shape[0] != obj.dims:
            raise e.SizeError(f"{obj.name} input should be {obj.dims}-dimensional")

        return f(obj, x)

    return _check_exact_dimension


def check_exact_dimension_and_auxiliary_matrix(f: callable) -> Any:
    """Checks whether the input dimension is exact to the demanded by the evaluated function and defines
    a proper auxiliary matrix accordingly.

    Args:
        f: Function to be checked.

    Returns:
        (Any): The function output or an error depending whether the check is valid.

    """

    def _check_exact_dimension_and_auxiliary_matrix(*args) -> Any:
        """Wraps the dimension checking in order to provide additional logic.

        Returns:
            (Any): The wrapped function output.

        """

        # Retrieving the object and the input from arguments
        obj, x = args[0], args[1]

        # Tries to squeeze the last dimension of `x` as it might be an array of (dim, 1)
        try:
            x = np.squeeze(x, axis=1)

        except ValueError:
            pass

        # If the input dimensions differs from function's allowed dimensions
        if x.shape[0] not in [2, 10, 30, 50]:
            raise e.SizeError(
                f"{obj.name} input should be 2-, 10-, 30- or 50-dimensional"
            )

        if x.shape[0] == 2:
            setattr(obj, "M", obj.M2)

        elif x.shape[0] == 10:
            setattr(obj, "M", obj.M10)

        elif x.shape[0] == 30:
            setattr(obj, "M", obj.M30)

        elif x.shape[0] == 50:
            setattr(obj, "M", obj.M50)

        return f(obj, x)

    return _check_exact_dimension_and_auxiliary_matrix


def check_less_equal_dimension(f: callable) -> Any:
    """Checks whether the input dimension is less or equal to the demanded by the evaluated function.

    Args:
        f: Function to be checked.

    Returns:
        (Any): The function output or an error depending whether the check is valid.

    """

    def _check_less_equal_dimension(*args) -> Any:
        """Wraps the dimension checking in order to provide additional logic.

        Returns:
            (Any): The wrapped function output.

        """

        # Retrieving the object and the input from arguments
        obj, x = args[0], args[1]

        # Tries to squeeze the last dimension of `x` as it might be an array of (dim, 1)
        try:
            x = np.squeeze(x, axis=1)

        except ValueError:
            pass

        # If the input dimensions is different from function's allowed dimensions
        if x.shape[0] > obj.dims:
            raise e.SizeError(
                f"{obj.name} input should be less or equal to {obj.dims}-dimensional"
            )

        return f(obj, x)

    return _check_less_equal_dimension
