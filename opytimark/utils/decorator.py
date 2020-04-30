import opytimark.utils.exception as e


def check_dimension(f):
    """Checks whether the input dimension is suitable for the evaluated function or not.

    Args:
        f (callable): Function to be checked.

    Returns:
        The function output or an error depending whether the check is valid.

    """

    def _check_dimension(*args):
        """Wraps the dimension checking in order to provide additional logic.

        Returns:
            The wrapped function output.

        """

        # Retrieving the object and the input from arguments
        obj, x = args[0], args[1]

        # If the function's number of dimensions is equal to `-1` (n-dimensional)
        if obj.dims == -1:
            # Checks if the input array is bigger than zero
            if x.shape[0] == 0:
                # If not, raises an error
                raise e.SizeError(f'{obj.name} input should be n-dimensional')

            return f(*args)

        # If the input dimensions is different from function's allowed dimensions
        if x.shape[0] != obj.dims:
            # Raises an error
            raise e.SizeError(f'{obj.name} input should be {obj.dims}-dimensional')

        return f(*args)

    return _check_dimension
