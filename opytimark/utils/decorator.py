import opytimark.utils.exception as e


def check_dimension(f):
    """
    """

    def wrapper(self, x):
        """
        """

        #
        name = getattr(self, 'name')

        #
        n_dims = getattr(self, 'dims')

        #
        if x.shape[0] != n_dims:
            #
            raise e.SizeError(f'{name} input should be {n_dims}-dimensional')

        return x

    return wrapper
