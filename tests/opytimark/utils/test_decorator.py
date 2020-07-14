import numpy as np

from opytimark.markers import n_dimensional
from opytimark.utils import decorator


def test_check_dimension():
    @decorator.check_dimension
    def call(obj, x):
        return x

    f = n_dimensional.Sphere()

    try:
        call(f, np.array([]))
    except:
        call(f, np.array([1]))

    f.dims = 1

    try:
        call(f, np.array([1, 2]))
    except:
        call(f, np.array([1]))
