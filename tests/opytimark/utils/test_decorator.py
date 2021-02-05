import numpy as np

from opytimark.markers import n_dimensional
from opytimark.markers.cec import year_2005
from opytimark.utils import decorator


def test_check_exact_dimension():
    @decorator.check_exact_dimension
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


def test_check_less_equal_dimension():
    @decorator.check_less_equal_dimension
    def call(obj, x):
        return x

    f = year_2005.F1()

    try:
        call(f, np.zeros(101))
    except:
        call(f, np.zeros(100))
