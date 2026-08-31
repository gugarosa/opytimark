import numpy as np
import pytest

from opytimark.markers.boolean import Knapsack
from opytimark.utils import constants, exception


def test_knapsack_preserves_full_constructor():
    benchmark = Knapsack(
        "bag",
        -1,
        False,
        False,
        False,
        False,
        False,
        (55, 10, 47),
        (95, 4, 60),
        100,
    )

    assert benchmark.name == "bag"
    assert benchmark.dims == 3
    assert benchmark(np.array([0, 1, 0])) == -10
    assert benchmark(np.array([1, 0, 1])) == constants.FLOAT_MAX


@pytest.mark.parametrize(
    ("attribute", "value", "error"),
    [
        ("values", [1], exception.TypeError),
        ("weights", [1], exception.TypeError),
        ("max_capacity", "1", exception.TypeError),
        ("max_capacity", -1, exception.ValueError),
    ],
)
def test_knapsack_setter_validation(attribute, value, error):
    benchmark = Knapsack()

    with pytest.raises(error):
        setattr(benchmark, attribute, value)


def test_knapsack_rejects_mismatched_items():
    with pytest.raises(exception.SizeError):
        Knapsack(values=(1, 2), weights=(1,))
