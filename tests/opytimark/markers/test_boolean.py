import numpy as np
import pytest

from opytimark.markers.boolean import Knapsack
from opytimark.utils.constants import FLOAT_MAX


def test_knapsack_validates_configuration():
    with pytest.raises(TypeError):
        Knapsack(values=[1], weights=(1,))
    with pytest.raises(TypeError):
        Knapsack(values=(1,), weights=[1])
    with pytest.raises(ValueError):
        Knapsack(values=(1, 2), weights=(1,))
    with pytest.raises(TypeError):
        Knapsack(max_capacity="1")
    with pytest.raises(ValueError):
        Knapsack(max_capacity=-1)


def test_knapsack_evaluates_capacity():
    benchmark = Knapsack(
        values=(55, 10, 47),
        weights=(95, 4, 60),
        max_capacity=100,
    )

    assert benchmark(np.array([0, 1, 0])) == -10
    assert benchmark(np.array([1, 0, 1])) == FLOAT_MAX
