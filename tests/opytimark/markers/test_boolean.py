import numpy as np

from opytimark.markers import boolean
from opytimark.utils import constants


def test_knapsack_values():
    try:
        new_knapsack = boolean.Knapsack(values=[1, 2, 3], weights=[1, 2])
    except:
        new_knapsack = boolean.Knapsack()

    assert new_knapsack.values[0] == 0


def test_knapsack_values_setter():
    new_knapsack = boolean.Knapsack()

    try:
        new_knapsack.values = 1
    except:
        new_knapsack.values = [0]

    assert new_knapsack.values[0] == 0


def test_knapsack_weights():
    new_knapsack = boolean.Knapsack()

    assert new_knapsack.weights[0] == 0


def test_knapsack_weights_setter():
    new_knapsack = boolean.Knapsack()

    try:
        new_knapsack.weights = 1
    except:
        new_knapsack.weights = [0]

    assert new_knapsack.weights[0] == 0


def test_knapsack_max_capacity():
    new_knapsack = boolean.Knapsack()

    assert new_knapsack.max_capacity == 0


def test_knapsack_max_capacity_setter():
    new_knapsack = boolean.Knapsack()

    try:
        new_knapsack.max_capacity = 'a'
    except:
        new_knapsack.max_capacity = 0

    assert new_knapsack.max_capacity == 0

    try:
        new_knapsack.max_capacity = -1
    except:
        new_knapsack.max_capacity = 0

    assert new_knapsack.max_capacity == 0


def test_knapsack():
    new_knapsack = boolean.Knapsack(values=[1, 2, 3], weights=[1, 2, 3], max_capacity=2)

    assert new_knapsack(np.array([1, 0, 0])) == -1

    assert new_knapsack(np.array([1, 0, 1])) == constants.FLOAT_MAX
