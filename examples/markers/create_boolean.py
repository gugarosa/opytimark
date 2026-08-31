import numpy as np

from opytimark.markers.boolean import Knapsack

f = Knapsack(values=(55, 10, 47, 5, 4), weights=(95, 4, 60, 32, 23), max_capacity=100)
x = np.array([0, 1, 1, 1, 0])
print(f(x))
