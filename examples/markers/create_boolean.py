import numpy as np

from opytimark.markers.boolean import Knapsack

# Declaring a function from the `boolean` package
f = Knapsack(values=(55, 10, 47, 5, 4), weights=(95, 4, 60, 32, 23), max_capacity=100)

# Declaring an input variable for feeding the function
x = np.array([0, 1, 1, 1, 0])

# Printing out the function's output
print(f(x))
