import numpy as np

from opytimark.markers.boolean import Knapsack

# Declaring a function from the `boolean` package
f = Knapsack()

# Declaring an input variable for feeding the function
x = np.array([1, 0, 1, 0])

# Printing out the function's output
print(f(x))
