import numpy as np

from opytimark.markers.n_dimensional import PowellSum, Sphere

# Declaring a function from the `n_dimensional` package
f = PowellSum()

# Declaring an input variable for feeding the function
x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

# Printing out the function's output
print(f(x))
