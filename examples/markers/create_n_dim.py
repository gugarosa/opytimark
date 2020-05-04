import numpy as np

from opytimark.markers.n_dim import Sphere

# Declaring a function from the `n_dim` package
f = Sphere()

# Declaring an input variable for feeding the function
x = np.array([0, 0, 0, 0])

# Printing out the function's output
print(f(x))
