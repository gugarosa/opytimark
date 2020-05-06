import numpy as np

from opytimark.markers.n_dimensional import Sphere

# Declaring a function from the `n_dimensional` package
f = Sphere()

# Declaring an input variable for feeding the function
x = np.zeros(50)

# Printing out the function's output
print(f(x))
