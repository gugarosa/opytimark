import numpy as np

from opytimark.markers.two_dimensional import Adjiman

# Declaring a function from the `two_dimensional` package
f = Adjiman()

# Declaring an input variable for feeding the function
x = np.array([2, 0.10578])

# Printing out the function's output
print(f(x))
