import numpy as np

from opytimark.markers.bi_dim import Adjiman

# Declaring a function from the `bi_dim` package
f = Adjiman()

# Declaring an input variable for feeding the function
x = np.array([2, 0.10578])

# Printing out the function's output
print(f(x))
