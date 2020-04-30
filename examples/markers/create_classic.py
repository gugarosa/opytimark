import numpy as np

import opytimark.markers.classic as c

# Declaring a function from the `classic` package
f = c.Sphere()

# Declaring an input variable for feeding the function
x = np.array([1, 1.5, 2, 2.5])

# Printing out the function's output
print(f(x))
