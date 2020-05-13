import numpy as np

from opytimark.markers.many_dimensional import Wolfe

# Declaring a function from the `many_dimensional` package
f = Wolfe()

# Declaring an input variable for feeding the function
x = np.array([0, 0, 0])

# Printing out the function's output
print(f(x))
