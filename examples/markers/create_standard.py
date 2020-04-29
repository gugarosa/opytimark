import numpy as np
import opytimark.markers as m

# Declaring a function from the `markers` package
f = m.Sphere()

# Declaring an input variable for feeding the function
x = np.array([1, 1.5, 2, 2.5])

print(f(x))
