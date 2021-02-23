import numpy as np

from opytimark.markers.cec.year_2005 import F1, F15

# Declaring a function from the `cec/year_2005` package
f = F15()

# Declaring an input variable for feeding the function
x = np.array([3.3253000e+000, -1.2835000e+000, 1.8984000e+000, -4.0950000e-001, 8.8100000e-002, 2.7580000e+000, 9.7760000e-001, -1.8090000e+000, -2.4957000e+000, 2.7367000e+000])

# Printing out the function's output
print(f(x))
