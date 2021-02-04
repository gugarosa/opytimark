import numpy as np

from opytimark.markers.cec.year_2005 import F1

# Declaring a function from the `cec/year_2005` package
f = F1()

# Declaring an input variable for feeding the function
x = np.zeros(50)

# Printing out the function's output
print(f(x))
