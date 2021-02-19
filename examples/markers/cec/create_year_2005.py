import numpy as np

from opytimark.markers.cec.year_2005 import F15

# Declaring a function from the `cec/year_2005` package
f = F15()

# Declaring an input variable for feeding the function
x = np.array([-39.311900, 58.899900])

# Printing out the function's output
print(f(x))
