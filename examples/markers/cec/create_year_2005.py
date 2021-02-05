import numpy as np

from opytimark.markers.cec.year_2005 import F1

# Declaring a function from the `cec/year_2005` package
f = F1()

# Declaring an input variable for feeding the function
x = np.array([-39.311900, 58.899900, -46.322400, -74.651500, -16.799700])

# Printing out the function's output
print(f(x))
