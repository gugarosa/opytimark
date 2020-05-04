import numpy as np

from opytimark.markers.one_dimensional import GramacyLee

# Declaring a function from the `one_dimensional` package
f = GramacyLee()

# Declaring an input variable for feeding the function
x = np.array([0.54856344411452])

# Printing out the function's output
print(f(x))
