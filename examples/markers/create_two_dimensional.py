import numpy as np

from opytimark.markers.two_dimensional import Adjiman

f = Adjiman()
x = np.array([2, 0.10578])
print(f(x))
