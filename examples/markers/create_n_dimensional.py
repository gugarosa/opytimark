import numpy as np

from opytimark.markers.n_dimensional import Sphere

f = Sphere()
x = np.zeros(50)
print(f(x))
