import numpy as np

from opytimark.markers.many_dimensional import Wolfe

f = Wolfe()
x = np.array([0, 0, 0])
print(f(x))
