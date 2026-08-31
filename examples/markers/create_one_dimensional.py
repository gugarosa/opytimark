import numpy as np

from opytimark.markers.one_dimensional import GramacyLee

f = GramacyLee()
x = np.array([0.54856344411452])
print(f(x))
