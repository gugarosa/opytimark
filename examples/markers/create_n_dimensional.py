# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import numpy as np

from opytimark.markers.n_dimensional import Sphere

f = Sphere()

x = np.zeros(50)

print(f(x))
