# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import numpy as np

from opytimark.markers.two_dimensional import Adjiman

f = Adjiman()

x = np.array([2, 0.10578])

print(f(x))
