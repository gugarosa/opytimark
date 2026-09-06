# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import numpy as np

from opytimark.markers.many_dimensional import Wolfe

f = Wolfe()

x = np.array([0, 0, 0])

print(f(x))
