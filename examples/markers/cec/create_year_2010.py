# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import numpy as np

from opytimark.markers.cec.year_2010 import F1

f = F1()

x = np.array([-39.311900, 58.899900, -46.322400, -74.651500, -16.799700])

print(f(x))
