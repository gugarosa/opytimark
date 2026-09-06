# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import numpy as np

from opytimark.markers.one_dimensional import GramacyLee

f = GramacyLee()

x = np.array([0.54856344411452])

print(f(x))
