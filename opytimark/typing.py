# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Describe benchmark results without coercing their precision or shape.

Notes:
    Normalized vectors usually produce scalar results, while historical
    higher-rank inputs can produce arrays.

"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

BenchmarkValue = int | float | np.number[Any] | NDArray[Any]
