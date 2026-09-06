# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "opytimark.markers.n_dimensional",
        "opytimark.markers.cec.year_2005",
        "opytimark.markers.cec.year_2008",
        "opytimark.markers.cec.year_2010",
        "opytimark.markers.cec.year_2013",
    ],
)
def test_import_preserves_caller_random_state(module):
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import sys
import numpy as np

np.random.seed(12345)
expected = np.random.RandomState(12345).random_sample(4)
importlib.import_module(sys.argv[1])
np.testing.assert_array_equal(np.random.random_sample(4), expected)
""",
            module,
        ],
        check=True,
    )
