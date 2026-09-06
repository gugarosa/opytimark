# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import sys

from opytimark.utils import constants


def test_constants():
    assert constants.DATA_FOLDER == "data/"

    assert constants.EPSILON == 1e-32

    assert constants.FLOAT_MAX == sys.float_info.max
