# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from logging import NullHandler

from opytimark.logging import get_logger

__version__ = "3.0.1"

_logger = get_logger(__name__)
if not any(isinstance(handler, NullHandler) for handler in _logger.handlers):
    _logger.addHandler(NullHandler())
