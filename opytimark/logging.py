# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a logger without configuring application output.

    Args:
        name: Fully qualified name of the module requesting the logger.

    Returns:
        Standard-library logger with application-controlled handlers, levels, and propagation.

    """

    return logging.getLogger(name)
