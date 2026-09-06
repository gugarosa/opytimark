# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import io
import logging
import subprocess
import sys
from importlib import reload
from pathlib import Path

import opytimark
import opytimark.logging as library_logging
from opytimark.logging import get_logger


def test_get_logger_preserves_application_configuration(monkeypatch):
    logger = logging.getLogger("opytimark.test_logging.configured")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    monkeypatch.setattr(logger, "handlers", [handler])
    monkeypatch.setattr(logger, "level", logging.WARNING)
    monkeypatch.setattr(logger, "propagate", False)
    root = logging.getLogger()
    root_handlers = root.handlers[:]
    root_level = root.level

    configured = get_logger(logger.name)
    configured.warning("`application` owns this output.")

    assert configured is logger
    assert configured.handlers == [handler]
    assert configured.level == logging.WARNING
    assert configured.propagate is False
    assert stream.getvalue() == "`application` owns this output.\n"
    assert root.handlers == root_handlers
    assert root.level == root_level


def test_get_logger_keeps_unconfigured_children_propagating():
    logger = get_logger("opytimark.test_logging.unconfigured")

    assert logger.handlers == []
    assert logger.level == logging.NOTSET
    assert logger.propagate is True
    assert logging.getLogger("opytimark").propagate is True


def test_package_reload_does_not_accumulate_handlers():
    logger = logging.getLogger("opytimark")
    handlers = logger.handlers[:]
    root = logging.getLogger()
    root_handlers = root.handlers[:]
    root_level = root.level

    for _ in range(3):
        reload(library_logging)
        reload(opytimark)
        get_logger("opytimark.test_logging.reload")

    assert logger.handlers == handlers
    assert sum(isinstance(handler, logging.NullHandler) for handler in logger.handlers) == 1
    assert logger.level == logging.NOTSET
    assert logger.propagate is True
    assert root.handlers == root_handlers
    assert root.level == root_level


def test_unconfigured_library_use_has_no_stdout_or_stderr():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from opytimark.logging import get_logger
from opytimark.markers.n_dimensional import Sphere
from opytimark.utils.exception import Error, SizeError, TypeError, ValueError

Sphere()(x=[1, 2])
Error("CustomError", "unformatted message")
for error_type in (SizeError, TypeError, ValueError):
    error_type("message already punctuated.")
get_logger("opytimark.example").warning("`example` has a warning.")
""",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout == ""
    assert completed.stderr == ""
