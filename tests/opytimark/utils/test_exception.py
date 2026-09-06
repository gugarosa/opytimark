# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import logging

import pytest

from opytimark.utils import exception


@pytest.mark.parametrize(
    ("error_type", "arguments", "identifier"),
    [
        (exception.Error, ("CustomError", "error"), "CustomError"),
        (exception.SizeError, ("error",), "SizeError"),
        (exception.TypeError, ("error",), "TypeError"),
        (exception.ValueError, ("error",), "ValueError"),
    ],
)
def test_custom_exceptions(error_type, arguments, identifier, caplog, capsys):
    with caplog.at_level(logging.ERROR, logger="opytimark.utils.exception"):
        error = error_type(*arguments)

    with pytest.raises(exception.Error, match="error"):
        raise error

    assert str(error) == "error"
    assert error.args == ("error",)
    assert caplog.record_tuples == [
        ("opytimark.utils.exception", logging.ERROR, f"`exception={identifier}` was raised with message 'error'.")
    ]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.parametrize("message", ["already punctuated.", "two\nlines", "", None, ("tuple", 1)])
def test_custom_exception_payloads_are_not_rewritten_for_logging(message, caplog):
    with caplog.at_level(logging.ERROR, logger="opytimark.utils.exception"):
        error = exception.Error("DiagnosticOwner", message)

    assert error.args == (message,)
    assert str(error) == str(message)
    assert caplog.messages == [f"`exception=DiagnosticOwner` was raised with message {message!r}."]
