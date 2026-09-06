# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from opytimark.logging import get_logger

logger = get_logger(__name__)


class Error(Exception):
    """
    Base Opytimark exception with application-controlled diagnostics.

    """

    def __init__(self, cls: str, msg: object) -> None:
        """Initialize the exception without changing the supplied message payload.

        Args:
            cls: Exception identifier included in the diagnostic log record.
            msg: Original payload exposed through the exception's args and string representation.

        """

        super().__init__(msg)
        logger.error("`exception=%s` was raised with message %r.", cls, msg)


class SizeError(Error):
    """
    Report an invalid length or dimension.

    """

    def __init__(self, error: object) -> None:
        """Initialize a size error with its original payload.

        Args:
            error: Message payload describing the invalid length or dimension.

        """

        super().__init__("SizeError", error)


class TypeError(Error):
    """
    Report an invalid value type.

    """

    def __init__(self, error: object) -> None:
        """Initialize a type error with its original payload.

        Args:
            error: Message payload describing the invalid type.

        """

        super().__init__("TypeError", error)


class ValueError(Error):
    """
    Report an invalid value.

    """

    def __init__(self, error: object) -> None:
        """Initialize a value error with its original payload.

        Args:
            error: Message payload describing the invalid value.

        """

        super().__init__("ValueError", error)
