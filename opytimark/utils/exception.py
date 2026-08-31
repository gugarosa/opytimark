"""Custom Opytimark exceptions."""


class Error(Exception):
    """Base Opytimark exception."""

    def __init__(self, cls, msg):
        super().__init__()
        print(f"{cls}: {msg}.")


class SizeError(Error):
    """Raised for invalid lengths or dimensions."""

    def __init__(self, error):
        super().__init__("SizeError", error)


class TypeError(Error):
    """Raised for invalid value types."""

    def __init__(self, error):
        super().__init__("TypeError", error)


class ValueError(Error):
    """Raised for invalid values."""

    def __init__(self, error):
        super().__init__("ValueError", error)
