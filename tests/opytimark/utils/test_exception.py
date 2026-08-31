import pytest

from opytimark.utils import exception


@pytest.mark.parametrize(
    "error",
    [
        exception.Error("Error", "error"),
        exception.SizeError("error"),
        exception.TypeError("error"),
        exception.ValueError("error"),
    ],
)
def test_custom_exceptions(error):
    with pytest.raises(exception.Error):
        raise error
