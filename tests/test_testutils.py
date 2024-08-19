import pytest

from .common import error_caused_by


def test_error_caused_by_simple() -> None:
    """
    Test that `error_caused_by` will only match the root error and not
    the error raised. For errors raised directly from other errors
    """

    with pytest.raises(Exception) as execinfo:
        raise KeyError("foo") from ValueError("bar")

    assert error_caused_by(execinfo, "ValueError: bar")
    assert not error_caused_by(execinfo, "KeyError: foo")


def test_error_caused_by_try_catch() -> None:
    """
    Test that `error_caused_by` will only match the root error and not
    the error raised for errors. For errors reraised in a try except chain.
    """
    with pytest.raises(KeyError) as execinfo:
        try:
            raise ValueError("bar")
        except ValueError as e:
            raise KeyError("foo") from e

    assert error_caused_by(execinfo, "ValueError: bar")
    assert not error_caused_by(execinfo, "KeyError: foo")


def test_error_caused_by_3_level() -> None:
    """
    Test that error caused by does not match the middle element in a chain
    of 3 exceptions
    """

    with pytest.raises(RuntimeError) as execinfo:
        try:
            raise ValueError("bar")
        except ValueError as e:
            try:
                raise KeyError("foo") from e
            except KeyError as ee:
                raise RuntimeError("goo") from ee

    assert not error_caused_by(execinfo, "RuntimeError: goo")
    assert not error_caused_by(execinfo, "KeyError: foo")
    assert error_caused_by(execinfo, "ValueError: bar")
