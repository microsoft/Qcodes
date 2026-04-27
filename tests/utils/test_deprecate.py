"""
Tests for qcodes.utils.deprecate - QCoDeSDeprecationWarning.
"""

import warnings

import pytest

from qcodes.utils.deprecate import QCoDeSDeprecationWarning


def test_is_subclass_of_runtime_warning() -> None:
    """Test that QCoDeSDeprecationWarning is a subclass of RuntimeWarning."""
    assert issubclass(QCoDeSDeprecationWarning, RuntimeWarning)


def test_is_not_subclass_of_deprecation_warning() -> None:
    """Test that it is not a subclass of the standard DeprecationWarning."""
    assert not issubclass(QCoDeSDeprecationWarning, DeprecationWarning)


def test_can_be_raised_and_caught() -> None:
    """Test that QCoDeSDeprecationWarning can be raised and caught."""
    with pytest.raises(QCoDeSDeprecationWarning, match="test message"):
        raise QCoDeSDeprecationWarning("test message")


def test_can_be_caught_as_runtime_warning() -> None:
    """Test that it can be caught as a RuntimeWarning."""
    with pytest.raises(RuntimeWarning):
        raise QCoDeSDeprecationWarning("test message")


def test_can_be_used_with_warnings_warn() -> None:
    """Test that it can be used with warnings.warn."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("deprecation message", QCoDeSDeprecationWarning, stacklevel=1)

    assert len(caught) == 1
    assert issubclass(caught[0].category, QCoDeSDeprecationWarning)
    assert "deprecation message" in str(caught[0].message)


def test_instance_attributes() -> None:
    """Test that the warning carries its message."""
    warning = QCoDeSDeprecationWarning("my message")
    assert str(warning) == "my message"
    assert isinstance(warning, RuntimeWarning)


def test_not_suppressed_by_default_warning_filters() -> None:
    """Test that QCoDeSDeprecationWarning is visible with default filters.

    Standard DeprecationWarning is suppressed by default, but since this
    inherits from RuntimeWarning it should not be.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        warnings.warn("should be visible", QCoDeSDeprecationWarning, stacklevel=1)

    assert len(caught) == 1
