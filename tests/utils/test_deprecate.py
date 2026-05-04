from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, TypeVar

import pytest

from qcodes.utils.deprecate import (
    QCoDeSDeprecationWarning,
    _make_deprecated_typevars_getattr,
)

_FixtureT = tuple[dict[str, TypeVar], Callable[[str], Any]]


@pytest.fixture
def deprecated_getattr() -> _FixtureT:
    """Create a __getattr__ with two deprecated TypeVars."""
    deprecated = {
        "MyT": TypeVar("MyT"),
        "MyK": TypeVar("MyK", bound=int),
    }
    getattr_fn = _make_deprecated_typevars_getattr("fake.module", deprecated)
    return deprecated, getattr_fn


def test_returns_typevar_and_warns(
    deprecated_getattr: _FixtureT,
) -> None:
    deprecated, getattr_fn = deprecated_getattr
    with pytest.warns(QCoDeSDeprecationWarning, match="'MyT'.*'fake.module'"):
        result = getattr_fn("MyT")
    assert result is deprecated["MyT"]


def test_warns_for_each_deprecated_name(
    deprecated_getattr: _FixtureT,
) -> None:
    deprecated, getattr_fn = deprecated_getattr
    with pytest.warns(QCoDeSDeprecationWarning, match="'MyK'"):
        result = getattr_fn("MyK")
    assert result is deprecated["MyK"]


def test_unknown_name_raises_attribute_error(
    deprecated_getattr: _FixtureT,
) -> None:
    _, getattr_fn = deprecated_getattr
    with pytest.raises(AttributeError, match=r"fake\.module"):
        getattr_fn("DoesNotExist")


def test_repeated_access_returns_same_object(
    deprecated_getattr: _FixtureT,
) -> None:
    deprecated, getattr_fn = deprecated_getattr
    with pytest.warns(QCoDeSDeprecationWarning):
        first = getattr_fn("MyT")
    with pytest.warns(QCoDeSDeprecationWarning):
        second = getattr_fn("MyT")
    assert first is second
    assert first is deprecated["MyT"]


def test_fallback_is_called_for_unknown_names() -> None:
    deprecated: dict[str, TypeVar] = {"X": TypeVar("X")}

    def fallback(name: str) -> str:
        return f"fallback:{name}"

    getattr_fn = _make_deprecated_typevars_getattr("mod", deprecated, fallback=fallback)
    assert getattr_fn("something") == "fallback:something"


def test_fallback_not_called_for_deprecated_names() -> None:
    deprecated: dict[str, TypeVar] = {"X": TypeVar("X")}
    fallback_called = False

    def fallback(name: str) -> str:
        nonlocal fallback_called
        fallback_called = True
        return name

    getattr_fn = _make_deprecated_typevars_getattr("mod", deprecated, fallback=fallback)
    with pytest.warns(QCoDeSDeprecationWarning):
        getattr_fn("X")
    assert not fallback_called


def test_real_module_import_triggers_warning() -> None:
    """Test that importing a deprecated TypeVar from an actual module works."""
    mod = importlib.import_module("qcodes.utils.deep_update_utils")
    with pytest.warns(QCoDeSDeprecationWarning, match="'K'"):
        k = mod.K

    assert isinstance(k, TypeVar)
