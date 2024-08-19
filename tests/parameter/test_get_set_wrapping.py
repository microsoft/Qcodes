from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pytest

from qcodes.parameters import Parameter, ParameterBase

from .conftest import (
    GetSetRawParameter,
    OverwriteGetParam,
    OverwriteSetParam,
    ParameterMemory,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def test_parameter_with_overwritten_get_raises() -> None:
    """
    Test that creating a parameter that overwrites get and set raises runtime errors
    """

    with pytest.raises(RuntimeError) as record:
        OverwriteGetParam(name="foo")
    assert "Overwriting get in a subclass of ParameterBase: foo is not allowed." == str(
        record.value
    )


def test_parameter_with_overwritten_set_raises() -> None:
    """
    Test that creating a parameter that overwrites get and set raises runtime errors
    """
    with pytest.raises(RuntimeError) as record:
        OverwriteSetParam(name="foo")
    assert "Overwriting set in a subclass of ParameterBase: foo is not allowed." == str(
        record.value
    )


@pytest.mark.parametrize(
    "get_cmd, set_cmd",
    [
        (False, False),
        (False, None),
        (None, None),
        (None, False),
        (lambda: 1, lambda x: x),
    ],
)
def test_gettable_settable_attributes_with_get_set_cmd(
    get_cmd: Literal[False] | None | Callable, set_cmd: Literal[False] | None | Callable
) -> None:
    a = Parameter(name="foo", get_cmd=get_cmd, set_cmd=set_cmd)
    expected_gettable = get_cmd is not False
    expected_settable = set_cmd is not False

    assert a.gettable is expected_gettable
    assert a.settable is expected_settable


@pytest.mark.parametrize("baseclass", [ParameterBase, Parameter])
def test_gettable_settable_attributes_with_get_set_raw(
    baseclass: type[ParameterBase],
) -> None:
    """Test that parameters that have get_raw,set_raw are
    listed as gettable/settable and reverse."""

    class GetSetParam(baseclass):  # type: ignore[valid-type,misc]
        def __init__(self, *args: Any, initial_value: Any = None, **kwargs: Any):
            self._value = initial_value
            super().__init__(*args, **kwargs)

        def get_raw(self) -> Any:
            return self._value

        def set_raw(self, value: Any) -> Any:
            self._value = value

    a = GetSetParam("foo", instrument=None, initial_value=1)

    assert a.gettable is True
    assert a.settable is True

    b = ParameterBase("foo", instrument=None)

    assert b.gettable is False
    assert b.settable is False


@pytest.mark.parametrize("working_get_cmd", (False, None))
@pytest.mark.parametrize("working_set_cmd", (False, None))
def test_get_raw_and_get_cmd_raises(
    working_get_cmd: Literal[False] | None, working_set_cmd: Literal[False] | None
) -> None:
    with pytest.raises(TypeError, match="get_raw"):
        GetSetRawParameter(
            name="param1", get_cmd="GiveMeTheValue", set_cmd=working_set_cmd
        )
    with pytest.raises(TypeError, match="set_raw"):
        GetSetRawParameter(
            name="param2", set_cmd="HereIsTheValue {}", get_cmd=working_get_cmd
        )
    GetSetRawParameter("param3", get_cmd=working_get_cmd, set_cmd=working_set_cmd)


def test_get_on_parameter_marked_as_non_gettable_raises() -> None:
    a = Parameter("param")
    a._gettable = False
    with pytest.raises(
        TypeError, match="Trying to get a parameter that is not gettable."
    ):
        a.get()


def test_set_on_parameter_marked_as_non_settable_raises() -> None:
    a = Parameter("param", set_cmd=None)
    a.set(2)
    assert a.get() == 2
    a._settable = False
    with pytest.raises(
        TypeError, match="Trying to set a parameter that is not settable."
    ):
        a.set(1)
    assert a.get() == 2


def test_settable() -> None:
    mem = ParameterMemory()

    p = Parameter("p", set_cmd=mem.set, get_cmd=False)

    p(10)
    assert mem.get() == 10
    with pytest.raises(NotImplementedError):
        p()

    assert hasattr(p, "set")
    assert p.settable
    assert not hasattr(p, "get")
    assert not p.gettable

    # For settable-only parameters, using ``cache.set`` may not make
    # sense, nevertheless, it works
    p.cache.set(7)
    assert p.get_latest() == 7


def test_gettable() -> None:
    mem = ParameterMemory()
    p = Parameter("p", get_cmd=mem.get)
    mem.set(21)

    assert p() == 21
    assert p.get() == 21

    with pytest.raises(NotImplementedError):
        p(10)

    assert hasattr(p, "get")
    assert p.gettable
    assert not hasattr(p, "set")
    assert not p.settable

    p.cache.set(7)
    assert p.get_latest() == 7
    # Nothing has been passed to the "instrument" at ``cache.set``
    # call, hence the following assertions should hold
    assert mem.get() == 21
    assert p() == 21
    assert p.get_latest() == 21
