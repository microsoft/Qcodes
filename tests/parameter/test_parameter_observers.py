from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import pytest

from qcodes.parameters import (
    DelegateParameter,
    Parameter,
    ParameterAccess,
    add_parameter_observer,
    observe_parameters,
    registered_parameter_observers,
    remove_parameter_observer,
)
from qcodes.validators import Numbers

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def accesses() -> Iterator[list[ParameterAccess]]:
    recorded: list[ParameterAccess] = []
    with observe_parameters(recorded.append):
        yield recorded


def test_observer_sees_get_and_set(accesses: list[ParameterAccess]) -> None:
    param = Parameter("p", set_cmd=None, get_cmd=None, initial_value=0.0)
    accesses.clear()

    param(1.5)
    assert param() == 1.5

    assert [(a.parameter, a.kind, a.value) for a in accesses] == [
        (param, "set", 1.5),
        (param, "get", 1.5),
    ]
    assert all(a.exception is None for a in accesses)
    assert all(a.duration >= 0 for a in accesses)
    assert all(a.timestamp > 0 for a in accesses)


def test_observer_sees_raw_values(accesses: list[ParameterAccess]) -> None:
    raw = {"value": 0.0}
    param = Parameter(
        "p",
        set_cmd=lambda value: raw.update(value=value),
        get_cmd=lambda: raw["value"],
        scale=10,
    )

    param(2)
    assert param() == 2

    set_access, get_access = accesses
    assert (set_access.value, set_access.raw_value) == (2, 20)
    assert (get_access.value, get_access.raw_value) == (2, 20)


def test_observer_sees_every_ramp_step(accesses: list[ParameterAccess]) -> None:
    param = Parameter("p", set_cmd=None, get_cmd=None, initial_value=0)
    param.step = 1
    accesses.clear()

    param(3)

    assert [(a.kind, a.value) for a in accesses] == [
        ("set", 1),
        ("set", 2),
        ("set", 3),
    ]


def test_observer_sees_failed_get(accesses: list[ParameterAccess]) -> None:
    def fail() -> float:
        raise RuntimeError("instrument did not answer")

    param = Parameter("p", get_cmd=fail)

    with pytest.raises(RuntimeError, match="instrument did not answer"):
        param()

    (access,) = accesses
    assert access.kind == "get"
    assert access.value is None
    assert access.raw_value is None
    assert isinstance(access.exception, RuntimeError)


def test_observer_sees_failed_set(accesses: list[ParameterAccess]) -> None:
    def fail(value: float) -> None:
        raise RuntimeError("instrument refused")

    param = Parameter("p", set_cmd=fail)

    with pytest.raises(RuntimeError, match="instrument refused"):
        param(1)

    (access,) = accesses
    assert (access.kind, access.value) == ("set", 1)
    assert isinstance(access.exception, RuntimeError)


def test_invalid_value_is_not_an_access(accesses: list[ParameterAccess]) -> None:
    param = Parameter("p", set_cmd=None, vals=Numbers(0, 1))

    with pytest.raises(ValueError):
        param(5)

    assert accesses == []


def test_delegate_and_source_are_both_reported(
    accesses: list[ParameterAccess],
) -> None:
    source = Parameter("source", set_cmd=None, get_cmd=None, initial_value=0)
    delegate = DelegateParameter("delegate", source=source)
    accesses.clear()

    delegate(1)

    assert [(a.parameter, a.kind) for a in accesses] == [
        (source, "set"),
        (delegate, "set"),
    ]


def test_observer_is_independent_of_on_set_callback(
    accesses: list[ParameterAccess],
) -> None:
    callback_values: list[Any] = []
    param = Parameter(
        "p",
        set_cmd=None,
        on_set_callback=lambda _param, value: callback_values.append(value),
    )

    param(1)

    assert callback_values == [1]
    assert [(a.kind, a.value) for a in accesses] == [("set", 1)]


def test_failing_observer_does_not_affect_access(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken(access: ParameterAccess) -> None:
        raise ValueError("observer bug")

    recorded: list[ParameterAccess] = []
    param = Parameter("p", set_cmd=None, get_cmd=None)
    with observe_parameters(broken), observe_parameters(recorded.append):
        param(1)

    assert param.cache.get() == 1
    assert [a.kind for a in recorded] == ["set"]
    assert "Exception in parameter observer" in caplog.text


def test_add_and_remove_observer() -> None:
    recorded: list[ParameterAccess] = []
    param = Parameter("p", set_cmd=None, get_cmd=None)

    add_parameter_observer(recorded.append)
    assert recorded.append in registered_parameter_observers()
    param(1)
    remove_parameter_observer(recorded.append)
    param(2)

    assert [a.value for a in recorded] == [1]
    assert recorded.append not in registered_parameter_observers()


def test_remove_unknown_observer_raises() -> None:
    with pytest.raises(ValueError):
        remove_parameter_observer(lambda access: None)


def test_observers_are_called_on_the_accessing_thread(
    accesses: list[ParameterAccess],
) -> None:
    threads: list[threading.Thread] = []
    param = Parameter("p", set_cmd=None, get_cmd=None)

    with observe_parameters(lambda access: threads.append(threading.current_thread())):
        worker = threading.Thread(target=param, args=(1,))
        worker.start()
        worker.join()

    assert threads == [worker]
