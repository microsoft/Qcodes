from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
from collections import UserList
from typing import TYPE_CHECKING, Any

import pytest

from qcodes.instrument_drivers.mock_instruments import DummyInstrument
from qcodes.parameters import (
    DelegateParameter,
    Parameter,
    ParameterAccess,
    log_parameter_accesses,
    record_parameter_accesses,
)
from qcodes.validators import Numbers

if TYPE_CHECKING:
    from collections.abc import Generator

ACCESS_LOGGER = "qcodes.parameters.parameter_access"
ERROR_LOGGER = "qcodes.parameters.parameter_access_errors"


class _FailingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        raise ValueError("broken filter")


@pytest.fixture
def failing_access_log() -> Generator[None, None, None]:
    access_log = logging.getLogger(ACCESS_LOGGER)
    level = access_log.level
    failing = _FailingFilter()
    access_log.setLevel(logging.DEBUG)
    access_log.addFilter(failing)
    try:
        yield
    finally:
        access_log.removeFilter(failing)
        access_log.setLevel(level)


def test_records_get_and_set() -> None:
    param = Parameter("p", set_cmd=None, get_cmd=None, unit="V", initial_value=0.0)

    with record_parameter_accesses() as accesses:
        param(1.5)
        assert param() == 1.5

    assert [(a.parameter_name, a.kind, a.value, a.unit) for a in accesses] == [
        ("p", "set", 1.5, "V"),
        ("p", "get", 1.5, "V"),
    ]
    assert all(a.error is None for a in accesses)
    assert all(a.duration >= 0 for a in accesses)
    assert all(a.timestamp > 0 for a in accesses)


def test_records_names_not_objects() -> None:
    instrument = DummyInstrument("recorded_dac", gates=["ch1"])
    try:
        with record_parameter_accesses() as accesses:
            instrument.ch1(0.5)
    finally:
        instrument.close()

    (access,) = accesses
    assert access.parameter_name == "recorded_dac_ch1"
    assert access.instrument_name == "recorded_dac"
    # A record is data: it holds no reference to a parameter or instrument.
    for field_value in (access.parameter_name, access.instrument_name, access.unit):
        assert isinstance(field_value, str)
    with pytest.raises(AttributeError):
        access.parameter  # type: ignore[attr-defined]  # noqa: B018


def test_records_are_immutable() -> None:
    param = Parameter("p", set_cmd=None)
    with record_parameter_accesses() as accesses:
        param(1)

    with pytest.raises(AttributeError):
        accesses[0].value = 2  # type: ignore[misc]


def test_records_raw_values() -> None:
    raw = {"value": 0.0}
    param = Parameter(
        "p",
        set_cmd=lambda value: raw.update(value=value),
        get_cmd=lambda: raw["value"],
        scale=10,
    )

    with record_parameter_accesses() as accesses:
        param(2)
        assert param() == 2

    set_access, get_access = accesses
    assert (set_access.value, set_access.raw_value) == (2, 20)
    assert (get_access.value, get_access.raw_value) == (2, 20)


def test_records_every_ramp_step() -> None:
    param = Parameter("p", set_cmd=None, get_cmd=None, initial_value=0)
    param.step = 1

    with record_parameter_accesses() as accesses:
        param(3)

    assert [(a.kind, a.value) for a in accesses] == [
        ("set", 1),
        ("set", 2),
        ("set", 3),
    ]


def test_records_failed_get() -> None:
    def fail() -> float:
        raise RuntimeError("instrument did not answer")

    param = Parameter("p", get_cmd=fail)

    with (
        record_parameter_accesses() as accesses,
        pytest.raises(RuntimeError, match="instrument did not answer"),
    ):
        param()

    (access,) = accesses
    assert access.kind == "get"
    assert access.value is None
    assert access.raw_value is None
    assert access.error == "RuntimeError: instrument did not answer"


def test_records_error_that_cannot_be_described() -> None:
    class UnprintableError(Exception):
        def __str__(self) -> str:
            raise ValueError("no description")

    def fail() -> float:
        raise UnprintableError

    param = Parameter("p", get_cmd=fail)

    with (
        record_parameter_accesses() as accesses,
        pytest.raises(UnprintableError),
    ):
        param()

    (access,) = accesses
    assert access.error == "UnprintableError: <exception str() failed>"


def test_records_failed_set() -> None:
    def fail(value: float) -> None:
        raise RuntimeError("instrument refused")

    param = Parameter("p", set_cmd=fail)

    with (
        record_parameter_accesses() as accesses,
        pytest.raises(RuntimeError, match="instrument refused"),
    ):
        param(1)

    (access,) = accesses
    assert (access.kind, access.value) == ("set", 1)
    assert access.error == "RuntimeError: instrument refused"


def test_invalid_value_is_not_an_access() -> None:
    param = Parameter("p", set_cmd=None, vals=Numbers(0, 1))

    with record_parameter_accesses() as accesses, pytest.raises(ValueError):
        param(5)

    assert accesses == []


def test_delegate_and_source_are_both_recorded() -> None:
    source = Parameter("source", set_cmd=None, get_cmd=None, initial_value=0)
    delegate = DelegateParameter("delegate", source=source)

    with record_parameter_accesses() as accesses:
        delegate(1)

    assert [(a.parameter_name, a.kind) for a in accesses] == [
        ("source", "set"),
        ("delegate", "set"),
    ]


def test_recording_is_independent_of_on_set_callback() -> None:
    callback_values: list[Any] = []
    param = Parameter(
        "p",
        set_cmd=None,
        on_set_callback=lambda _param, value: callback_values.append(value),
    )

    with record_parameter_accesses() as accesses:
        param(1)

    assert callback_values == [1]
    assert [(a.kind, a.value) for a in accesses] == [("set", 1)]


def test_nothing_is_recorded_outside_the_block() -> None:
    param = Parameter("p", set_cmd=None, get_cmd=None)

    with record_parameter_accesses() as accesses:
        param(1)
    param(2)

    assert [a.value for a in accesses] == [1]


def test_records_into_a_given_list() -> None:
    param = Parameter("p", set_cmd=None)
    records: list[ParameterAccess] = []

    with record_parameter_accesses(records) as accesses:
        param(1)

    assert accesses is records
    assert [a.value for a in records] == [1]


@pytest.mark.parametrize("records", [(), set(), UserList(), type("L", (list,), {})()])
def test_only_a_plain_list_is_accepted(records: Any) -> None:
    with (
        pytest.raises(TypeError, match="only be recorded in a list"),
        record_parameter_accesses(records),
    ):
        pass


def test_nested_recordings_both_record() -> None:
    param = Parameter("p", set_cmd=None)

    with record_parameter_accesses() as outer:
        param(1)
        with record_parameter_accesses() as inner:
            param(2)
        param(3)

    assert [a.value for a in outer] == [1, 2, 3]
    assert [a.value for a in inner] == [2]


def test_other_threads_are_not_recorded() -> None:
    param = Parameter("p", set_cmd=None)

    with record_parameter_accesses() as accesses:
        worker = threading.Thread(target=param, args=(1,))
        worker.start()
        worker.join()
        param(2)

    assert [a.value for a in accesses] == [2]


def test_recording_follows_asyncio_tasks() -> None:
    param = Parameter("p", set_cmd=None)

    async def set_in_task(value: float) -> None:
        param(value)

    async def main() -> list[ParameterAccess]:
        with record_parameter_accesses() as recorded:
            # A task runs in a copy of the context it was created in.
            await asyncio.create_task(set_in_task(1))
        await asyncio.create_task(set_in_task(2))
        return recorded

    assert [a.value for a in asyncio.run(main())] == [1]


def test_copied_context_in_other_thread_is_not_recorded() -> None:
    param = Parameter("p", set_cmd=None)

    async def main() -> list[ParameterAccess]:
        with record_parameter_accesses() as recorded:
            # to_thread runs the function in a copy of the current context.
            await asyncio.to_thread(param, 1)
            context = contextvars.copy_context()
            worker = threading.Thread(target=context.run, args=(param, 2))
            worker.start()
            worker.join()
            param(3)
        return recorded

    assert [a.value for a in asyncio.run(main())] == [3]


def test_task_is_not_recorded_after_the_block() -> None:
    param = Parameter("p", set_cmd=None)

    async def main() -> list[ParameterAccess]:
        block_exited = asyncio.Event()

        async def set_values() -> None:
            param(1)
            await block_exited.wait()
            param(2)

        with record_parameter_accesses() as recorded:
            task = asyncio.create_task(set_values())
            await asyncio.sleep(0)
        block_exited.set()
        await task
        return recorded

    assert [a.value for a in asyncio.run(main())] == [1]


def test_logs_accesses(caplog: pytest.LogCaptureFixture) -> None:
    param = Parameter("logged_p", set_cmd=None, get_cmd=None)

    with caplog.at_level(logging.DEBUG, logger="qcodes.parameters.parameter_access"):
        with log_parameter_accesses():
            param(1)
            param()
        # Not logged: outside the block.
        param(2)

    records = [
        r for r in caplog.records if r.name == "qcodes.parameters.parameter_access"
    ]
    assert [r.getMessage() for r in records] == [
        "set logged_p = 1",
        "get logged_p = 1",
    ]
    access = records[0].parameter_access  # type: ignore[attr-defined]
    assert isinstance(access, ParameterAccess)
    assert records[0].parameter_name == "logged_p"  # type: ignore[attr-defined]


def test_logs_failed_accesses(caplog: pytest.LogCaptureFixture) -> None:
    def fail(value: float) -> None:
        raise RuntimeError("instrument refused")

    param = Parameter("logged_p", set_cmd=fail)

    with (
        caplog.at_level(logging.INFO, logger="qcodes.parameters.parameter_access"),
        log_parameter_accesses(logging.INFO),
        pytest.raises(RuntimeError),
    ):
        param(1)

    (record,) = [
        r for r in caplog.records if r.name == "qcodes.parameters.parameter_access"
    ]
    assert record.levelno == logging.INFO
    assert (
        record.getMessage() == "set logged_p failed: RuntimeError: instrument refused"
    )


def test_recording_and_logging_together(caplog: pytest.LogCaptureFixture) -> None:
    param = Parameter("p", set_cmd=None)

    with (
        caplog.at_level(logging.DEBUG, logger="qcodes.parameters.parameter_access"),
        record_parameter_accesses() as accesses,
        log_parameter_accesses(),
    ):
        param(1)

    assert [a.value for a in accesses] == [1]
    assert any(r.name == "qcodes.parameters.parameter_access" for r in caplog.records)


def test_accesses_made_while_logging_are_not_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    param = Parameter("p", set_cmd=None, get_cmd=None, initial_value=0)
    status = Parameter("status", set_cmd=None, get_cmd=None, initial_value=0)

    class StatusHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            status()

    handler = StatusHandler()
    access_log = logging.getLogger(ACCESS_LOGGER)
    access_log.addHandler(handler)
    try:
        with (
            caplog.at_level(logging.DEBUG, logger=ACCESS_LOGGER),
            record_parameter_accesses() as accesses,
            log_parameter_accesses(),
        ):
            param()
    finally:
        access_log.removeHandler(handler)

    # The handler reads status after p was appended, while p is logged.
    assert [a.parameter_name for a in accesses] == ["p", "status"]
    assert [r.getMessage() for r in caplog.records if r.name == ACCESS_LOGGER] == [
        "get p = 0"
    ]


@pytest.mark.usefixtures("failing_access_log")
def test_failing_log_does_not_affect_the_access(
    caplog: pytest.LogCaptureFixture,
) -> None:
    param = Parameter("p", set_cmd=None, get_cmd=None)

    with (
        caplog.at_level(logging.DEBUG, logger=ACCESS_LOGGER),
        log_parameter_accesses(),
        record_parameter_accesses() as accesses,
    ):
        param(1)
        assert param() == 1

    # Sinks after the failing one still receive every access.
    assert [(a.kind, a.value) for a in accesses] == [("set", 1), ("get", 1)]
    errors = [r for r in caplog.records if r.name == ERROR_LOGGER]
    assert [r.getMessage() for r in errors] == [
        "Recording the set of p failed",
        "Recording the get of p failed",
    ]
    assert errors[0].exc_info is not None
    assert isinstance(errors[0].exc_info[1], ValueError)


@pytest.mark.usefixtures("failing_access_log")
def test_failing_log_keeps_the_original_error() -> None:
    def fail(value: float) -> None:
        raise RuntimeError("instrument refused")

    param = Parameter("p", set_cmd=fail)

    with (
        log_parameter_accesses(),
        pytest.raises(RuntimeError, match="instrument refused"),
    ):
        param(1)


@pytest.mark.usefixtures("failing_access_log")
def test_failing_log_does_not_stop_a_ramp() -> None:
    param = Parameter("p", set_cmd=None, initial_value=0, vals=Numbers())
    param.step = 1

    with log_parameter_accesses(), record_parameter_accesses() as accesses:
        param(3)

    assert [a.value for a in accesses] == [1, 2, 3]
    assert param.cache.get() == 3


@pytest.mark.usefixtures("failing_access_log")
def test_failure_to_log_a_failure_does_not_affect_the_access(
    capsys: pytest.CaptureFixture[str],
) -> None:
    param = Parameter("p", set_cmd=None)
    error_log = logging.getLogger(ERROR_LOGGER)
    failing = _FailingFilter()
    error_log.addFilter(failing)
    try:
        with log_parameter_accesses():
            param(1)
    finally:
        error_log.removeFilter(failing)

    assert param.cache.get() == 1
    assert "broken filter" in capsys.readouterr().err
