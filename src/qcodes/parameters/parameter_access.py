"""
Recording of parameter access.

Every ``get`` and ``set`` of a parameter can be recorded as a
:class:`ParameterAccess`: what was accessed, the value, when and for how long,
and whether it failed. This makes it possible to see what a piece of code did
to the instruments it controls without changing the code or the drivers::

    >>> from qcodes.parameters import record_parameter_accesses
    >>> with record_parameter_accesses() as accesses:
    ...     dac.ch1(0.1)
    ...     dmm.v1()
    >>> [(access.kind, access.parameter_name) for access in accesses]
    [('set', 'dac_ch1'), ('get', 'dmm_v1')]

Recording is deliberately restricted to storing what happened:

* Accesses are appended to a plain :class:`list` or logged to a fixed logger
  (:func:`log_parameter_accesses`). Recording calls no user-supplied code, so
  it cannot react to an access, for example by setting another parameter.
  Logging an access runs the handlers and filters configured for logging, as
  any log call does. An access made by such a handler is recorded in lists but
  not logged, so a handler cannot log accesses recursively.
* Recording never changes the outcome of an access. If recording an access
  raises, for example because a logging handler or filter raised, the error
  is logged to the ``qcodes.parameters.parameter_access_errors`` logger and the
  access returns or raises as it would without recording.
* A :class:`ParameterAccess` holds data only: the names of the parameter and
  its instrument rather than the objects themselves, and a description of an
  error rather than the exception. Metadata such as the unit is not recorded,
  since a driver may read it from the instrument.
* Recording is local to the thread it was started in and ends when its block
  exits. Accesses made in other threads are never recorded, not even by code
  running in a copy of the recording context, such as a function run with
  :func:`asyncio.to_thread`. In the recording thread, the current
  :mod:`asyncio` task and the tasks it starts are recorded while the block
  runs. Recordings can be nested; each one receives every access made while
  it is active.
"""

from __future__ import annotations

import logging
import threading
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Generator

    from .parameter_base import ParameterBase

LOG = logging.getLogger(__name__)
"""The logger :func:`log_parameter_accesses` logs accesses to."""

# Not a child of LOG, so that a failing handler or filter of LOG does not
# also prevent the failure from being logged.
_ERROR_LOG = logging.getLogger(f"{__name__}_errors")

ParameterAccessKind = Literal["get", "set"]


@dataclass(frozen=True, slots=True)
class ParameterAccess:
    """
    A single ``get`` or ``set`` of a parameter.

    A set that ramps the parameter in steps (see :attr:`.ParameterBase.step`)
    is recorded once per step, since each step is a separate call to the
    instrument.
    """

    parameter_name: str
    """The full name of the parameter that was accessed."""
    instrument_name: str | None
    """The full name of the parameter's root instrument, if it has one."""
    kind: ParameterAccessKind
    """Whether the parameter was read (``"get"``) or written (``"set"``)."""
    value: Any
    """
    The value that was read, or the value of the step that was written. ``None``
    if a get failed before a value was obtained. The value is recorded as the
    parameter returned or received it, not copied.
    """
    raw_value: Any
    """
    The raw value returned by ``get_raw`` or passed to ``set_raw``. ``None`` if
    a get failed before the raw value was obtained.
    """
    timestamp: float
    """Time the access started, in seconds since the epoch (:func:`time.time`)."""
    duration: float
    """
    Seconds spent in ``get_raw`` or ``set_raw`` and the conversions around it.
    Delays configured on the parameter, such as ``post_delay``, are excluded.
    """
    error: str | None = None
    """
    The type and message of the exception the access raised, such as
    ``"TimeoutError: no reply"``, or ``None`` if it succeeded.
    """


class _Sink:
    """Where the accesses of one recording go."""

    __slots__ = ("active", "thread")

    def __init__(self) -> None:
        # The thread that started the recording.
        self.thread = threading.get_ident()
        # Contexts copied into tasks and threads share the sink, so it is
        # switched off rather than removed when the recording ends.
        self.active = True

    def record(self, access: ParameterAccess) -> None:
        raise NotImplementedError


class _ListSink(_Sink):
    """Appends accesses to a list."""

    __slots__ = ("_records",)

    def __init__(self, records: list[ParameterAccess]) -> None:
        super().__init__()
        self._records = records

    def record(self, access: ParameterAccess) -> None:
        self._records.append(access)


class _LogSink(_Sink):
    """Logs accesses to :data:`LOG`."""

    __slots__ = ("_level",)

    def __init__(self, level: int) -> None:
        super().__init__()
        self._level = level

    def record(self, access: ParameterAccess) -> None:
        if not LOG.isEnabledFor(self._level):
            return
        if access.error is None:
            message, args = (
                "%s %s = %r",
                (
                    access.kind,
                    access.parameter_name,
                    access.value,
                ),
            )
        else:
            message, args = (
                "%s %s failed: %s",
                (
                    access.kind,
                    access.parameter_name,
                    access.error,
                ),
            )
        LOG.log(
            self._level,
            message,
            *args,
            extra={
                "parameter_access": access,
                "parameter_name": access.parameter_name,
                "instrument_name": access.instrument_name,
            },
        )


_active_sinks: ContextVar[tuple[_Sink, ...]] = ContextVar(
    "qcodes_parameter_access_sinks", default=()
)

_recording: ContextVar[bool] = ContextVar(
    "qcodes_parameter_access_recording", default=False
)
"""Whether an access is being recorded, e.g. while logging handlers run."""


@contextmanager
def record_parameter_accesses(
    records: list[ParameterAccess] | None = None,
) -> Generator[list[ParameterAccess], None, None]:
    """
    Record every parameter access made in the current context in a list.

    Accesses made by the current thread while the block runs, by the current
    asyncio task and the tasks it starts, are appended to the list in the
    order they complete. An access that raises is recorded too, with its
    :attr:`ParameterAccess.error`.

    Args:
        records: The list to append to. A new list is used if ``None``.

    Yields:
        The list the accesses are appended to.

    Raises:
        TypeError: If ``records`` is not a :class:`list`. Subclasses are
            refused too, so that recording can only ever append.

    """
    if records is None:
        records = []
    if type(records) is not list:
        raise TypeError(
            "Parameter accesses can only be recorded in a list, "
            f"not in {type(records).__name__}."
        )
    with _sink_active(_ListSink(records)):
        yield records


@contextmanager
def log_parameter_accesses(
    level: int = logging.DEBUG,
) -> Generator[None, None, None]:
    """
    Log every parameter access made in the current context.

    Accesses made by the current thread while the block runs, by the current
    asyncio task and the tasks it starts, are logged to the
    ``qcodes.parameters.parameter_access`` logger. Each log record carries the
    :class:`ParameterAccess` as its ``parameter_access`` attribute, and the
    parameter and instrument names as its ``parameter_name`` and
    ``instrument_name`` attributes.

    Accesses that logging handlers or filters make while an access is logged
    are not logged. If a handler or filter raises, the error is logged to the
    ``qcodes.parameters.parameter_access_errors`` logger and the access is
    not affected.

    Args:
        level: The level to log accesses at.

    """
    with _sink_active(_LogSink(level)):
        yield


@contextmanager
def _sink_active(sink: _Sink) -> Generator[None, None, None]:
    token = _active_sinks.set((*_active_sinks.get(), sink))
    try:
        yield
    finally:
        sink.active = False
        _active_sinks.reset(token)


def _record(
    sinks: tuple[_Sink, ...],
    parameter: ParameterBase,
    kind: ParameterAccessKind,
    value: Any,
    raw_value: Any,
    timestamp: float,
    duration: float,
    exception: BaseException | None = None,
) -> None:
    thread = threading.get_ident()
    # An access made while another one is logged comes from a logging handler
    # or filter. Logging it could recurse without end.
    nested = _recording.get()
    eligible = [sink for sink in sinks if sink.active and sink.thread == thread]
    # Lists first: a logging handler that accesses a parameter must not come
    # between the appends of one access, or simultaneous recordings would see
    # accesses in different orders.
    sinks = (
        *(sink for sink in eligible if not isinstance(sink, _LogSink)),
        *(sink for sink in eligible if isinstance(sink, _LogSink) and not nested),
    )
    if not sinks:
        return
    token = _recording.set(True)
    try:
        try:
            access = _make_access(
                parameter, kind, value, raw_value, timestamp, duration, exception
            )
        except Exception:  # noqa: BLE001 recording must never affect the access
            _report_failure(kind, parameter.short_name)
            return
        for sink in sinks:
            try:
                sink.record(access)
            except Exception:  # noqa: BLE001 recording must never affect the access
                _report_failure(kind, access.parameter_name)
    finally:
        _recording.reset(token)


def _make_access(
    parameter: ParameterBase,
    kind: ParameterAccessKind,
    value: Any,
    raw_value: Any,
    timestamp: float,
    duration: float,
    exception: BaseException | None,
) -> ParameterAccess:
    # Only names, which QCoDeS derives without talking to the instrument.
    # Metadata such as the unit can be a property that a driver implements by
    # querying the instrument, so recording would add communication of its own.
    root = parameter.root_instrument
    return ParameterAccess(
        parameter_name=parameter.full_name,
        instrument_name=None if root is None else root.full_name,
        kind=kind,
        value=value,
        raw_value=raw_value,
        timestamp=timestamp,
        duration=duration,
        error=None if exception is None else _describe(exception),
    )


def _report_failure(kind: ParameterAccessKind, parameter_name: str) -> None:
    try:
        _ERROR_LOG.exception("Recording the %s of %s failed", kind, parameter_name)
    except Exception:  # noqa: BLE001 reported below; the access must not fail
        # Logging fails too, e.g. because a handler of the root logger
        # raises. Report it the way logging reports its own errors.
        if logging.raiseExceptions:
            traceback.print_exc()


def _describe(exception: BaseException) -> str:
    # The arguments only: the context QCoDeS appends to e.args (e.g.
    # "getting dac_ch1") is added after the access is recorded.
    try:
        message = str(exception)
    except Exception:  # noqa: BLE001 the access must not fail; as traceback does
        message = "<exception str() failed>"
    return f"{type(exception).__name__}: {message}"


__all__ = [
    "ParameterAccess",
    "ParameterAccessKind",
    "log_parameter_accesses",
    "record_parameter_accesses",
]
