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
  (:func:`log_parameter_accesses`). No user-supplied code runs while a
  parameter is accessed, so recording cannot react to an access, for example
  by setting another parameter.
* A :class:`ParameterAccess` holds data only: the names of the parameter and
  its instrument rather than the objects themselves, and a description of an
  error rather than the exception.
* Recording is local to the context it was started in: the current thread, or
  the current :mod:`asyncio` task. Accesses made in other threads are not
  recorded. Recordings can be nested; each one receives every access made
  while it is active.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Generator

    from .parameter_base import ParameterBase

LOG = logging.getLogger(__name__)
"""The logger :func:`log_parameter_accesses` logs accesses to."""

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
    unit: str
    """The unit of the parameter."""
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


class _ListSink:
    """Appends accesses to a list."""

    __slots__ = ("_records",)

    def __init__(self, records: list[ParameterAccess]) -> None:
        self._records = records

    def record(self, access: ParameterAccess) -> None:
        self._records.append(access)


class _LogSink:
    """Logs accesses to :data:`LOG`."""

    __slots__ = ("_level",)

    def __init__(self, level: int) -> None:
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


_Sink = _ListSink | _LogSink

_active_sinks: ContextVar[tuple[_Sink, ...]] = ContextVar(
    "qcodes_parameter_access_sinks", default=()
)


@contextmanager
def record_parameter_accesses(
    records: list[ParameterAccess] | None = None,
) -> Generator[list[ParameterAccess], None, None]:
    """
    Record every parameter access made in the current context in a list.

    Accesses made by the current thread (or asyncio task) while the block runs
    are appended to the list, in the order they complete. An access that
    raises is recorded too, with its :attr:`ParameterAccess.error`.

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

    Accesses made by the current thread (or asyncio task) while the block runs
    are logged to the ``qcodes.parameters.parameter_access`` logger. Each log
    record carries the :class:`ParameterAccess` as its ``parameter_access``
    attribute, and the parameter and instrument names as its
    ``parameter_name`` and ``instrument_name`` attributes.

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
    root = parameter.root_instrument
    access = ParameterAccess(
        parameter_name=parameter.full_name,
        instrument_name=None if root is None else root.full_name,
        kind=kind,
        value=value,
        raw_value=raw_value,
        unit=getattr(parameter, "unit", ""),
        timestamp=timestamp,
        duration=duration,
        error=None if exception is None else _describe(exception),
    )
    for sink in sinks:
        sink.record(access)


def _describe(exception: BaseException) -> str:
    # The arguments only: the context QCoDeS appends to e.args (e.g.
    # "getting dac_ch1") is added after the access is recorded.
    return f"{type(exception).__name__}: {exception}"


__all__ = [
    "ParameterAccess",
    "ParameterAccessKind",
    "log_parameter_accesses",
    "record_parameter_accesses",
]
