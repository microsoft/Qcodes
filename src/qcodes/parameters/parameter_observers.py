"""
Observers of parameter access.

An observer is a callable that is notified of every ``get`` and ``set`` of
every parameter in the process, after the access has happened. This makes it
possible to record what a piece of code did to the instruments it controls
without changing the code or the drivers involved::

    >>> from qcodes.parameters import observe_parameters
    >>> accesses = []
    >>> with observe_parameters(accesses.append):
    ...     dac.ch1(0.1)
    ...     dmm.v1()

Unlike :attr:`.ParameterBase.on_set_callback` and
:attr:`.ParameterBase.global_on_set_callback`, any number of observers can be
registered, they see ``get`` as well as ``set``, and they are independent of
the callbacks of individual parameters.

Observers are called synchronously on the thread that accessed the parameter,
so they should be fast. An exception raised by an observer is logged and does
not affect the access, nor the other observers.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Generator

    from .parameter_base import ParameterBase

LOG = logging.getLogger(__name__)

ParameterAccessKind = Literal["get", "set"]


@dataclass(frozen=True, slots=True)
class ParameterAccess:
    """
    A single ``get`` or ``set`` of a parameter, as passed to an observer.

    A set that ramps the parameter in steps (see :attr:`.ParameterBase.step`)
    is reported once per step, since each step is a separate call to the
    instrument.
    """

    parameter: ParameterBase
    """The parameter that was accessed."""
    kind: ParameterAccessKind
    """Whether the parameter was read (``"get"``) or written (``"set"``)."""
    value: Any
    """
    The value that was read, or the value of the step that was written. ``None``
    if a get failed before a value was obtained.
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
    exception: BaseException | None = None
    """The exception raised by the access, or ``None`` if it succeeded."""


type ParameterObserver = Callable[[ParameterAccess], None]
"""A callable that is notified of every parameter access."""

_lock = threading.Lock()
# Replaced rather than mutated, so a reader never needs the lock: it sees the
# tuple that was registered when it started.
_observers: tuple[ParameterObserver, ...] = ()


def add_parameter_observer(observer: ParameterObserver) -> None:
    """
    Register an observer of every parameter ``get`` and ``set``.

    Args:
        observer: Called with a :class:`ParameterAccess` after each access.
            Registering the same observer twice notifies it twice.

    """
    global _observers
    with _lock:
        _observers = (*_observers, observer)


def remove_parameter_observer(observer: ParameterObserver) -> None:
    """
    Unregister an observer added with :func:`add_parameter_observer`.

    If the observer was registered more than once, one registration is removed.

    Args:
        observer: The observer to remove.

    Raises:
        ValueError: If the observer is not registered.

    """
    global _observers
    with _lock:
        observers = list(_observers)
        observers.remove(observer)
        _observers = tuple(observers)


@contextmanager
def observe_parameters(observer: ParameterObserver) -> Generator[None, None, None]:
    """
    Context manager that registers ``observer`` for the duration of the block.

    Args:
        observer: Called with a :class:`ParameterAccess` after each access.

    """
    add_parameter_observer(observer)
    try:
        yield
    finally:
        remove_parameter_observer(observer)


def registered_parameter_observers() -> tuple[ParameterObserver, ...]:
    """
    Return the currently registered observers, in registration order.
    """
    return _observers


def _notify(observers: tuple[ParameterObserver, ...], access: ParameterAccess) -> None:
    for observer in observers:
        try:
            observer(access)
        except Exception:
            LOG.warning(
                "Exception in parameter observer %r for %s of %s",
                observer,
                access.kind,
                access.parameter.full_name,
                exc_info=True,
            )


__all__ = [
    "ParameterAccess",
    "ParameterAccessKind",
    "ParameterObserver",
    "add_parameter_observer",
    "observe_parameters",
    "registered_parameter_observers",
    "remove_parameter_observer",
]
