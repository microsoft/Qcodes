"""Sink protocol and the criticality helper.

A sink is any callable that accepts a single :py:class:`Event`. There is
no required base class. Sinks MAY declare a ``critical`` attribute
(class- or instance-level) to opt into "abort run on failure" semantics;
sinks without the attribute default to non-critical (failures are logged
but the run continues).

The engine inspects ``critical`` via :py:func:`is_critical`, which uses
``getattr`` with a default — so plain functions work as sinks without any
boilerplate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from qcodes.measure_v2.events import Event


@runtime_checkable
class DataSink(Protocol):
    """A callable that consumes events.

    Implementations may be plain functions or classes implementing
    ``__call__``. Class-based sinks may declare ``critical: bool`` to
    opt into "abort run on failure" semantics.
    """

    def __call__(self, event: Event, /) -> None: ...


def is_critical(sink: DataSink) -> bool:
    """Return whether a sink declares itself as critical.

    Returns ``getattr(sink, 'critical', False)``. Critical sinks abort the
    run if they raise on ``RunStarted``; their failures during the run
    are propagated to ``RunResult.error`` while still allowing other
    sinks to finish.
    """
    return bool(getattr(sink, "critical", False))
