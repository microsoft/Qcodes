"""Plan message vocabulary.

Plans are Python generators that yield instances of these frozen dataclasses.
A measurement engine iterates the generator, dispatching each message to
instruments and returning results to the plan via :py:meth:`generator.send`.

The vocabulary is deliberately small. The tracer-bullet scope is six message
types; ``Call`` and ``Describe`` are deferred to v1.

See the design document at
``files/async-measurement-architecture.md`` (in the design session
workspace) for the full rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from qcodes.dataset.descriptions.versioning.rundescribertypes import Shapes
    from qcodes.dataset.experiment_container import Experiment
    from qcodes.parameters import ParameterBase


@dataclass(frozen=True)
class Set:
    """Set a single parameter to a value.

    The engine calls ``param.set(value)`` and records the value in its
    per-run state cache under ``param.register_name``.
    """

    param: ParameterBase
    value: Any


@dataclass(frozen=True)
class Read:
    """Read one or more parameters.

    The engine calls ``param.get()`` for each parameter (potentially in
    parallel, grouped by ``underlying_instrument``) and returns a
    ``dict[ParameterBase, Any]`` to the plan via ``generator.send``.

    Reading a parameter that is not declared in the run descriptor's
    ``measured`` tuple is an error.
    """

    params: tuple[ParameterBase, ...]


@dataclass(frozen=True)
class Sleep:
    """Sleep for a duration.

    The engine implements this as a cancellable sleep (chunked checks of the
    cancel flag every ~100 ms) so a long sleep does not delay cancellation.
    """

    seconds: float


@dataclass(frozen=True)
class Emit:
    """Emit one dataset row for the current run.

    The row is built from the engine's per-run state cache (last value seen
    for each declared parameter via :py:class:`Set` or :py:class:`Read`),
    overlaid with any ``overrides``. The completed row is forwarded to all
    sinks as a :py:class:`~qcodes.measure_v2.events.RowEmitted` event.

    ``overrides`` may only reference parameters that are already declared in
    the run descriptor; providing an undeclared parameter raises before the
    row is published. There is no lazy schema registration.
    """

    overrides: Mapping[ParameterBase, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenRun:
    """Open a run.

    Emitted by the ``run(...)`` decorator at the start of plan execution.
    The descriptor declares the run's schema (setpoint and measured
    parameters, plus optional shapes). The sink uses it to register the
    dataset; the engine uses it to validate subsequent ``Set``/``Read``/
    ``Emit`` messages.
    """

    name: str
    setpoint_params: tuple[ParameterBase, ...]
    measured_params: tuple[ParameterBase, ...]
    exp: Experiment | None = None
    shapes: Shapes | None = None
    write_period: float | None = None


@dataclass(frozen=True)
class CloseRun:
    """Close the current run.

    Emitted by the ``run(...)`` decorator at the end of plan execution
    (success, error, or cancel). The sink finalizes the dataset.
    """


Msg = Set | Read | Sleep | Emit | OpenRun | CloseRun
"""Union of all plan message types. A plan is a ``Generator[Msg, Any, None]``."""
