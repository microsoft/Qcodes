"""Event vocabulary published by the engine to sinks.

Sinks receive instances of these dataclasses via their ``__call__``. The
event stream for a single run is always:

    RunStarted -> RowEmitted* -> RunStopped

with exactly one ``RunStarted`` and exactly one ``RunStopped`` per
``run_id``. The publisher thread guarantees ordering.

See ``files/async-measurement-architecture.md`` §6 for the full sink and
event protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping
    from uuid import UUID

    from qcodes.dataset.descriptions.versioning.rundescribertypes import Shapes
    from qcodes.dataset.experiment_container import Experiment
    from qcodes.parameters import ParameterBase


RunStopReason = Literal[
    "completed",
    "cancelled",
    "interrupted",
    "error",
    "engine_shutdown",
    "cancelled_before_start",
]


@dataclass(frozen=True)
class Descriptor:
    """Schema declaration for a run.

    Built by the ``run(...)`` decorator from either explicit kwargs or a
    ``Describe`` first message in the plan. Attached to ``RunStarted`` so
    sinks can register the dataset before any rows arrive.
    """

    setpoints: tuple[ParameterBase, ...]
    measured: tuple[ParameterBase, ...]
    shapes: Shapes | None = None


@dataclass(frozen=True)
class RunStarted:
    """A run has been opened. Sinks should set up dataset state here."""

    run_id: UUID
    name: str
    descriptor: Descriptor
    exp: Experiment | None
    write_period: float | None
    started_at: float


@dataclass(frozen=True)
class RowEmitted:
    """A single row of measurement data, snapshotted from the engine state.

    ``snapshot`` is keyed by :py:class:`~qcodes.parameters.ParameterBase`
    objects from the descriptor. Each value is whatever ``param.get()``
    returned for measured params, or the last ``Set`` value for setpoints.
    Array-valued measurements are stored as ndarrays — the SQLite sink
    fans them out into multiple dataset rows via existing
    ``DataSaver.add_result`` logic.
    """

    run_id: UUID
    snapshot: Mapping[ParameterBase, Any]
    seq: int


@dataclass(frozen=True)
class RunStopped:
    """The run is over. Sinks should finalize/close dataset state."""

    run_id: UUID
    reason: RunStopReason
    error: BaseException | None
    started_at: float
    stopped_at: float
    # Tracer scope: cancel_latency and n_rows_emitted are present in the
    # event but may be left None / 0 by the v0 engine.
    cancel_latency: float | None = None
    n_rows_emitted: int = 0


Event = RunStarted | RowEmitted | RunStopped
"""Union of all event types delivered to sinks."""


@dataclass(frozen=True)
class RunResult:
    """Final outcome of a run, returned to the user via ``RunHandle``.

    A condensed view of the run's lifecycle, suitable for assertions in
    tests and for synchronous return values from ``qc.measure_v2.scan``.
    """

    run_id: UUID
    reason: RunStopReason
    error: BaseException | None
    started_at: float
    stopped_at: float
    cancel_latency: float | None = None
    n_rows_emitted: int = 0
