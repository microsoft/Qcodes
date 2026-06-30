"""Experimental parallel measurement API for QCoDeS.

See ``files/async-measurement-architecture.md`` in the design session for the
full architecture proposal. This package is **experimental and unstable**;
the public surface may change between releases.

Tracer-bullet scope (current): plan message vocabulary, ``run(...)``
decorator, ``scan_1d`` plan-builder, and unit-test helpers. Engine,
sinks, and convenience layer are not yet implemented.
"""

from qcodes.measure_v2.convenience import (
    default_engine,
    reset_default_engine,
    scan,
)
from qcodes.measure_v2.decorators import run
from qcodes.measure_v2.engine import MeasurementEngine, RunHandle, RunStatus
from qcodes.measure_v2.events import (
    Descriptor,
    Event,
    RowEmitted,
    RunResult,
    RunStarted,
    RunStopped,
)
from qcodes.measure_v2.exceptions import CancelRequested, PlanError
from qcodes.measure_v2.messages import (
    CloseRun,
    Emit,
    Msg,
    OpenRun,
    Read,
    Set,
    Sleep,
)
from qcodes.measure_v2.plans import scan_1d
from qcodes.measure_v2.sinks import DataSink, MemorySink, SqliteSink, is_critical

__all__ = [
    "CancelRequested",
    "CloseRun",
    "DataSink",
    "Descriptor",
    "Emit",
    "Event",
    "MeasurementEngine",
    "MemorySink",
    "Msg",
    "OpenRun",
    "PlanError",
    "Read",
    "RowEmitted",
    "RunHandle",
    "RunResult",
    "RunStarted",
    "RunStatus",
    "RunStopped",
    "Set",
    "Sleep",
    "SqliteSink",
    "default_engine",
    "is_critical",
    "reset_default_engine",
    "run",
    "scan",
    "scan_1d",
]
