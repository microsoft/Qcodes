"""User-facing convenience surface for ``measure_v2``.

The 95% case: ``qc.measure_v2.scan(LinSweep(g, 0, 1, 11), measure=[i])``
returns a :py:class:`~qcodes.dataset.data_set_protocol.DataSetProtocol`.

In tracer scope, only single-sweep (1D) scans are supported. Multi-sweep
scans will be added once ``scan_inner_outer`` is implemented.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from qcodes.measure_v2.decorators import run
from qcodes.measure_v2.engine import MeasurementEngine, RunHandle
from qcodes.measure_v2.plans import scan_1d
from qcodes.measure_v2.sinks import SqliteSink

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qcodes.dataset.data_set_protocol import DataSetProtocol
    from qcodes.dataset.dond.sweeps import AbstractSweep
    from qcodes.dataset.experiment_container import Experiment
    from qcodes.parameters import ParameterBase


_default_engine: MeasurementEngine | None = None
_default_engine_lock = threading.Lock()


def default_engine() -> MeasurementEngine:
    """Return the process-wide default :py:class:`MeasurementEngine`.

    Lazily instantiated on first use with a default
    :py:class:`SqliteSink` that writes to the database configured via
    ``qc.config["core"]["db_location"]``. Subsequent calls return the
    same instance.

    For tests, prefer constructing an explicit engine rather than relying
    on the default — the default's sink references the current
    ``db_location`` at construction time, which won't pick up later
    config changes.
    """
    global _default_engine
    with _default_engine_lock:
        if _default_engine is None:
            _default_engine = MeasurementEngine(sinks=[SqliteSink()])
        return _default_engine


def reset_default_engine() -> None:
    """Shut down and drop the cached default engine.

    Primarily for tests that need to recreate the engine after changing
    ``db_location`` or sink configuration.
    """
    global _default_engine
    with _default_engine_lock:
        if _default_engine is not None:
            _default_engine.shutdown(wait=True, timeout=5.0)
            _default_engine = None


def scan(
    *sweeps: AbstractSweep,
    measure: Sequence[ParameterBase],
    wait: bool = True,
    name: str = "",
    exp: Experiment | None = None,
    engine: MeasurementEngine | None = None,
) -> DataSetProtocol | RunHandle | None:
    """Run a scan.

    Tracer scope: one sweep only. The sweep parameter is set across its
    setpoints; ``measure`` parameters are read at each point; a row is
    emitted per point. On exit (success, error, or cancel), the swept
    parameter is set back to 0.0 (the ``scan_1d`` cleanup contract).

    Args:
        *sweeps: Sweeps to perform. Currently exactly one sweep is required.
        measure: Parameters to read at each setpoint.
        wait: If ``True`` (default), block until the run completes and
            return the resulting dataset. If ``False``, return the
            :py:class:`RunHandle` immediately for non-blocking workflows.
        name: Dataset name.
        exp: Experiment to attach the dataset to. If ``None``, the sink
            creates a default experiment.
        engine: Engine to submit on. Defaults to :py:func:`default_engine`.

    Returns:
        - If ``wait=True``: the resulting dataset (or ``None`` if no sink
          provided one).
        - If ``wait=False``: a :py:class:`RunHandle` for the running submission.

    """
    if len(sweeps) != 1:
        raise NotImplementedError(
            "measure_v2.scan currently supports exactly one sweep "
            f"(got {len(sweeps)}). Multi-dimensional scans are planned for v1."
        )

    sweep = sweeps[0]
    eng = engine if engine is not None else default_engine()

    setpoints = (sweep.param,)
    measured = tuple(measure)

    plan = run(
        name=name,
        exp=exp,
        setpoints=setpoints,
        measured=measured,
    )(scan_1d(sweep, measured))

    handle = eng.submit(plan)
    if not wait:
        return handle
    handle.wait()
    return handle.dataset.result()
