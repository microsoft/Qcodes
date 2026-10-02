"""Acceptance tests — the three programs from the tracer plan.

These are the success criterion for the tracer bullet. If all three pass,
the architecture has cleared the proof bar end-to-end:

1. Blocking ``qc.measure_v2.scan`` returns a populated dataset.
2. Non-blocking ``qc.measure_v2.scan(..., wait=False)`` + ``cancel()`` runs
   ``scan_1d``'s ``finally`` ramp-to-zero. Partial dataset is preserved.
3. Engine-level submission with ``MemorySink`` produces the expected events.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

import qcodes as qc
from qcodes import measure_v2 as mv2
from qcodes.dataset.dond.sweeps import LinSweep
from qcodes.measure_v2 import (
    MeasurementEngine,
    MemorySink,
    RowEmitted,
    SqliteSink,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from qcodes.parameters import ParameterBase


@pytest.fixture
def fresh_engine(empty_db) -> Generator[MeasurementEngine, None, None]:
    """Acceptance engine: SqliteSink + isolated db, separate from default."""
    del empty_db
    sink = SqliteSink(experiment_name="acceptance", sample_name="tracer")
    eng = MeasurementEngine(sinks=[sink])
    try:
        yield eng
    finally:
        eng.shutdown(wait=True, timeout=5.0)


def _make_params() -> tuple[ParameterBase, ParameterBase]:
    g = qc.Parameter("g", initial_value=0.0, set_cmd=None, get_cmd=None)
    i = qc.Parameter("i", get_cmd=lambda: g.cache.get() ** 2)
    return g, i


def test_acceptance_blocking_scan_returns_populated_dataset(
    fresh_engine: MeasurementEngine,
) -> None:
    """Acceptance #1: ``scan(..., wait=True)`` returns a real dataset."""
    g, i = _make_params()

    ds = mv2.scan(
        LinSweep(g, 0.0, 1.0, 11),
        measure=[i],
        name="acceptance-1",
        engine=fresh_engine,
    )

    assert ds is not None
    # Verify shape via the cache (publisher-thread connection → cache is the
    # canonical way to read on the main thread).
    data = ds.cache.data()
    assert "i" in data
    assert len(data["i"]["g"]) == 11
    assert len(data["i"]["i"]) == 11


def test_acceptance_nonblocking_cancel_preserves_partial_data(
    fresh_engine: MeasurementEngine,
) -> None:
    """Acceptance #2: cancel mid-scan; finally ramps g to 0; partial data persisted."""
    g, i = _make_params()

    handle = mv2.scan(
        LinSweep(g, 0.0, 1.0, 1001, delay=0.01),
        measure=[i],
        wait=False,
        name="acceptance-2",
        engine=fresh_engine,
    )
    assert isinstance(handle, mv2.RunHandle)

    time.sleep(0.1)  # let some rows accumulate
    handle.cancel()
    result = handle.wait(timeout=5.0)

    assert result.reason == "cancelled"
    # Cleanup contract: scan_1d's finally ramped g back to 0
    assert g.cache.get() == pytest.approx(0.0)

    # Partial dataset is preserved and committed
    ds = handle.dataset.result(timeout=1.0)
    assert ds is not None
    assert ds.completed
    assert len(ds.cache.data()["i"]["g"]) >= 1


def test_acceptance_engine_with_memorysink() -> None:
    """Acceptance #3: engine-level submission with MemorySink yields events."""
    g, i = _make_params()
    sink = MemorySink()
    eng = MeasurementEngine(sinks=[sink])
    try:
        plan = mv2.run(name="acceptance-3", setpoints=(g,), measured=(i,))(
            mv2.scan_1d(LinSweep(g, 0.0, 1.0, 11), [i])
        )
        result = eng.submit(plan).wait(timeout=10.0)
    finally:
        eng.shutdown(wait=True, timeout=5.0)

    assert result.reason == "completed"
    rows = [e for e in sink.events if isinstance(e, RowEmitted)]
    assert len(rows) == 11
    # Each row's snapshot has both g and i
    assert all(g in r.snapshot and i in r.snapshot for r in rows)
    # i = g**2 — first row is (0,0), last is (1,1)
    assert rows[0].snapshot[g] == pytest.approx(0.0)
    assert rows[-1].snapshot[g] == pytest.approx(1.0)
    assert rows[-1].snapshot[i] == pytest.approx(1.0)
