"""L2 / L3 tests for ``scan_nd`` through the engine + sinks."""

from __future__ import annotations

import time

import pytest

from qcodes.dataset.dond.sweeps import LinSweep
from qcodes.measure_v2 import (
    MeasurementEngine,
    MemorySink,
    RowEmitted,
    RunStarted,
    RunStopped,
    SqliteSink,
    run,
    scan,
    scan_nd,
)
from qcodes.parameters import Parameter, ParameterBase


def _make_params() -> tuple[ParameterBase, ParameterBase, ParameterBase]:
    g = Parameter("g", initial_value=0.0, set_cmd=None, get_cmd=None)
    h = Parameter("h", initial_value=0.0, set_cmd=None, get_cmd=None)
    # i depends on both g and h
    i = Parameter("i", get_cmd=lambda: g.cache.get() ** 2 + h.cache.get())
    return g, h, i


# ----------------------------------------------------------------------------
# L2 — MemorySink
# ----------------------------------------------------------------------------


@pytest.fixture
def memory_engine() -> MeasurementEngine:
    sink = MemorySink()
    eng = MeasurementEngine(sinks=[sink])
    eng._test_sink = sink  # type: ignore[attr-defined]
    try:
        yield eng
    finally:
        eng.shutdown(wait=True, timeout=5.0)


def test_scan_nd_2d_emits_n_times_m_rows(memory_engine: MeasurementEngine) -> None:
    g, h, i = _make_params()
    outer = LinSweep(g, 0.0, 1.0, 3)
    inner = LinSweep(h, 0.0, 1.0, 4)
    plan = run(setpoints=(g, h), measured=(i,))(scan_nd(outer, inner, measured=[i]))

    handle = memory_engine.submit(plan)
    result = handle.wait(timeout=10.0)

    assert result.reason == "completed"
    assert result.n_rows_emitted == 3 * 4
    sink: MemorySink = memory_engine._test_sink  # type: ignore[attr-defined]
    rows = [e for e in sink.events if isinstance(e, RowEmitted)]
    assert len(rows) == 3 * 4


def test_scan_nd_2d_snapshot_contains_both_setpoints(
    memory_engine: MeasurementEngine,
) -> None:
    g, h, i = _make_params()
    outer = LinSweep(g, 0.0, 1.0, 3)
    inner = LinSweep(h, 0.0, 2.0, 3)
    plan = run(setpoints=(g, h), measured=(i,))(scan_nd(outer, inner, measured=[i]))

    memory_engine.submit(plan).wait(timeout=10.0)

    sink: MemorySink = memory_engine._test_sink  # type: ignore[attr-defined]
    rows = [e for e in sink.events if isinstance(e, RowEmitted)]
    # Every row's snapshot includes g, h, and i
    for row in rows:
        assert g in row.snapshot
        assert h in row.snapshot
        assert i in row.snapshot
    # First row: g=0, h=0, i=0
    # Last row: g=1, h=2, i=1+2=3
    assert rows[0].snapshot[g] == pytest.approx(0.0)
    assert rows[0].snapshot[h] == pytest.approx(0.0)
    assert rows[-1].snapshot[g] == pytest.approx(1.0)
    assert rows[-1].snapshot[h] == pytest.approx(2.0)
    assert rows[-1].snapshot[i] == pytest.approx(1.0 + 2.0)


def test_scan_nd_3d_emits_product_of_lengths(memory_engine: MeasurementEngine) -> None:
    g, h, _ = _make_params()
    k = Parameter("k", initial_value=0.0, set_cmd=None, get_cmd=None)
    i = Parameter("i", get_cmd=lambda: g.cache.get() + h.cache.get() + k.cache.get())
    s1 = LinSweep(g, 0.0, 1.0, 2)
    s2 = LinSweep(h, 0.0, 1.0, 3)
    s3 = LinSweep(k, 0.0, 1.0, 4)
    plan = run(setpoints=(g, h, k), measured=(i,))(scan_nd(s1, s2, s3, measured=[i]))

    memory_engine.submit(plan).wait(timeout=10.0)

    sink: MemorySink = memory_engine._test_sink  # type: ignore[attr-defined]
    assert sum(1 for e in sink.events if isinstance(e, RowEmitted)) == 2 * 3 * 4


def test_scan_nd_cancel_ramps_all_sweep_params_to_zero(
    memory_engine: MeasurementEngine,
) -> None:
    g, h, i = _make_params()
    outer = LinSweep(g, 0.0, 1.0, 100, delay=0.005)
    inner = LinSweep(h, 0.0, 1.0, 100, delay=0.005)
    plan = run(setpoints=(g, h), measured=(i,))(scan_nd(outer, inner, measured=[i]))

    handle = memory_engine.submit(plan)
    time.sleep(0.1)
    handle.cancel()
    result = handle.wait(timeout=5.0)

    assert result.reason == "cancelled"
    # Both swept params must be back at 0.0
    assert g.cache.get() == pytest.approx(0.0)
    assert h.cache.get() == pytest.approx(0.0)


def test_scan_nd_get_after_set_uses_readback_in_snapshot(
    memory_engine: MeasurementEngine,
) -> None:
    """A param with set_parser/non-identity readback shows the readback in rows."""
    # Build a parameter whose get() returns its cache value rounded — so
    # setting 0.37 with get_after_set=True stores 0.37 (since cache=0.37 → round=0).
    # Actually let's keep it simple: set_cmd doubles the value internally.
    state = {"v": 0.0}

    def _set(val):
        state["v"] = val * 2.0  # the "instrument" applies 2x

    def _get():
        return state["v"]

    g = Parameter("g", set_cmd=_set, get_cmd=_get)
    i = Parameter("i", get_cmd=lambda: 0.0)
    sweep = LinSweep(g, 1.0, 3.0, 3, get_after_set=True)
    plan = run(setpoints=(g,), measured=(i,))(scan_nd(sweep, measured=[i]))

    memory_engine.submit(plan).wait(timeout=5.0)

    sink: MemorySink = memory_engine._test_sink  # type: ignore[attr-defined]
    rows = [e for e in sink.events if isinstance(e, RowEmitted)]
    # The dataset rows reflect what the "instrument" actually has: 2*set_val.
    g_in_rows = [r.snapshot[g] for r in rows]
    assert g_in_rows == pytest.approx([2.0, 4.0, 6.0])


def test_scan_nd_event_ordering(memory_engine: MeasurementEngine) -> None:
    g, h, i = _make_params()
    outer = LinSweep(g, 0.0, 1.0, 2)
    inner = LinSweep(h, 0.0, 1.0, 3)
    plan = run(setpoints=(g, h), measured=(i,))(scan_nd(outer, inner, measured=[i]))

    memory_engine.submit(plan).wait(timeout=5.0)

    sink: MemorySink = memory_engine._test_sink  # type: ignore[attr-defined]
    assert isinstance(sink.events[0], RunStarted)
    assert isinstance(sink.events[-1], RunStopped)
    middle = sink.events[1:-1]
    assert all(isinstance(e, RowEmitted) for e in middle)


# ----------------------------------------------------------------------------
# L3 — SqliteSink (persistence)
# ----------------------------------------------------------------------------


@pytest.fixture
def sqlite_engine(empty_db) -> MeasurementEngine:
    del empty_db
    sink = SqliteSink(experiment_name="measure_v2_2d_test", sample_name="tracer")
    eng = MeasurementEngine(sinks=[sink])
    try:
        yield eng
    finally:
        eng.shutdown(wait=True, timeout=5.0)


def test_scan_nd_2d_persists_to_sqlite(sqlite_engine: MeasurementEngine) -> None:
    g, h, i = _make_params()
    outer = LinSweep(g, 0.0, 1.0, 3)
    inner = LinSweep(h, 0.0, 1.0, 4)
    plan = run(setpoints=(g, h), measured=(i,))(scan_nd(outer, inner, measured=[i]))

    handle = sqlite_engine.submit(plan)
    handle.wait(timeout=10.0)

    ds = handle.dataset.result(timeout=1.0)
    assert ds is not None
    assert ds.completed
    cache = ds.cache.data()
    assert len(cache["i"]["g"]) == 3 * 4
    assert len(cache["i"]["h"]) == 3 * 4
    assert len(cache["i"]["i"]) == 3 * 4


def test_convenience_scan_dispatches_to_scan_nd(
    sqlite_engine: MeasurementEngine,
) -> None:
    """``qc.measure_v2.scan(sweep1, sweep2, measure=[...])`` runs scan_nd."""
    g, h, i = _make_params()

    ds = scan(
        LinSweep(g, 0.0, 1.0, 3),
        LinSweep(h, 0.0, 1.0, 4),
        measure=[i],
        name="conv-2d",
        engine=sqlite_engine,
    )
    assert ds is not None
    cache = ds.cache.data()
    assert len(cache["i"]["g"]) == 3 * 4


def test_convenience_scan_with_zero_sweeps_raises() -> None:
    """No-sweep scan is a usage error."""
    g, h, i = _make_params()
    del g, h
    with pytest.raises(ValueError, match="at least one sweep"):
        scan(measure=[i])
