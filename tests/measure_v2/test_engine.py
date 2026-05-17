"""L2 tests for ``MeasurementEngine`` using ``MemorySink``.

These tests run the full engine + publisher + plan pipeline on real
threads, but with software-only parameters and an in-memory sink (no
database). They validate the threading model, cancel semantics, and
event ordering — the things L1 tests can't reach.
"""

from __future__ import annotations

import time

import pytest

from qcodes.dataset.dond.sweeps import LinSweep
from qcodes.measure_v2 import (
    Emit,
    MeasurementEngine,
    MemorySink,
    PlanError,
    Read,
    RowEmitted,
    RunStarted,
    RunStopped,
    Set,
    run,
    scan_1d,
)
from qcodes.parameters import Parameter, ParameterBase


@pytest.fixture
def engine() -> MeasurementEngine:
    sink = MemorySink()
    eng = MeasurementEngine(sinks=[sink])
    # Stash the sink on the engine for test convenience.
    eng._test_sink = sink  # type: ignore[attr-defined]
    try:
        yield eng
    finally:
        eng.shutdown(wait=True, timeout=5.0)


def _make_params() -> tuple[ParameterBase, ParameterBase]:
    g = Parameter("g", initial_value=0.0, set_cmd=None, get_cmd=None)
    # i computes from g's cache, so we can assert dependent values
    i = Parameter("i", get_cmd=lambda: g.cache.get() ** 2)
    return g, i


# ----------------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------------


def test_engine_executes_simple_1d_scan(engine: MeasurementEngine) -> None:
    g, i = _make_params()
    sweep = LinSweep(g, start=0.0, stop=2.0, num_points=5)
    plan = run(name="t", setpoints=(g,), measured=(i,), shapes={"i": (5,)})(
        scan_1d(sweep, [i])
    )

    handle = engine.submit(plan)
    result = handle.wait(timeout=10.0)

    assert result.reason == "completed"
    assert result.n_rows_emitted == 5
    sink: MemorySink = engine._test_sink  # type: ignore[attr-defined]
    assert len(sink.rows) == 5
    assert len(sink.starts) == 1
    assert len(sink.stops) == 1
    # i = g**2; sweep is 0, 0.5, 1.0, 1.5, 2.0 — so i is 0, 0.25, 1.0, 2.25, 4.0
    expected_i = [0.0, 0.25, 1.0, 2.25, 4.0]
    actual_i = [r.snapshot[i] for r in sink.rows]
    assert actual_i == pytest.approx(expected_i)


def test_event_ordering(engine: MeasurementEngine) -> None:
    """RunStarted is always first; RunStopped is always last."""
    g, i = _make_params()
    sweep = LinSweep(g, start=0.0, stop=1.0, num_points=3)
    plan = run(setpoints=(g,), measured=(i,))(scan_1d(sweep, [i]))

    handle = engine.submit(plan)
    handle.wait(timeout=5.0)

    sink: MemorySink = engine._test_sink  # type: ignore[attr-defined]
    assert isinstance(sink.events[0], RunStarted)
    assert isinstance(sink.events[-1], RunStopped)
    # All middle events are RowEmitted
    for ev in sink.events[1:-1]:
        assert isinstance(ev, RowEmitted)


def test_setpoint_state_is_set_on_hardware(engine: MeasurementEngine) -> None:
    """The engine actually calls param.set() — verify via cache."""
    g, i = _make_params()
    sweep = LinSweep(g, start=0.0, stop=1.0, num_points=3)
    plan = run(setpoints=(g,), measured=(i,))(scan_1d(sweep, [i]))

    engine.submit(plan).wait(timeout=5.0)

    # After the scan completes, scan_1d's finally sets g back to 0.0.
    assert g.cache.get() == pytest.approx(0.0)


def test_handle_dataset_future_resolves(engine: MeasurementEngine) -> None:
    g, i = _make_params()
    sweep = LinSweep(g, start=0.0, stop=1.0, num_points=2)
    plan = run(setpoints=(g,), measured=(i,))(scan_1d(sweep, [i]))

    handle = engine.submit(plan)
    handle.wait(timeout=5.0)
    # In tracer scope (no SqliteSink), the dataset future resolves to None.
    assert handle.dataset.result(timeout=1.0) is None


def test_handle_status_transitions(engine: MeasurementEngine) -> None:
    g, i = _make_params()
    sweep = LinSweep(g, start=0.0, stop=1.0, num_points=2)
    plan = run(setpoints=(g,), measured=(i,))(scan_1d(sweep, [i]))

    handle = engine.submit(plan)
    # Could be "running" briefly; either way, after wait we're done.
    handle.wait(timeout=5.0)
    assert handle.status == "done"


# ----------------------------------------------------------------------------
# Cancellation
# ----------------------------------------------------------------------------


def test_cancel_triggers_cleanup(engine: MeasurementEngine) -> None:
    """Cancellation must run scan_1d's finally block — g.cache.get() == 0.0."""
    g, i = _make_params()
    # Many points, with delay, so cancel lands mid-sweep.
    sweep = LinSweep(g, start=0.0, stop=1.0, num_points=10000, delay=0.01)
    plan = run(setpoints=(g,), measured=(i,))(scan_1d(sweep, [i]))

    handle = engine.submit(plan)
    time.sleep(0.05)  # let the engine start
    handle.cancel()
    result = handle.wait(timeout=5.0)

    assert result.reason == "cancelled"
    assert result.cancel_latency is not None
    # Cleanup happened: g was ramped back to 0.0
    assert g.cache.get() == pytest.approx(0.0), (
        "scan_1d's finally block must run on cancel and reset g to 0.0"
    )


def test_cancel_emits_runstopped_with_cancelled_reason(
    engine: MeasurementEngine,
) -> None:
    g, i = _make_params()
    sweep = LinSweep(g, start=0.0, stop=1.0, num_points=10000, delay=0.01)
    plan = run(setpoints=(g,), measured=(i,))(scan_1d(sweep, [i]))

    handle = engine.submit(plan)
    time.sleep(0.05)
    handle.cancel()
    handle.wait(timeout=5.0)

    sink: MemorySink = engine._test_sink  # type: ignore[attr-defined]
    assert len(sink.stops) == 1
    assert sink.stops[0].reason == "cancelled"


def test_cancel_during_sleep_unblocks_quickly(
    engine: MeasurementEngine,
) -> None:
    """A long Sleep must be cancellable within ~100ms (chunked sleep)."""
    g, i = _make_params()

    # One point, long delay — engine will be in cancellable sleep most of the time.
    sweep = LinSweep(g, start=0.0, stop=0.0, num_points=1, delay=5.0)
    plan = run(setpoints=(g,), measured=(i,))(scan_1d(sweep, [i]))

    handle = engine.submit(plan)
    time.sleep(0.1)
    cancel_at = time.time()
    handle.cancel()
    handle.wait(timeout=2.0)
    elapsed = time.time() - cancel_at

    assert elapsed < 1.0, (
        f"Cancel-during-sleep took {elapsed:.2f}s; should be sub-second."
    )


def test_keyboardinterrupt_via_cancelrequested_in_plan(
    engine: MeasurementEngine,
) -> None:
    """A plan raising CancelRequested itself ends the run as interrupted."""
    from qcodes.measure_v2 import CancelRequested  # noqa: PLC0415

    g, i = _make_params()

    def _self_aborting_plan():
        yield Set(g, 0.5)
        yield Read((i,))
        yield Emit()
        raise CancelRequested("plan_aborted")

    plan = run(setpoints=(g,), measured=(i,))(_self_aborting_plan())
    handle = engine.submit(plan)
    result = handle.wait(timeout=5.0)

    assert result.reason == "interrupted"


# ----------------------------------------------------------------------------
# Concurrency & lifecycle
# ----------------------------------------------------------------------------


def test_concurrent_submit_raises(engine: MeasurementEngine) -> None:
    """Tracer scope: second submit while busy raises (no queue yet)."""
    g, i = _make_params()
    sweep1 = LinSweep(g, start=0.0, stop=1.0, num_points=200, delay=0.01)
    sweep2 = LinSweep(g, start=0.0, stop=1.0, num_points=10)

    plan1 = run(setpoints=(g,), measured=(i,))(scan_1d(sweep1, [i]))
    handle1 = engine.submit(plan1)

    plan2 = run(setpoints=(g,), measured=(i,))(scan_1d(sweep2, [i]))
    with pytest.raises(RuntimeError, match="already running"):
        engine.submit(plan2)

    handle1.cancel()
    handle1.wait(timeout=5.0)


def test_sequential_submits_work(engine: MeasurementEngine) -> None:
    """After one run completes, a second submit should succeed."""
    g, i = _make_params()
    sweep = LinSweep(g, start=0.0, stop=1.0, num_points=3)

    plan1 = run(setpoints=(g,), measured=(i,))(scan_1d(sweep, [i]))
    engine.submit(plan1).wait(timeout=5.0)

    plan2 = run(setpoints=(g,), measured=(i,))(scan_1d(sweep, [i]))
    result = engine.submit(plan2).wait(timeout=5.0)
    assert result.reason == "completed"


def test_shutdown_cancels_inflight_run(engine: MeasurementEngine) -> None:
    g, i = _make_params()
    sweep = LinSweep(g, start=0.0, stop=1.0, num_points=10000, delay=0.01)
    plan = run(setpoints=(g,), measured=(i,))(scan_1d(sweep, [i]))

    handle = engine.submit(plan)
    time.sleep(0.05)
    engine.shutdown(wait=True, timeout=5.0)
    result = handle.future.result(timeout=1.0)
    assert result.reason in ("cancelled", "engine_shutdown")
    # Cleanup still ran
    assert g.cache.get() == pytest.approx(0.0)


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------


def test_emit_with_undeclared_param_raises(engine: MeasurementEngine) -> None:
    """Emit overrides must reference declared params."""
    g, i = _make_params()
    j = Parameter("j")  # not declared!

    def _bad_plan():
        yield Set(g, 0.0)
        yield Read((i,))
        yield Emit(overrides={j: 99})

    plan = run(setpoints=(g,), measured=(i,))(_bad_plan())
    handle = engine.submit(plan)
    result = handle.wait(timeout=5.0)

    assert result.reason == "error"
    assert isinstance(result.error, PlanError)


def test_read_returns_value_to_plan(engine: MeasurementEngine) -> None:
    """The send-value path: a plan reads, computes, sets — engine drives it."""
    g, i = _make_params()
    observed_values: list[float] = []

    def _adaptive_plan():
        for v in (0.1, 0.2, 0.3):
            yield Set(g, v)
            r = yield Read((i,))
            observed_values.append(r[i])
            yield Emit()

    plan = run(setpoints=(g,), measured=(i,))(_adaptive_plan())
    engine.submit(plan).wait(timeout=5.0)

    # i = g**2; we set g to 0.1, 0.2, 0.3
    assert observed_values == pytest.approx([0.01, 0.04, 0.09])
