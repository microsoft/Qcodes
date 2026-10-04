"""L3 tests for the SQLite sink and engine end-to-end with persistence."""

from __future__ import annotations

import time

import pytest

from qcodes.dataset.dond.sweeps import LinSweep
from qcodes.measure_v2 import (
    MeasurementEngine,
    SqliteSink,
    run,
    scan_1d,
)
from qcodes.parameters import Parameter, ParameterBase


@pytest.fixture
def engine_with_sqlite(empty_db) -> MeasurementEngine:
    del empty_db  # fixture activated for side effects
    sink = SqliteSink(experiment_name="measure_v2_test", sample_name="tracer")
    eng = MeasurementEngine(sinks=[sink])
    try:
        yield eng
    finally:
        eng.shutdown(wait=True, timeout=5.0)


def _make_params() -> tuple[ParameterBase, ParameterBase]:
    g = Parameter("g", initial_value=0.0, set_cmd=None, get_cmd=None)
    i = Parameter("i", get_cmd=lambda: g.cache.get() ** 2)
    return g, i


def test_sqlite_sink_persists_rows(
    engine_with_sqlite: MeasurementEngine,
) -> None:
    g, i = _make_params()
    sweep = LinSweep(g, start=0.0, stop=2.0, num_points=5)
    plan = run(name="persisted", setpoints=(g,), measured=(i,))(scan_1d(sweep, [i]))

    handle = engine_with_sqlite.submit(plan)
    result = handle.wait(timeout=10.0)

    assert result.reason == "completed"
    assert result.n_rows_emitted == 5

    dataset = handle.dataset.result(timeout=1.0)
    assert dataset is not None
    assert dataset.completed

    # Read via the in-memory cache (thread-safe-ish) since the dataset's
    # connection is bound to the publisher thread.
    cache_data = dataset.cache.data()
    # cache.data() returns {measured_param_name: {param_name: ndarray}}
    assert "i" in cache_data
    measured = cache_data["i"]
    g_vals = list(measured["g"])
    i_vals = list(measured["i"])
    assert g_vals == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])
    assert i_vals == pytest.approx([0.0, 0.25, 1.0, 2.25, 4.0])


def test_sqlite_sink_handle_dataset_resolves(
    engine_with_sqlite: MeasurementEngine,
) -> None:
    g, i = _make_params()
    sweep = LinSweep(g, start=0.0, stop=1.0, num_points=3)
    plan = run(setpoints=(g,), measured=(i,))(scan_1d(sweep, [i]))

    handle = engine_with_sqlite.submit(plan)
    # The dataset future should resolve once RunStarted is processed.
    dataset = handle.dataset.result(timeout=5.0)
    assert dataset is not None
    handle.wait(timeout=5.0)


def test_sqlite_sink_cancel_finalizes_partial_dataset(
    engine_with_sqlite: MeasurementEngine,
) -> None:
    g, i = _make_params()
    sweep = LinSweep(g, start=0.0, stop=1.0, num_points=10000, delay=0.01)
    plan = run(setpoints=(g,), measured=(i,))(scan_1d(sweep, [i]))

    handle = engine_with_sqlite.submit(plan)
    time.sleep(0.05)
    handle.cancel()
    result = handle.wait(timeout=5.0)

    assert result.reason == "cancelled"
    dataset = handle.dataset.result(timeout=1.0)
    assert dataset is not None
    assert dataset.completed  # exit ran successfully despite cancel
    cache_data = dataset.cache.data()
    # Some rows were written before cancel
    assert len(cache_data["i"]["g"]) >= 1
