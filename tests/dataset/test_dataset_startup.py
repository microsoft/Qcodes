"""Regression tests for atomic dataset startup with a retained snapshot."""

from __future__ import annotations

import json
from contextlib import closing
from typing import TYPE_CHECKING

import numpy as np
import pytest

import qcodes as qc
from qcodes.dataset.data_set import DataSet, _BackgroundWriter
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.guids import generate_guid
from qcodes.dataset.linked_datasets.links import Link
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.sqlite.database import connect
from qcodes.parameters import ParamSpecBase

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path
    from typing import Any

    from pytest_mock import MockerFixture

    from qcodes.dataset.experiment_container import Experiment


@pytest.fixture(params=["sqlite_main_db", "sqlite_per_dataset_db"])
def startup_dataset(
    experiment: Experiment, request: pytest.FixtureRequest, tmp_path: Path
) -> Generator[DataSet, None, None]:
    """Create a pristine dataset using either supported SQLite backend."""
    qc.config.dataset.raw_data_backend = request.param
    qc.config.dataset.raw_data_backend_config = {
        "sqlite_per_dataset_db": {"raw_data_path": str(tmp_path / "raw_data")}
    }
    dataset = DataSet(conn=experiment.conn, metadata={"retained": "metadata"})
    try:
        yield dataset
    finally:
        if dataset.running:
            dataset.mark_completed()
        dataset.unsubscribe_all()
        dataset._results_backend.close()


@pytest.fixture
def startup_interdeps() -> InterDependencies_:
    """Describe a dependent parameter and its setpoint."""
    return InterDependencies_(
        dependencies={
            ParamSpecBase("current", "numeric"): (ParamSpecBase("voltage", "numeric"),)
        }
    )


def _read_persisted(dataset: DataSet) -> dict[str, Any]:
    """Read startup state using a separately owned database connection."""
    assert dataset.path_to_db is not None
    with closing(connect(dataset.path_to_db)) as conn:
        loaded = DataSet(conn=conn, run_id=dataset.run_id)
        try:
            return {
                "snapshot": loaded.snapshot,
                "started": loaded.started,
                "timestamp": loaded.run_timestamp_raw,
                "interdeps": loaded.description.interdeps,
                "shapes": loaded.description.shapes,
                "links": loaded.parent_dataset_links,
                "metadata": loaded.metadata,
            }
        finally:
            loaded._results_backend.close()


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("in_memory_cache", [False, True])
def test_prepare_commits_snapshot_before_observers(
    startup_dataset: DataSet,
    startup_interdeps: InterDependencies_,
    background: bool,
    in_memory_cache: bool,
    mocker: MockerFixture,
) -> None:
    dataset = startup_dataset
    dataset._in_memory_cache = in_memory_cache
    snapshot = {"station": {"payload": "snapshot" * 1000, "value": np.int64(42)}}
    expected_snapshot = {"station": {"payload": "snapshot" * 1000, "value": 42}}
    parent = {"tail": generate_guid(), "edge_type": "derived", "description": "parent"}
    expected_links = [Link(head=dataset.guid, **parent)]
    boundaries: list[str] = []
    statements: list[str] = []

    def assert_committed(boundary: str) -> None:
        persisted = _read_persisted(dataset)
        assert persisted == {
            "snapshot": expected_snapshot,
            "started": True,
            "timestamp": dataset.run_timestamp_raw,
            "interdeps": startup_interdeps,
            "shapes": {"current": (1,)},
            "links": expected_links,
            "metadata": {"retained": "metadata"},
        }
        assert persisted["timestamp"] is not None
        assert not dataset.conn.in_transaction
        boundaries.append(boundary)

    prepare_cache = dataset.cache.prepare
    start_subscribers = dataset._start_pending_subscribers
    start_writer = _BackgroundWriter.start

    def cache_boundary() -> None:
        assert_committed("cache")
        prepare_cache()

    def subscriber_boundary() -> None:
        assert_committed("subscribers")
        start_subscribers()

    def writer_boundary(writer: _BackgroundWriter) -> None:
        assert_committed("writer")
        start_writer(writer)

    mocker.patch.object(dataset.cache, "prepare", side_effect=cache_boundary)
    mocker.patch.object(
        dataset, "_start_pending_subscribers", side_effect=subscriber_boundary
    )
    mocker.patch.object(
        _BackgroundWriter, "start", autospec=True, side_effect=writer_boundary
    )
    dataset.conn.set_trace_callback(statements.append)
    try:
        dataset.prepare(
            snapshot=snapshot,
            interdeps=startup_interdeps,
            shapes={"current": (1,)},
            parent_datasets=[parent],
            write_in_background=background,
        )
    finally:
        dataset.conn.set_trace_callback(None)

    update_indices = [
        index
        for index, sql in enumerate(statements)
        if sql.upper().replace('"', "").replace("'", "").split()[:2]
        == ["UPDATE", "RUNS"]
    ]
    assert len(update_indices) > 1
    assert "snapshot" in statements[update_indices[-1]]
    assert not any(
        sql in ("BEGIN", "COMMIT")
        for sql in statements[update_indices[0] : update_indices[-1]]
    )
    assert boundaries == (["writer"] if background else []) + ["cache", "subscribers"]
    assert dataset.started
    dataset.add_results([{"voltage": 1.0, "current": 0.125}])
    dataset.mark_completed()
    assert dataset.number_of_results == 1
    np.testing.assert_array_equal(
        dataset.get_parameter_data()["current"]["current"], [0.125]
    )
    assert dataset.snapshot == expected_snapshot
    assert _read_persisted(dataset)["snapshot"] == expected_snapshot


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("stage", ["parameter", "timestamp", "snapshot", "backend"])
@pytest.mark.parametrize("interrupt", [False, True])
def test_prepare_failure_rolls_back(
    startup_dataset: DataSet,
    startup_interdeps: InterDependencies_,
    background: bool,
    stage: str,
    interrupt: bool,
    mocker: MockerFixture,
) -> None:
    dataset = startup_dataset
    before = _read_persisted(dataset)
    writer_before = (
        dataset._writer_status.write_in_background,
        dataset._writer_status.bg_writer,
        dataset._writer_status.active_datasets.copy(),
    )
    error = KeyboardInterrupt() if interrupt else ValueError("startup failure")
    if stage == "snapshot":
        add_snapshot = dataset.add_snapshot

        def fail_snapshot(snapshot: str, overwrite: bool = False) -> None:
            add_snapshot(snapshot, overwrite)
            raise error

        patch = mocker.patch.object(dataset, "add_snapshot", side_effect=fail_snapshot)
    elif stage == "backend":
        patch = mocker.patch.object(
            dataset._results_backend, "setup_on_start", side_effect=error
        )
    else:
        function = "add_parameter" if stage == "parameter" else "set_run_timestamp"
        patch = mocker.patch(f"qcodes.dataset.data_set.{function}", side_effect=error)

    cache = mocker.spy(dataset.cache, "prepare")
    subscribers = mocker.spy(dataset, "_start_pending_subscribers")
    expected_exception = KeyboardInterrupt if interrupt else RuntimeError
    with pytest.raises(expected_exception) as exc_info:
        dataset.prepare(
            snapshot={"station": {"retained": "snapshot"}},
            interdeps=startup_interdeps,
            write_in_background=background,
        )
    if interrupt:
        assert exc_info.value is error
    else:
        assert exc_info.value.__cause__ is error
    assert _read_persisted(dataset) == before
    assert dataset.pristine
    assert not dataset.conn.in_transaction
    assert not dataset.conn.atomic_in_progress
    assert (
        dataset._writer_status.write_in_background,
        dataset._writer_status.bg_writer,
        dataset._writer_status.active_datasets,
    ) == writer_before
    assert dataset.conn.execute(
        "SELECT COUNT(*) FROM layouts WHERE run_id = ?", (dataset.run_id,)
    ).fetchone() == (0,)
    cache.assert_not_called()
    subscribers.assert_not_called()

    mocker.stop(patch)
    dataset.prepare(
        snapshot={"station": {"retained": "snapshot"}},
        interdeps=startup_interdeps,
        write_in_background=background,
    )
    dataset.add_results([{"voltage": 1.0, "current": 0.125}])
    dataset.mark_completed()
    assert dataset.number_of_results == 1
    assert dataset.snapshot == {"station": {"retained": "snapshot"}}


@pytest.mark.parametrize("background", [False, True])
def test_prepare_writer_conflict_does_not_publish_startup(
    startup_dataset: DataSet,
    startup_interdeps: InterDependencies_,
    background: bool,
) -> None:
    dataset = startup_dataset
    first = DataSet(conn=dataset.conn)
    before = _read_persisted(dataset)
    try:
        first.mark_started(start_bg_writer=background)
        with pytest.raises(RuntimeError, match="All datasets written"):
            dataset.prepare(
                snapshot={"station": "snapshot"},
                interdeps=startup_interdeps,
                write_in_background=not background,
            )
        assert dataset.pristine
        assert _read_persisted(dataset) == before
        assert dataset.run_id not in dataset._writer_status.active_datasets
    finally:
        first.mark_completed()
        first._results_backend.close()


def test_prepare_preserves_existing_snapshot(
    startup_dataset: DataSet, startup_interdeps: InterDependencies_
) -> None:
    dataset = startup_dataset
    dataset.add_snapshot(json.dumps({"original": "snapshot"}))
    dataset.prepare(snapshot={"replacement": "snapshot"}, interdeps=startup_interdeps)
    assert dataset.snapshot == {"original": "snapshot"}
    timestamp = dataset.run_timestamp_raw
    dataset.mark_started(start_bg_writer=True)
    assert dataset.run_timestamp_raw == timestamp
    assert dataset._writer_status.write_in_background is False


def test_mark_started_without_snapshot(
    startup_dataset: DataSet, startup_interdeps: InterDependencies_
) -> None:
    dataset = startup_dataset
    dataset.set_interdependencies(startup_interdeps)
    dataset.mark_started()
    assert dataset.started
    assert dataset.snapshot is None
    dataset.add_results([{"voltage": 1.0, "current": 0.125}])
    dataset.mark_completed()
    assert dataset.number_of_results == 1


def test_prepare_invalid_parameters_does_not_write_snapshot(
    startup_dataset: DataSet,
) -> None:
    dataset = startup_dataset
    with pytest.raises(RuntimeError, match="No parameters supplied"):
        dataset.prepare(
            snapshot={"station": "snapshot"}, interdeps=InterDependencies_()
        )
    assert dataset.snapshot is None
    assert dataset.pristine


def test_prepare_snapshot_encoding_error(
    startup_dataset: DataSet, startup_interdeps: InterDependencies_
) -> None:
    dataset = startup_dataset
    with pytest.raises(TypeError, match="keys must be"):
        dataset.prepare(snapshot={object(): "invalid"}, interdeps=startup_interdeps)
    assert dataset.pristine
    assert dataset.snapshot is None
    assert dataset.run_timestamp_raw is None


def test_pending_subscriber_sees_committed_snapshot(
    startup_dataset: DataSet,
    startup_interdeps: InterDependencies_,
    mocker: MockerFixture,
) -> None:
    dataset = startup_dataset
    observed: list[dict[str, Any]] = []

    def callback(results: object, length: int, state: object) -> None:
        observed.append(_read_persisted(dataset))

    create_subscriber = dataset._create_and_start_subscriber

    def subscriber_boundary(*args: Any, **kwargs: Any) -> None:
        assert _read_persisted(dataset)["snapshot"] == {"station": "snapshot"}
        create_subscriber(*args, **kwargs)

    dataset.subscribe(callback, min_count=1)
    mocker.patch.object(
        dataset, "_create_and_start_subscriber", side_effect=subscriber_boundary
    )
    dataset.prepare(snapshot={"station": "snapshot"}, interdeps=startup_interdeps)
    dataset.add_results([{"voltage": 1.0, "current": 0.125}])
    dataset.mark_completed()
    dataset.unsubscribe_all()
    assert observed
    assert all(state["snapshot"] == {"station": "snapshot"} for state in observed)
    assert all(state["started"] for state in observed)


def test_measurement_subscriber_registration_sees_snapshot(
    startup_dataset: DataSet, mocker: MockerFixture
) -> None:
    measurement = Measurement()
    measurement.register_custom_parameter("voltage")
    measurement.add_subscriber(lambda results, length, state: None, state={})
    subscribe = DataSet.subscribe
    snapshots: list[dict[str, Any]] = []

    def subscribe_boundary(dataset: DataSet, *args: Any, **kwargs: Any) -> str:
        assert dataset.started
        persisted = _read_persisted(dataset)
        assert persisted["snapshot"] is not None
        snapshots.append(persisted)
        return subscribe(dataset, *args, **kwargs)

    mocker.patch.object(
        DataSet, "subscribe", autospec=True, side_effect=subscribe_boundary
    )
    with measurement.run() as datasaver:
        assert datasaver.dataset.snapshot == snapshots[0]["snapshot"]
        datasaver.add_result(("voltage", 1.0))
    assert len(snapshots) == 1
