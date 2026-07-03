"""Tests for :mod:`qcodes.dataset.sqlite.db_overview`."""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

import pytest

from qcodes.dataset import (
    get_db_overview,
    initialise_or_create_database_at,
    load_or_create_experiment,
    new_data_set,
)
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.sqlite.database import connect
from qcodes.dataset.sqlite.db_overview import (
    _format_timestamp,
    _records_from_run_description,
)
from qcodes.parameters import ParamSpecBase

if TYPE_CHECKING:
    from pathlib import Path


def _make_db_with_runs(
    db_path: str,
    n_runs: int = 1,
    n_points: int = 10,
    experiment_name: str = "test_exp",
    sample_name: str = "test_sample",
) -> None:
    """Create a database at ``db_path`` with ``n_runs`` simple numeric runs."""
    initialise_or_create_database_at(db_path)
    load_or_create_experiment(experiment_name, sample_name=sample_name)
    p_x = ParamSpecBase("x", "numeric")
    p_y = ParamSpecBase("y", "numeric")
    interdeps = InterDependencies_(dependencies={p_y: (p_x,)})

    for r in range(n_runs):
        ds = new_data_set(f"run_{r + 1}")
        ds.set_interdependencies(interdeps)
        ds.mark_started()
        for i in range(n_points):
            ds.add_results([{p_x.name: float(i), p_y.name: float(i**2)}])
        ds.mark_completed()
        ds.conn.close()


def test_records_from_run_description() -> None:
    desc = json.dumps({"version": 3, "shapes": {"dep1": [100, 50]}})
    assert _records_from_run_description(desc) == 5000

    multi = json.dumps({"shapes": {"dep1": [10], "dep2": [5, 4]}})
    assert _records_from_run_description(multi) == 30

    assert _records_from_run_description(json.dumps({"version": 3})) == 0
    assert _records_from_run_description(json.dumps({"shapes": {}})) == 0
    assert _records_from_run_description(None) == 0
    assert _records_from_run_description("") == 0
    assert _records_from_run_description("not valid json") == 0


def test_format_timestamp() -> None:
    assert _format_timestamp(None) == ("", "")
    assert _format_timestamp(0) == ("", "")

    date, time = _format_timestamp(1_600_000_000.0)
    assert len(date) == len("YYYY-MM-DD")
    assert date.count("-") == 2
    assert len(time) == len("HH:MM:SS")
    assert time.count(":") == 2


def test_get_db_overview_basic_fields(tmp_path: Path) -> None:
    db_path = str(tmp_path / "basic.db")
    _make_db_with_runs(db_path, n_runs=2)

    overview = get_db_overview(db_path)

    assert set(overview.keys()) == {1, 2}
    for run_id, info in overview.items():
        assert info["run_id"] == run_id
        assert info["experiment"] == "test_exp"
        assert info["sample"] == "test_sample"
        assert info["name"] == f"run_{run_id}"
        assert info["guid"]
        assert info["started_date"] and info["started_time"]
        assert info["completed_date"] and info["completed_time"]


def test_get_db_overview_counts_result_rows(tmp_path: Path) -> None:
    db_path = str(tmp_path / "counts.db")
    _make_db_with_runs(db_path, n_runs=3, n_points=10)

    overview = get_db_overview(db_path)

    conn = sqlite3.connect(db_path)
    try:
        for run_id, info in overview.items():
            (table_name,) = conn.execute(
                "SELECT result_table_name FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            (actual,) = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()
            assert info["records"] == actual == 10
    finally:
        conn.close()


def test_get_db_overview_incremental(tmp_path: Path) -> None:
    db_path = str(tmp_path / "incremental.db")
    _make_db_with_runs(db_path, n_runs=2)

    assert set(get_db_overview(db_path).keys()) == {1, 2}
    assert get_db_overview(db_path, start_run_id=2) == {}

    _make_db_with_runs(db_path, n_runs=1, experiment_name="test_exp2", sample_name="s2")

    incremental = get_db_overview(db_path, start_run_id=2)
    assert set(incremental.keys()) == {3}
    assert incremental[3]["experiment"] == "test_exp2"


def test_get_db_overview_extra_columns(tmp_path: Path) -> None:
    db_path = str(tmp_path / "extra.db")
    initialise_or_create_database_at(db_path)
    load_or_create_experiment("exp", sample_name="sample")
    p_x = ParamSpecBase("x", "numeric")
    p_y = ParamSpecBase("y", "numeric")
    interdeps = InterDependencies_(dependencies={p_y: (p_x,)})
    ds = new_data_set("tagged_run")
    ds.set_interdependencies(interdeps)
    ds.mark_started()
    ds.add_results([{p_x.name: 1.0, p_y.name: 2.0}])
    ds.mark_completed()
    ds.add_metadata("my_tag", "hello")
    ds.conn.close()

    # An existing ad-hoc metadata column is returned ...
    overview = get_db_overview(db_path, extra_columns=["my_tag"])
    assert overview[1]["my_tag"] == "hello"  # type: ignore[typeddict-item]

    # ... while a non-existent column is silently skipped.
    overview = get_db_overview(db_path, extra_columns=["does_not_exist"])
    assert "does_not_exist" not in overview[1]


def test_get_db_overview_accepts_connection(tmp_path: Path) -> None:
    db_path = str(tmp_path / "conn.db")
    _make_db_with_runs(db_path, n_runs=1)

    conn = connect(db_path)
    try:
        overview = get_db_overview(conn=conn)
        assert set(overview.keys()) == {1}
        # the externally supplied connection must remain usable
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    finally:
        conn.close()


def test_get_db_overview_rejects_conn_and_path(tmp_path: Path) -> None:
    db_path = str(tmp_path / "both.db")
    _make_db_with_runs(db_path, n_runs=1)

    conn = connect(db_path)
    try:
        with pytest.raises(ValueError):
            get_db_overview(path_to_db=db_path, conn=conn)
    finally:
        conn.close()
