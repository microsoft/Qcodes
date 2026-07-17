"""
Tests for per-dataset raw data SQLite storage.

When ``dataset.raw_data_to_separate_db`` is enabled, measurement data
(results tables) are written to individual SQLite files while metadata
remains in the main database.
"""

from __future__ import annotations

import gc
import shutil
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

import qcodes as qc
from qcodes.dataset import new_data_set, new_experiment
from qcodes.dataset._raw_data_storage import (
    cleanup_datasets,
    connect_to_raw_data_db,
    create_raw_data_db,
    get_raw_data_db_path,
    get_raw_data_folder,
    is_raw_data_storage_enabled,
    purge_orphaned_datasets,
    update_raw_data_paths,
)
from qcodes.dataset.data_set import DataSet, load_by_id
from qcodes.dataset.database_extract_runs import (
    export_datasets_and_create_metadata_db,
    extract_runs_into_db,
)
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.sqlite.database import connect, initialise_database
from qcodes.parameters import ParamSpecBase

if TYPE_CHECKING:
    from collections.abc import Generator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _raw_data_db(tmp_path: Path) -> Generator[None, None, None]:
    """Set up a temp DB with raw_data_to_separate_db enabled."""
    db_path = str(tmp_path / "test.db")
    qc.config["core"]["db_location"] = db_path
    qc.config["core"]["db_debug"] = False
    qc.config["dataset"]["raw_data_to_separate_db"] = True
    qc.config["dataset"]["raw_data_path"] = str(tmp_path / "raw_data")
    initialise_database()
    try:
        yield
    finally:
        qc.config["dataset"]["raw_data_to_separate_db"] = False
        qc.config["dataset"]["raw_data_path"] = "{db_location}"
        gc.collect()


@pytest.fixture()
def _raw_data_experiment(_raw_data_db: None) -> Generator[None, None, None]:
    """Create a test experiment inside the raw data DB."""
    e = new_experiment("test-experiment", sample_name="test-sample")
    try:
        yield
    finally:
        e.conn.close()


# ---------------------------------------------------------------------------
# Unit tests - raw_data_storage module
# ---------------------------------------------------------------------------


class TestRawDataStorageHelpers:
    def test_is_raw_data_storage_enabled_default(self, tmp_path: Path) -> None:
        assert not is_raw_data_storage_enabled()

    def test_is_raw_data_storage_enabled_on(self, _raw_data_db: None) -> None:
        assert is_raw_data_storage_enabled()

    def test_get_raw_data_folder(self, _raw_data_db: None, tmp_path: Path) -> None:
        folder = get_raw_data_folder()
        assert folder == (tmp_path / "raw_data")

    def test_get_raw_data_db_path(self, tmp_path: Path) -> None:
        folder = tmp_path / "raw_data"
        path = get_raw_data_db_path("abc-123", folder)
        assert path == folder / "abc-123.db"

    def test_create_raw_data_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "raw" / "test.db"
        params = [
            ParamSpecBase("x", "numeric"),
            ParamSpecBase("y", "numeric"),
        ]
        conn = create_raw_data_db(db_path, "results_table", params)
        try:
            # Verify table exists and has the right columns
            cursor = conn.execute("PRAGMA table_info('results_table')")
            columns = {row[1] for row in cursor.fetchall()}
            assert "id" in columns
            assert "x" in columns
            assert "y" in columns
        finally:
            conn.close()

    def test_connect_to_raw_data_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = connect_to_raw_data_db(db_path)
        try:
            # Should be a valid connection with numpy adapters
            conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val REAL)")
            conn.execute("INSERT INTO t (val) VALUES (?)", (3.14,))
            conn.commit()
            cursor = conn.execute("SELECT val FROM t")
            assert cursor.fetchone()[0] == pytest.approx(3.14)
        finally:
            conn.close()

    def test_connect_to_raw_data_db_read_only(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        # Create first
        conn = connect_to_raw_data_db(db_path)
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()
        # Now open read-only
        conn = connect_to_raw_data_db(db_path, read_only=True)
        try:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("INSERT INTO t (id) VALUES (1)")
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Integration tests - DataSet with split raw data
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_raw_data_experiment")
class TestDataSetWithSplitRawData:
    def _make_dataset_with_data(
        self, n_rows: int = 10
    ) -> tuple[DataSet, list[dict[str, float]]]:
        """Create a dataset, add data, mark completed, return ds and data."""
        ds = new_data_set("test-split")
        x = ParamSpecBase("x", "numeric")
        y = ParamSpecBase("y", "numeric")
        idps = InterDependencies_(dependencies={y: (x,)})
        ds.set_interdependencies(idps)
        ds.mark_started()

        results = [{"x": float(i), "y": float(i**2)} for i in range(n_rows)]
        ds.add_results(results)
        ds.mark_completed()
        return ds, results

    @staticmethod
    def _close_ds(ds: DataSet) -> None:
        """Close both main and raw data connections."""
        if ds._raw_data_conn is not None:
            ds._raw_data_conn.close()
        ds.conn.close()

    def test_raw_data_conn_is_set(self) -> None:
        """When split is enabled, DataSet should have a raw data connection."""
        ds = new_data_set("test-split")
        x = ParamSpecBase("x", "numeric")
        y = ParamSpecBase("y", "numeric")
        idps = InterDependencies_(dependencies={y: (x,)})
        ds.set_interdependencies(idps)
        ds.mark_started()
        assert ds._raw_data_conn is not None
        self._close_ds(ds)

    def test_raw_data_file_created(self, tmp_path: Path) -> None:
        """A per-dataset SQLite file should be created."""
        ds, _ = self._make_dataset_with_data()
        raw_folder = tmp_path / "raw_data"
        raw_file = raw_folder / f"{ds.guid}.db"
        assert raw_file.is_file()
        self._close_ds(ds)

    def test_raw_data_db_path_in_metadata(self) -> None:
        """The path to the raw data file should be stored in metadata."""
        ds, _ = self._make_dataset_with_data()
        assert "raw_data_db_path" in ds.metadata
        assert Path(ds.metadata["raw_data_db_path"]).is_file()
        self._close_ds(ds)

    def test_data_is_in_raw_db_not_main(self) -> None:
        """Data should be in the raw data file, not the main DB."""
        ds, _ = self._make_dataset_with_data(n_rows=5)
        table_name = ds.table_name
        main_conn = ds.conn

        # Main DB should have the results table (schema) but no data rows
        cursor = main_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        assert cursor.fetchone() is not None
        cursor = main_conn.execute(f'SELECT COUNT(*) FROM "{table_name}"')
        assert cursor.fetchone()[0] == 0

        # Raw data DB should have the actual data
        raw_conn = ds._raw_data_conn
        assert raw_conn is not None
        cursor = raw_conn.execute(f'SELECT COUNT(*) FROM "{table_name}"')
        raw_count = cursor.fetchone()[0]
        assert raw_count == 5
        self._close_ds(ds)

    def test_number_of_results(self) -> None:
        """number_of_results should read from the raw data file."""
        ds, _ = self._make_dataset_with_data(n_rows=7)
        assert ds.number_of_results == 7
        self._close_ds(ds)

    def test_get_parameter_data(self) -> None:
        """get_parameter_data should read from the raw data file."""
        ds, results = self._make_dataset_with_data(n_rows=5)
        data = ds.get_parameter_data()
        assert "y" in data
        np.testing.assert_array_almost_equal(
            data["y"]["x"], np.array([r["x"] for r in results])
        )
        np.testing.assert_array_almost_equal(
            data["y"]["y"], np.array([r["y"] for r in results])
        )
        self._close_ds(ds)

    def test_cache_loads_from_raw_data(self) -> None:
        """The cache should also read from the raw data file."""
        ds, results = self._make_dataset_with_data(n_rows=5)
        cache = ds.cache
        cache.load_data_from_db()
        cache_data = cache.data()
        assert "y" in cache_data
        np.testing.assert_array_almost_equal(
            cache_data["y"]["x"], np.array([r["x"] for r in results])
        )
        self._close_ds(ds)

    def test_load_by_id_with_split_data(self) -> None:
        """Loading by ID should automatically use the raw data connection."""
        ds, results = self._make_dataset_with_data(n_rows=3)
        run_id = ds.run_id
        self._close_ds(ds)

        # Re-load from the database
        loaded = load_by_id(run_id)
        assert isinstance(loaded, DataSet)
        assert loaded._raw_data_conn is not None
        data = loaded.get_parameter_data()
        np.testing.assert_array_almost_equal(
            data["y"]["y"], np.array([r["y"] for r in results])
        )
        self._close_ds(loaded)

    def test_multiple_datasets_split(self) -> None:
        """Multiple datasets should each get their own raw data file."""
        ds1, _ = self._make_dataset_with_data(n_rows=3)
        ds2, _ = self._make_dataset_with_data(n_rows=5)

        assert ds1._raw_data_conn is not None
        assert ds2._raw_data_conn is not None
        assert ds1._raw_data_conn.path_to_dbfile != ds2._raw_data_conn.path_to_dbfile
        assert ds1.number_of_results == 3
        assert ds2.number_of_results == 5
        self._close_ds(ds1)
        self._close_ds(ds2)

    def test_metadata_remains_in_main_db(self) -> None:
        """Run metadata should be in the main DB, not the raw data DB."""
        ds, _ = self._make_dataset_with_data()
        main_conn = ds.conn
        # runs table should be populated in main DB
        cursor = main_conn.execute(
            "SELECT name, is_completed FROM runs WHERE run_id=?", (ds.run_id,)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "test-split"
        assert row[1] == 1  # completed
        self._close_ds(ds)

    def test_missing_raw_data_file_raises(self, tmp_path: Path) -> None:
        """Loading a dataset whose raw data file is missing should raise."""
        ds, _ = self._make_dataset_with_data(n_rows=3)
        run_id = ds.run_id
        raw_path = Path(ds.metadata["raw_data_db_path"])
        self._close_ds(ds)

        # Delete the raw data file
        raw_path.unlink()
        assert not raw_path.exists()

        # Attempting to load should raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match=r"Raw data file.*not found"):
            load_by_id(run_id)


# ---------------------------------------------------------------------------
# Tests that the feature doesn't interfere when disabled
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("experiment")
class TestDataSetWithoutSplitRawData:
    def test_raw_data_conn_is_none(self) -> None:
        """When split is disabled, _raw_data_conn should be None."""
        ds = new_data_set("test-no-split")
        x = ParamSpecBase("x", "numeric")
        y = ParamSpecBase("y", "numeric")
        idps = InterDependencies_(dependencies={y: (x,)})
        ds.set_interdependencies(idps)
        ds.mark_started()
        assert ds._raw_data_conn is None
        ds.conn.close()

    def test_data_in_main_db(self) -> None:
        """When split is disabled, data should be in the main DB as usual."""
        ds = new_data_set("test-no-split")
        x = ParamSpecBase("x", "numeric")
        y = ParamSpecBase("y", "numeric")
        idps = InterDependencies_(dependencies={y: (x,)})
        ds.set_interdependencies(idps)
        ds.mark_started()
        ds.add_results([{"x": 1.0, "y": 2.0}])
        ds.mark_completed()

        cursor = ds.conn.execute(f'SELECT COUNT(*) FROM "{ds.table_name}"')
        assert cursor.fetchone()[0] == 1
        assert ds.number_of_results == 1
        ds.conn.close()


# ---------------------------------------------------------------------------
# Tests for update_raw_data_paths helper
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_raw_data_experiment")
class TestUpdateRawDataPaths:
    @staticmethod
    def _close_ds(ds: DataSet) -> None:
        if ds._raw_data_conn is not None:
            ds._raw_data_conn.close()
        ds.conn.close()

    def test_update_after_move(self, tmp_path: Path) -> None:
        """Paths should be updated after raw data files are moved."""
        ds = new_data_set("test-move")
        x = ParamSpecBase("x", "numeric")
        y = ParamSpecBase("y", "numeric")
        idps = InterDependencies_(dependencies={y: (x,)})
        ds.set_interdependencies(idps)
        ds.mark_started()
        ds.add_results([{"x": 1.0, "y": 2.0}])
        ds.mark_completed()

        raw_path = Path(ds.metadata["raw_data_db_path"])
        db_path = ds.path_to_db
        assert db_path is not None
        self._close_ds(ds)

        # Move the raw data file to a new folder
        new_folder = tmp_path / "moved_raw"
        new_folder.mkdir()
        new_file = new_folder / raw_path.name

        shutil.move(str(raw_path), str(new_file))

        # Run the update
        updated = update_raw_data_paths(db_path, new_folder)
        assert len(updated) == 1
        run_id, old_path, new_path = updated[0]
        assert old_path == str(raw_path)
        assert new_path == str(new_file)

        # Verify the dataset can now be loaded
        loaded = load_by_id(run_id)
        assert isinstance(loaded, DataSet)
        assert loaded.number_of_results == 1
        data = loaded.get_parameter_data()
        assert data["y"]["x"][0] == 1.0
        self._close_ds(loaded)

    def test_update_skips_missing_files(self, tmp_path: Path) -> None:
        """Runs whose files are not in the new folder should be skipped."""
        ds = new_data_set("test-skip")
        x = ParamSpecBase("x", "numeric")
        y = ParamSpecBase("y", "numeric")
        idps = InterDependencies_(dependencies={y: (x,)})
        ds.set_interdependencies(idps)
        ds.mark_started()
        ds.add_results([{"x": 1.0, "y": 2.0}])
        ds.mark_completed()

        db_path = ds.path_to_db
        assert db_path is not None
        self._close_ds(ds)

        # Point to an empty folder — file not there
        empty_folder = tmp_path / "empty"
        empty_folder.mkdir()

        updated = update_raw_data_paths(db_path, empty_folder)
        assert len(updated) == 0

    def test_update_nonexistent_db_raises(self, tmp_path: Path) -> None:
        """Should raise if the main DB file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Database file not found"):
            update_raw_data_paths(tmp_path / "nonexistent.db", tmp_path)

    def test_update_nonexistent_folder_raises(self, tmp_path: Path) -> None:
        """Should raise if the new folder doesn't exist."""
        db_path = tmp_path / "test.db"
        db_path.touch()
        with pytest.raises(FileNotFoundError, match="New raw data folder not found"):
            update_raw_data_paths(db_path, tmp_path / "no_such_folder")


# ---------------------------------------------------------------------------
# purge_orphaned_datasets and cleanup_datasets tests
# ---------------------------------------------------------------------------


class TestPurgeOrphanedDatasets:
    """Tests for purge_orphaned_datasets."""

    @staticmethod
    def _close_ds(ds: DataSet) -> None:
        if ds._raw_data_conn is not None:
            ds._raw_data_conn.close()
        ds.conn.close()

    @pytest.mark.usefixtures("_raw_data_db")
    def test_dry_run_reports_orphans(self, tmp_path: Path) -> None:
        """Dry run should report orphaned datasets without removing them."""

        new_experiment("test-exp", sample_name="test-sample")
        ds = new_data_set("test-purge")
        x = ParamSpecBase("x", "numeric")
        y = ParamSpecBase("y", "numeric")
        idps = InterDependencies_(dependencies={y: (x,)})
        ds.set_interdependencies(idps)
        ds.mark_started()
        ds.add_results([{"x": 1.0, "y": 2.0}])
        ds.mark_completed()

        raw_path = Path(ds.metadata["raw_data_db_path"])
        db_path = ds.path_to_db
        assert db_path is not None
        self._close_ds(ds)

        # Delete the raw data file to create an orphan
        raw_path.unlink()

        result = purge_orphaned_datasets(db_path, dry_run=True)
        assert result.total_datasets_with_raw_data == 1
        assert len(result.orphaned_datasets) == 1
        assert len(result.removed_datasets) == 0
        assert result.dry_run is True

        # Verify the dataset is still in the DB (direct query since load_by_id
        # will raise FileNotFoundError due to missing raw data file)

        conn = connect(db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM runs WHERE run_id = 1")
        assert cursor.fetchone()[0] == 1
        conn.close()

    @pytest.mark.usefixtures("_raw_data_db")
    def test_removes_orphans_when_not_dry_run(self, tmp_path: Path) -> None:
        """With dry_run=False, orphaned datasets should be removed."""

        new_experiment("test-exp", sample_name="test-sample")
        ds = new_data_set("test-purge-remove")
        x = ParamSpecBase("x", "numeric")
        y = ParamSpecBase("y", "numeric")
        idps = InterDependencies_(dependencies={y: (x,)})
        ds.set_interdependencies(idps)
        ds.mark_started()
        ds.add_results([{"x": 1.0, "y": 2.0}])
        ds.mark_completed()

        raw_path = Path(ds.metadata["raw_data_db_path"])
        db_path = ds.path_to_db
        assert db_path is not None
        run_id = ds.run_id
        self._close_ds(ds)

        # Delete the raw data file
        raw_path.unlink()

        result = purge_orphaned_datasets(db_path, dry_run=False)
        assert len(result.orphaned_datasets) == 1
        assert len(result.removed_datasets) == 1
        assert result.removed_datasets[0].run_id == run_id
        assert result.dry_run is False

        # Verify the dataset no longer exists in the DB

        conn = connect(db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM runs WHERE run_id = ?", (run_id,))
        assert cursor.fetchone()[0] == 0
        conn.close()

    @pytest.mark.usefixtures("_raw_data_db")
    def test_skips_datasets_with_existing_files(self, tmp_path: Path) -> None:
        """Datasets whose raw data files exist should not be purged."""

        new_experiment("test-exp", sample_name="test-sample")
        ds = new_data_set("test-no-purge")
        x = ParamSpecBase("x", "numeric")
        y = ParamSpecBase("y", "numeric")
        idps = InterDependencies_(dependencies={y: (x,)})
        ds.set_interdependencies(idps)
        ds.mark_started()
        ds.add_results([{"x": 1.0, "y": 2.0}])
        ds.mark_completed()

        db_path = ds.path_to_db
        assert db_path is not None
        self._close_ds(ds)

        # Don't delete the file — should not be orphaned
        result = purge_orphaned_datasets(db_path, dry_run=False)
        assert len(result.orphaned_datasets) == 0
        assert len(result.removed_datasets) == 0

    def test_nonexistent_db_raises(self, tmp_path: Path) -> None:
        """Should raise if the DB file doesn't exist."""

        with pytest.raises(FileNotFoundError, match="Database file not found"):
            purge_orphaned_datasets(tmp_path / "nonexistent.db")


class TestCleanupDatasets:
    """Tests for cleanup_datasets."""

    @staticmethod
    def _close_ds(ds: DataSet) -> None:
        if ds._raw_data_conn is not None:
            ds._raw_data_conn.close()
        ds.conn.close()

    @pytest.mark.usefixtures("_raw_data_db")
    def test_cleanup_by_sample_name(self, tmp_path: Path) -> None:
        """Should remove datasets matching exact sample name."""

        new_experiment("exp1", sample_name="sample-A")
        ds1 = new_data_set("ds1")
        x = ParamSpecBase("x", "numeric")
        y = ParamSpecBase("y", "numeric")
        idps = InterDependencies_(dependencies={y: (x,)})
        ds1.set_interdependencies(idps)
        ds1.mark_started()
        ds1.add_results([{"x": 1.0, "y": 2.0}])
        ds1.mark_completed()
        raw_path1 = Path(ds1.metadata["raw_data_db_path"])
        db_path = ds1.path_to_db
        assert db_path is not None
        self._close_ds(ds1)

        # Create another dataset with a different sample
        new_experiment("exp2", sample_name="sample-B")
        ds2 = new_data_set("ds2")
        ds2.set_interdependencies(idps)
        ds2.mark_started()
        ds2.add_results([{"x": 3.0, "y": 4.0}])
        ds2.mark_completed()
        raw_path2 = Path(ds2.metadata["raw_data_db_path"])
        self._close_ds(ds2)

        # Dry run first
        result = cleanup_datasets(db_path, sample_name="sample-A", dry_run=True)
        assert len(result.matching_datasets) == 1
        assert len(result.removed_datasets) == 0
        assert raw_path1.is_file()

        # Actual removal
        result = cleanup_datasets(db_path, sample_name="sample-A", dry_run=False)
        assert len(result.matching_datasets) == 1
        assert len(result.removed_datasets) == 1
        assert not raw_path1.is_file()
        assert raw_path2.is_file()  # other sample untouched

    @pytest.mark.usefixtures("_raw_data_db")
    def test_cleanup_by_size(self, tmp_path: Path) -> None:
        """Should remove datasets larger than the specified threshold."""

        new_experiment("exp1", sample_name="test-sample")
        ds = new_data_set("ds-large")
        x = ParamSpecBase("x", "numeric")
        y = ParamSpecBase("y", "numeric")
        idps = InterDependencies_(dependencies={y: (x,)})
        ds.set_interdependencies(idps)
        ds.mark_started()
        # Add enough data so the file has some size
        for i in range(100):
            ds.add_results([{"x": float(i), "y": float(i * 2)}])
        ds.mark_completed()
        raw_path = Path(ds.metadata["raw_data_db_path"])
        db_path = ds.path_to_db
        assert db_path is not None
        self._close_ds(ds)

        file_size_mb = raw_path.stat().st_size / (1024 * 1024)

        # Set threshold below actual size — should match
        result = cleanup_datasets(
            db_path, larger_than_mb=file_size_mb * 0.5, dry_run=True
        )
        assert len(result.matching_datasets) == 1

        # Set threshold above actual size — should not match
        result = cleanup_datasets(
            db_path, larger_than_mb=file_size_mb * 2, dry_run=True
        )
        assert len(result.matching_datasets) == 0

    @pytest.mark.usefixtures("_raw_data_db")
    def test_cleanup_by_age(self, tmp_path: Path) -> None:
        """Should remove datasets older than the specified number of days."""

        new_experiment("exp1", sample_name="test-sample")
        ds = new_data_set("ds-old")
        x = ParamSpecBase("x", "numeric")
        y = ParamSpecBase("y", "numeric")
        idps = InterDependencies_(dependencies={y: (x,)})
        ds.set_interdependencies(idps)
        ds.mark_started()
        ds.add_results([{"x": 1.0, "y": 2.0}])
        ds.mark_completed()
        db_path = ds.path_to_db
        assert db_path is not None
        run_id = ds.run_id
        self._close_ds(ds)

        # Set the completed_timestamp to 10 days ago
        old_ts = time.time() - (10 * 86400)
        conn = connect(db_path)
        conn.execute(
            "UPDATE runs SET completed_timestamp = ? WHERE run_id = ?",
            (old_ts, run_id),
        )
        conn.commit()
        conn.close()

        # older_than_days=5 should match (ds is 10 days old)
        result = cleanup_datasets(db_path, older_than_days=5, dry_run=True)
        assert len(result.matching_datasets) == 1

        # older_than_days=15 should NOT match (ds is only 10 days old)
        result = cleanup_datasets(db_path, older_than_days=15, dry_run=True)
        assert len(result.matching_datasets) == 0

    def test_no_criteria_raises(self, tmp_path: Path) -> None:
        """Should raise if no criteria are specified."""

        db_path = tmp_path / "test.db"
        db_path.touch()
        with pytest.raises(ValueError, match="At least one cleanup criterion"):
            cleanup_datasets(db_path)

    def test_nonexistent_db_raises(self, tmp_path: Path) -> None:
        """Should raise if the DB file doesn't exist."""

        with pytest.raises(FileNotFoundError, match="Database file not found"):
            cleanup_datasets(tmp_path / "nonexistent.db", older_than_days=1)


# ---------------------------------------------------------------------------
# Integration tests - extract/export with split raw data
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_raw_data_experiment")
class TestExtractExportWithSplitRawData:
    """Verify that extract_runs_into_db and export_datasets_and_create_metadata_db
    work correctly when raw data is stored in separate per-dataset SQLite files."""

    @staticmethod
    def _make_dataset_with_data(
        n_rows: int = 10,
    ) -> tuple[DataSet, list[dict[str, float]]]:
        ds = new_data_set("test-split")
        x = ParamSpecBase("x", "numeric")
        y = ParamSpecBase("y", "numeric")
        idps = InterDependencies_(dependencies={y: (x,)})
        ds.set_interdependencies(idps)
        ds.mark_started()
        results = [{"x": float(i), "y": float(i**2)} for i in range(n_rows)]
        ds.add_results(results)
        ds.mark_completed()
        return ds, results

    @staticmethod
    def _close_ds(ds: DataSet) -> None:
        if ds._raw_data_conn is not None:
            ds._raw_data_conn.close()
        ds.conn.close()

    def test_extract_runs_into_db_with_split_data(self, tmp_path: Path) -> None:
        """extract_runs_into_db should copy data from external raw data file."""
        ds, results = self._make_dataset_with_data(n_rows=5)
        run_id = ds.run_id
        source_db = ds.path_to_db
        assert source_db is not None
        self._close_ds(ds)

        target_db = str(tmp_path / "target.db")
        extract_runs_into_db(source_db, target_db, run_id)

        # Verify data was copied to target DB
        target_conn = connect(target_db)
        target_ds = load_by_id(1, conn=target_conn)
        data = target_ds.get_parameter_data()
        np.testing.assert_array_almost_equal(
            data["y"]["y"], np.array([r["y"] for r in results])
        )
        assert target_ds.number_of_results == 5
        target_conn.close()

    def test_export_and_create_metadata_db_with_split_data(
        self, tmp_path: Path
    ) -> None:
        """export_datasets_and_create_metadata_db should work with split data."""
        ds, _results = self._make_dataset_with_data(n_rows=5)
        source_db = ds.path_to_db
        assert source_db is not None
        self._close_ds(ds)

        target_db = tmp_path / "metadata_only.db"
        export_path = tmp_path / "netcdf_exports"

        statuses = export_datasets_and_create_metadata_db(
            source_db, target_db, export_path=export_path
        )

        assert len(statuses) == 1
        # Should either be exported to NetCDF or copied as-is
        assert next(iter(statuses.values())) in ("exported", "copied_as_is")
