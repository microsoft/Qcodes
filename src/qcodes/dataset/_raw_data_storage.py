"""
Module for managing per-dataset raw data SQLite files.

When the ``dataset.raw_data_to_separate_db`` config option is enabled,
measurement data (results tables) are written to individual SQLite files
- one per dataset - instead of the main QCoDeS database file.  All metadata
(runs, experiments, parameters) remains in the main database.

The per-dataset files are stored in the folder given by
``dataset.raw_data_path`` and are named ``<guid>.db``.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import qcodes
from qcodes.dataset.export_config import _expand_export_path
from qcodes.dataset.sqlite.connection import AtomicConnection, atomic
from qcodes.dataset.sqlite.database import (
    _adapt_array,
    _adapt_complex,
    _adapt_float,
    _convert_array,
    _convert_complex,
    _convert_numeric,
    connect,
)
from qcodes.dataset.sqlite.queries import (
    get_datasets_with_raw_data_path,
    remove_dataset_from_db,
)
from qcodes.dataset.sqlite.query_helpers import is_column_in_table
from qcodes.utils.types import complex_types, numpy_floats, numpy_ints

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qcodes.parameters import ParamSpecBase

log = logging.getLogger(__name__)

_RAW_DATA_CONFIG_SECTION = "dataset"
_RAW_DATA_ENABLED_KEY = "raw_data_to_separate_db"
_RAW_DATA_PATH_KEY = "raw_data_path"


def is_raw_data_storage_enabled() -> bool:
    """Return True if per-dataset raw data storage is enabled in config."""
    return bool(
        qcodes.config[_RAW_DATA_CONFIG_SECTION].get(_RAW_DATA_ENABLED_KEY, False)
    )


def get_raw_data_folder() -> Path:
    """Return the resolved folder path for raw data SQLite files.

    The path template from config is expanded the same way as the
    export path (``{db_location}`` is replaced with a folder derived
    from the main database path).
    """
    raw_path_template: str = qcodes.config[_RAW_DATA_CONFIG_SECTION].get(
        _RAW_DATA_PATH_KEY, "{db_location}"
    )
    return Path(_expand_export_path(raw_path_template)).expanduser().absolute()


def get_raw_data_db_path(guid: str, folder: Path | None = None) -> Path:
    """Return the full path for a dataset's raw data SQLite file.

    Args:
        guid: The GUID of the dataset.
        folder: Override folder.  If *None*, uses :func:`get_raw_data_folder`.

    """
    if folder is None:
        folder = get_raw_data_folder()
    return folder / f"{guid}.db"


def connect_to_raw_data_db(
    path: str | Path,
    *,
    read_only: bool = False,
) -> AtomicConnection:
    """Open (or create) a lightweight SQLite connection for raw data.

    Unlike the main QCoDeS :func:`~qcodes.dataset.sqlite.database.connect`,
    this does **not** create the full metadata schema (experiments, runs, ...).
    It only registers the numpy/sqlite type adapters that QCoDeS needs to
    round-trip array and numeric data.

    Args:
        path: Path to the raw-data SQLite file.
        read_only: Open the database in read-only mode.

    Returns:
        An :class:`AtomicConnection` to the raw-data database.

    """
    # Register adapters/converters (idempotent calls)
    sqlite3.register_adapter(np.ndarray, _adapt_array)
    sqlite3.register_converter("array", _convert_array)
    for numpy_int in numpy_ints:
        sqlite3.register_adapter(numpy_int, int)
    sqlite3.register_converter("numeric", _convert_numeric)
    for numpy_float in (float, *numpy_floats):
        sqlite3.register_adapter(numpy_float, _adapt_float)
    for complex_type in complex_types:
        sqlite3.register_adapter(complex_type, _adapt_complex)  # type: ignore[arg-type]
    sqlite3.register_converter("complex", _convert_complex)

    uri = f"file:{path!s}"
    if read_only:
        uri += "?mode=ro"

    conn = sqlite3.connect(
        uri,
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=True,
        uri=True,
        factory=AtomicConnection,
    )
    return conn


def create_raw_data_db(
    path: str | Path,
    table_name: str,
    paramspecs: Sequence[ParamSpecBase],
) -> AtomicConnection:
    """Create a per-dataset raw-data SQLite file with a results table.

    The file is created if it does not exist.  The parent directory is
    created if needed.

    Args:
        path: Full path for the new SQLite file.
        table_name: Name of the results table to create (matches the name
            in the main database).
        paramspecs: Parameter specifications describing the columns.

    Returns:
        An :class:`AtomicConnection` to the newly created database.

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = connect_to_raw_data_db(path)

    if paramspecs:
        columns = ",".join(f'"{p.name}" {p.type}' for p in paramspecs)
        sql = f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            id INTEGER PRIMARY KEY,
            {columns}
        );
        """
    else:
        sql = f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            id INTEGER PRIMARY KEY
        );
        """

    conn.execute(sql)
    conn.commit()

    log.info(
        "Created raw data database at %s with table %s",
        path,
        table_name,
    )
    return conn


def update_raw_data_paths(
    db_path: str | Path,
    new_raw_data_folder: str | Path,
) -> list[tuple[int, str, str]]:
    """Update raw data file paths in the main database after files have moved.

    Use this when per-dataset raw data files have been relocated to a new
    folder but the main database still references the old paths3.

    The function scans all runs that have a ``raw_data_db_path`` metadata
    entry, verifies that a file with the expected GUID-based name exists in
    *new_raw_data_folder*, and updates the stored path in the database.

    Args:
        db_path: Path to the main QCoDeS database file.
        new_raw_data_folder: The new folder where the per-dataset SQLite
            files now reside.

    Returns:
        A list of ``(run_id, old_path, new_path)`` tuples for every run
        whose path was updated.

    Raises:
        FileNotFoundError: If the main database file does not exist.
        FileNotFoundError: If *new_raw_data_folder* does not exist.

    """
    db_path = Path(db_path)
    new_raw_data_folder = Path(new_raw_data_folder)

    if not db_path.is_file():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    if not new_raw_data_folder.is_dir():
        raise FileNotFoundError(f"New raw data folder not found: {new_raw_data_folder}")

    conn = connect(str(db_path))

    if not is_column_in_table(conn, "runs", "raw_data_db_path"):
        log.info("No raw_data_db_path column found in %s; nothing to update.", db_path)
        conn.close()
        return []

    cursor = conn.execute(
        "SELECT run_id, raw_data_db_path FROM runs WHERE raw_data_db_path IS NOT NULL"
    )
    rows = cursor.fetchall()

    updated: list[tuple[int, str, str]] = []

    for run_id, old_path_str in rows:
        old_path = Path(old_path_str)
        # The per-dataset file name is always <guid>.db — preserved on move
        new_path = new_raw_data_folder / old_path.name

        if not new_path.is_file():
            log.warning(
                "Run %d: expected raw data file %s not found in new folder; skipping.",
                run_id,
                new_path,
            )
            continue

        if str(new_path) == old_path_str:
            continue  # already correct

        new_path_str = str(new_path)
        with atomic(conn) as aconn:
            aconn.execute(
                "UPDATE runs SET raw_data_db_path = ? WHERE run_id = ?",
                (new_path_str, run_id),
            )
        updated.append((run_id, old_path_str, new_path_str))
        log.debug(
            "Run %d: updated raw_data_db_path from %s to %s",
            run_id,
            old_path_str,
            new_path_str,
        )

    conn.close()
    log.info("Updated %d raw data paths in %s", len(updated), db_path)
    return updated


# ---------------------------------------------------------------------------
# Dataset management helpers
# ---------------------------------------------------------------------------


@dataclass
class DatasetInfo:
    """Summary information about a dataset in the main database."""

    run_id: int
    guid: str
    experiment_name: str
    sample_name: str
    run_timestamp: float | None
    completed_timestamp: float | None
    result_table_name: str
    raw_data_db_path: str | None
    raw_data_size_bytes: int | None


@dataclass
class PurgeResult:
    """Result of a purge_orphaned_datasets operation."""

    total_datasets_with_raw_data: int
    orphaned_datasets: list[DatasetInfo]
    removed_datasets: list[DatasetInfo]
    dry_run: bool
    errors: list[tuple[int, Exception]] = field(default_factory=list)


@dataclass
class CleanupResult:
    """Result of a cleanup_datasets operation."""

    total_datasets_scanned: int
    matching_datasets: list[DatasetInfo]
    removed_datasets: list[DatasetInfo]
    total_size_freed_bytes: int
    dry_run: bool
    errors: list[tuple[int, Exception]] = field(default_factory=list)


def _build_dataset_info_list(
    conn: AtomicConnection,
) -> list[DatasetInfo]:
    """Query datasets with raw data paths and enrich with file size info."""
    rows = get_datasets_with_raw_data_path(conn)

    datasets: list[DatasetInfo] = []
    for (
        run_id,
        guid,
        exp_name,
        sample_name,
        run_ts,
        completed_ts,
        table_name,
        raw_path,
    ) in rows:
        raw_size: int | None = None
        if raw_path and Path(raw_path).is_file():
            raw_size = Path(raw_path).stat().st_size

        datasets.append(
            DatasetInfo(
                run_id=run_id,
                guid=guid,
                experiment_name=exp_name,
                sample_name=sample_name,
                run_timestamp=run_ts,
                completed_timestamp=completed_ts,
                result_table_name=table_name,
                raw_data_db_path=raw_path,
                raw_data_size_bytes=raw_size,
            )
        )
    return datasets


def purge_orphaned_datasets(
    db_path: str | Path,
    *,
    dry_run: bool = True,
) -> PurgeResult:
    """Find and optionally remove dataset records whose raw data files are missing.

    When using split raw data storage, users may archive and delete
    individual per-dataset SQLite files.  This function identifies
    datasets in the main database that reference raw data files which
    no longer exist on disk, and optionally removes those dataset
    records from the main database.

    Args:
        db_path: Path to the main QCoDeS database file.
        dry_run: If *True* (default), only report which datasets would
            be removed without making any changes. Set to *False* to
            actually delete the orphaned dataset records.

    Returns:
        A :class:`PurgeResult` with the list of orphaned datasets and,
        if *dry_run* is False, the list of datasets that were removed.

    Raises:
        FileNotFoundError: If the main database file does not exist.

    """
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    with closing(connect(str(db_path))) as conn:
        all_datasets = _build_dataset_info_list(conn)

        # Orphaned = raw_data_size_bytes is None (file not found on disk)
        orphaned = [ds for ds in all_datasets if ds.raw_data_size_bytes is None]

        msg = (
            f"Found {len(all_datasets)} datasets with raw data references in {db_path}, "
            f"{len(orphaned)} orphaned (file missing)."
        )
        log.info(msg)
        print(msg)

        removed: list[DatasetInfo] = []
        errors: list[tuple[int, Exception]] = []

        if not dry_run and orphaned:
            for ds_info in orphaned:
                try:
                    remove_dataset_from_db(
                        conn, ds_info.run_id, ds_info.result_table_name
                    )
                    removed.append(ds_info)
                    log.debug(
                        "Removed orphaned dataset run_id=%d (guid=%s) from %s.",
                        ds_info.run_id,
                        ds_info.guid,
                        db_path,
                    )
                except Exception as exc:
                    log.error("Failed to remove run_id=%d: %s", ds_info.run_id, exc)
                    errors.append((ds_info.run_id, exc))

    result = PurgeResult(
        total_datasets_with_raw_data=len(all_datasets),
        orphaned_datasets=orphaned,
        removed_datasets=removed,
        dry_run=dry_run,
        errors=errors,
    )

    if dry_run:
        msg = f"Dry run: {len(orphaned)} orphaned datasets would be removed from {db_path}."
    else:
        msg = f"Removed {len(removed)} orphaned datasets from {db_path}."
    log.info(msg)
    print(msg)

    return result


def cleanup_datasets(
    db_path: str | Path,
    *,
    older_than_days: int | None = None,
    sample_name: str | None = None,
    larger_than_mb: float | None = None,
    dry_run: bool = True,
) -> CleanupResult:
    """Remove datasets and their raw data files matching given criteria.

    This function helps manage disk space by removing datasets that match
    one or more of the specified criteria. It removes both the raw data
    SQLite file on disk and the corresponding records in the main database.

    Criteria are combined with AND logic: a dataset must match **all**
    specified criteria to be selected for removal. Specify at least one
    criterion.

    Args:
        db_path: Path to the main QCoDeS database file.
        older_than_days: Remove datasets whose *completed_timestamp*
            (or *run_timestamp* if not completed) is older than this
            many days ago.
        sample_name: Remove datasets belonging to experiments with this
            exact sample name.
        larger_than_mb: Remove datasets whose raw data file is larger
            than this many megabytes.
        dry_run: If *True* (default), only report which datasets would
            be removed without making any changes. Set to *False* to
            actually delete datasets and their raw data files.

    Returns:
        A :class:`CleanupResult` with details of the operation.

    Raises:
        FileNotFoundError: If the main database file does not exist.
        ValueError: If no criteria are specified.

    """
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    if older_than_days is None and sample_name is None and larger_than_mb is None:
        raise ValueError("At least one cleanup criterion must be specified.")

    with closing(connect(str(db_path))) as conn:
        all_datasets = _build_dataset_info_list(conn)

        # Apply filters (AND logic)
        matching: list[DatasetInfo] = []
        cutoff_ts: float | None = None
        if older_than_days is not None:
            cutoff_dt = datetime.now(tz=UTC) - timedelta(days=older_than_days)
            cutoff_ts = cutoff_dt.timestamp()

        size_threshold_bytes: int | None = None
        if larger_than_mb is not None:
            size_threshold_bytes = int(larger_than_mb * 1024 * 1024)

        for ds in all_datasets:
            # Age filter
            if cutoff_ts is not None:
                ts = ds.completed_timestamp or ds.run_timestamp
                if ts is None or ts >= cutoff_ts:
                    continue

            # Sample name filter
            if sample_name is not None:
                if ds.sample_name != sample_name:
                    continue

            # Size filter
            if size_threshold_bytes is not None:
                if (
                    ds.raw_data_size_bytes is None
                    or ds.raw_data_size_bytes <= size_threshold_bytes
                ):
                    continue

            matching.append(ds)

        msg = (
            f"Found {len(all_datasets)} datasets with raw data in {db_path}, "
            f"{len(matching)} match cleanup criteria."
        )
        log.info(msg)
        print(msg)

        removed: list[DatasetInfo] = []
        errors: list[tuple[int, Exception]] = []
        total_freed: int = 0

        if not dry_run and matching:
            for ds_info in matching:
                try:
                    # Delete the raw data file from disk
                    if ds_info.raw_data_db_path:
                        raw_path = Path(ds_info.raw_data_db_path)
                        if raw_path.is_file():
                            file_size = raw_path.stat().st_size
                            raw_path.unlink()
                            total_freed += file_size
                            log.debug("Deleted raw data file: %s", raw_path)

                    # Remove dataset records from the main DB
                    remove_dataset_from_db(
                        conn, ds_info.run_id, ds_info.result_table_name
                    )
                    removed.append(ds_info)
                    log.debug(
                        "Removed dataset run_id=%d (guid=%s) from %s.",
                        ds_info.run_id,
                        ds_info.guid,
                        db_path,
                    )
                except Exception as exc:
                    log.error("Failed to remove run_id=%d: %s", ds_info.run_id, exc)
                    errors.append((ds_info.run_id, exc))

    result = CleanupResult(
        total_datasets_scanned=len(all_datasets),
        matching_datasets=matching,
        removed_datasets=removed,
        total_size_freed_bytes=total_freed,
        dry_run=dry_run,
        errors=errors,
    )

    if dry_run:
        total_size = sum(
            ds.raw_data_size_bytes for ds in matching if ds.raw_data_size_bytes
        )
        msg = f"Dry run: {len(matching)} datasets ({total_size} bytes) would be removed from {db_path}."
    else:
        msg = f"Removed {len(removed)} datasets from {db_path}, freed {total_freed} bytes."
    log.info(msg)
    print(msg)

    return result
