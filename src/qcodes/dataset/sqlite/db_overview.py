"""
This module provides a fast, lightweight overview of the runs stored in a
QCoDeS database.

:func:`get_db_overview` issues a single ``JOIN`` query against the ``runs`` and
``experiments`` tables to collect run metadata (experiment/sample names, time
stamps, record counts, guids, ...) without instantiating a ``DataSet`` object
per run. This avoids the expensive ``experiments()`` + ``data_sets()``
enumeration and makes it possible to list the contents of databases with many
thousands of runs almost instantly. It is primarily intended for tools that
need to display a table of runs (e.g. dataset browsers).
"""

from __future__ import annotations

import datetime
import json
import logging
import sqlite3
from contextlib import closing, nullcontext
from typing import TYPE_CHECKING

from typing_extensions import TypedDict

from qcodes.dataset.sqlite.database import conn_from_dbpath_or_conn
from qcodes.dataset.sqlite.query_helpers import is_column_in_table

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from qcodes.dataset.sqlite.connection import AtomicConnection

log = logging.getLogger(__name__)


class RunOverviewDict(TypedDict):
    """
    Lightweight overview of a single run.

    Contains only cheap-to-query metadata: no snapshot, no data and no full
    ``DataSet`` object. Extra ad-hoc metadata columns requested via the
    ``extra_columns`` argument of :func:`get_db_overview` are added to the
    dictionary under their column name in addition to the keys documented here.
    """

    #: ``run_id`` of the run.
    run_id: int
    #: Name of the experiment the run belongs to.
    experiment: str
    #: Sample name of the experiment the run belongs to.
    sample: str
    #: Name of the run.
    name: str
    #: Local date the run was started, formatted as ``YYYY-MM-DD`` (empty
    #: string if unknown).
    started_date: str
    #: Local time the run was started, formatted as ``HH:MM:SS`` (empty string
    #: if unknown).
    started_time: str
    #: Local date the run was completed, formatted as ``YYYY-MM-DD`` (empty
    #: string if the run has not completed).
    completed_date: str
    #: Local time the run was completed, formatted as ``HH:MM:SS`` (empty
    #: string if the run has not completed).
    completed_time: str
    #: Best-effort number of data points in the run, see :func:`get_db_overview`.
    records: int
    #: guid of the run.
    guid: str


def _format_timestamp(ts: float | None) -> tuple[str, str]:
    """
    Convert a unix timestamp into ``(date, time)`` strings in local time.

    Returns a pair of empty strings if the timestamp is missing or invalid.
    """
    if ts is None or ts == 0:
        return "", ""
    try:
        dt = datetime.datetime.fromtimestamp(ts)
    except (OSError, ValueError, OverflowError):
        return "", ""
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")


def _records_from_run_description(run_description_json: str | None) -> int:
    """
    Extract a data-point count from the ``shapes`` field of a run description.

    A QCoDeS run description may contain a ``shapes`` mapping from dependent
    parameter names to their shape tuples. The count returned here is the sum
    over all dependent parameters of the product of their shape dimensions.
    Returns ``0`` if the run description is missing, cannot be parsed or does
    not contain shape information.
    """
    if not run_description_json:
        return 0
    try:
        desc = json.loads(run_description_json)
    except (json.JSONDecodeError, TypeError):
        return 0
    shapes = desc.get("shapes") if isinstance(desc, dict) else None
    if not shapes:
        return 0
    total = 0
    for shape in shapes.values():
        if isinstance(shape, (list, tuple)) and len(shape) > 0:
            n = 1
            for dim in shape:
                n *= dim
            total += n
    return total


def get_db_overview(
    path_to_db: str | Path | None = None,
    *,
    conn: AtomicConnection | None = None,
    start_run_id: int = 0,
    extra_columns: Sequence[str] | None = None,
) -> dict[int, RunOverviewDict]:
    """
    Get a lightweight overview of the runs in a QCoDeS database.

    This uses a single SQL ``JOIN`` query on the ``runs`` and ``experiments``
    tables to fetch run metadata, avoiding the much more expensive
    ``experiments()`` + ``data_sets()`` enumeration that instantiates a
    ``DataSet`` object per run. It is therefore well suited for listing the
    contents of databases with many runs. The (potentially large) snapshot of
    each run is deliberately not read, as it would slow down building the
    overview significantly.

    The reported number of ``records`` is a best-effort estimate of the number
    of data points in a run and may be less precise than
    ``DataSet.number_of_results``:

    * For completed runs the shape information stored in the run description is
      preferred (it is the authoritative final count), falling back to the
      number of rows in the results table.
    * For runs that are still in progress the number of rows in the results
      table is preferred (it grows as data is added), falling back to the run
      description shapes.
    * If neither is available the count is reported as ``0`` (unknown).

    Only one of ``path_to_db`` and ``conn`` should be supplied. If a
    ``path_to_db`` is given the database is opened in read-only mode and the
    connection is closed again before returning.

    Args:
        path_to_db: Path to the database file. Opened read-only if given.
        conn: An existing connection to use instead of ``path_to_db``. It is
            left open by this function.
        start_run_id: Only return runs whose ``run_id`` is strictly greater
            than this value. Use ``0`` (the default) to get all runs, or pass
            the last known ``run_id`` to fetch only newly added runs.
        extra_columns: Names of additional ``runs``-table columns to include in
            each :class:`RunOverviewDict`. Columns that do not exist in the
            ``runs`` table of the given database are silently skipped. This is
            useful for reading ad-hoc metadata columns added via
            ``DataSet.add_metadata``.

    Returns:
        A dictionary mapping ``run_id`` to a :class:`RunOverviewDict`.

    """
    overview: dict[int, RunOverviewDict] = {}

    created_conn = conn is None
    connection = conn_from_dbpath_or_conn(
        conn=conn, path_to_db=path_to_db, read_only=True
    )
    manager = closing(connection) if created_conn else nullcontext(connection)

    with manager as c:
        valid_extra_columns = [
            col for col in (extra_columns or []) if is_column_in_table(c, "runs", col)
        ]
        extra_select = "".join(f", r.{col}" for col in valid_extra_columns)

        # ``run_description`` is queried to derive the record count for
        # completed runs; the (potentially large) snapshot is deliberately
        # excluded.
        query = f"""
            SELECT r.run_id, e.name, e.sample_name, r.name,
                   r.run_timestamp, r.completed_timestamp,
                   r.guid, r.result_table_name,
                   r.run_description{extra_select}
            FROM runs r
            JOIN experiments e ON r.exp_id = e.exp_id
            WHERE r.run_id > ?
            ORDER BY r.run_id
        """

        try:
            rows = c.execute(query, (start_run_id,)).fetchall()
        except sqlite3.Error as e:
            log.warning("Could not query database overview: %s", e)
            return overview

        # ``result_counter`` in the runs table is the run's ordinal within its
        # experiment, not a data-point count, so it is not usable here. For the
        # ``array`` paramtype a single INSERT can also contain many data points.
        # The real number of data points is therefore the row count of the
        # results table, queried separately.
        result_tables = {row[7] for row in rows if row[7]}
        row_counts: dict[str, int] = {}
        for table in result_tables:
            try:
                (count,) = c.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
            except sqlite3.Error:
                continue  # results table may not exist (yet)
            row_counts[table] = count

        n_fixed = 9  # number of columns selected before ``extra_columns``
        for row in rows:
            run_id = row[0]
            started_date, started_time = _format_timestamp(row[4])
            completed_date, completed_time = _format_timestamp(row[5])
            result_table = row[7] or ""
            is_completed = row[5] is not None and row[5] != 0

            # The record count is a best-effort data-point count. For completed
            # runs the run-description shapes are the authoritative final count;
            # for in-progress runs the live results-table row count is preferred
            # as it grows while data is added. ``0`` means "unknown".
            if is_completed:
                records = _records_from_run_description(row[8])
                if records == 0:
                    records = row_counts.get(result_table, 0)
            else:
                records = row_counts.get(result_table, 0)
                if records == 0:
                    records = _records_from_run_description(row[8])

            entry: RunOverviewDict = {
                "run_id": run_id,
                "experiment": row[1] or "",
                "sample": row[2] or "",
                "name": row[3] or "",
                "started_date": started_date,
                "started_time": started_time,
                "completed_date": completed_date,
                "completed_time": completed_time,
                "records": records,
                "guid": row[6] or "",
            }
            if valid_extra_columns:
                extra = {
                    col: row[n_fixed + i] for i, col in enumerate(valid_extra_columns)
                }
                # The keys of ``extra`` are only known at runtime (they are the
                # user-supplied ``extra_columns``), so they cannot be part of
                # the closed ``RunOverviewDict`` definition.
                entry.update(extra)  # type: ignore[typeddict-item]

            overview[run_id] = entry

    return overview
