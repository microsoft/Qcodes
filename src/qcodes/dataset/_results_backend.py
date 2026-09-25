"""Pluggable storage backends for a :class:`~qcodes.dataset.data_set.DataSet`'s
results (raw measurement data).

A :class:`ResultsBackend` encapsulates *where and how* the results-table data of
a dataset is stored, so that :class:`~qcodes.dataset.data_set.DataSet` itself
stays free of storage-specific conditionals. The concrete backend is chosen
inside ``DataSet.__init__`` (based on config for new runs, and on the run's
recorded state for existing runs), which keeps auto-detection working for any
``DataSet(run_id=...)`` construction.

Two backends are provided:

- :class:`MainDatabaseResultsBackend` (the default) keeps the results table in
  the main QCoDeS database.
- :class:`SeparateSqliteFileResultsBackend` writes results to an individual
  per-dataset SQLite file while all metadata remains in the main database.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from qcodes.dataset._raw_data_storage import (
    connect_to_raw_data_db,
    create_raw_data_db,
    get_raw_data_db_path,
    is_raw_data_storage_enabled,
)
from qcodes.dataset.sqlite.connection import atomic, atomic_transaction
from qcodes.dataset.sqlite.queries import (
    _check_if_table_found,
    _create_run_table,
    get_parameter_data,
    get_raw_data_db_path_for_run,
    get_shaped_parameter_data_for_one_paramtree,
    set_raw_data_db_path_for_run,
)
from qcodes.dataset.sqlite.query_helpers import (
    insert_column,
    insert_many_values,
    length,
    one,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from qcodes.dataset.data_set import DataSet
    from qcodes.dataset.data_set_protocol import ParameterData
    from qcodes.dataset.sqlite.connection import AtomicConnection
    from qcodes.dataset.sqlite.query_helpers import VALUE

log = logging.getLogger(__name__)


class ResultsBackend:
    """Strategy describing where a :class:`.DataSet`'s results table lives.

    The base class keeps results in the main QCoDeS database; subclasses store
    them elsewhere (e.g. a per-dataset SQLite file) while metadata stays in the
    main database. A backend is owned by exactly one dataset.

    Subclasses customise only what differs: :attr:`results_conn`, the
    ``setup_on_*`` / :meth:`create_results_table` / :meth:`close` lifecycle,
    :attr:`results_db_path`, and - if the read path differs -
    :meth:`read_parameter_data`. The results-table operations below are written
    against :attr:`results_conn`, so both backends reuse them unchanged (which
    is why :class:`MainDatabaseResultsBackend` needs no code of its own).
    """

    def __init__(self, dataset: DataSet) -> None:
        self._dataset = dataset

    @property
    def results_conn(self) -> AtomicConnection:
        """The connection on which the results table lives."""
        return self._dataset.conn

    # -- lifecycle -------------------------------------------------------

    def setup_on_load(self, *, read_only: bool) -> None:
        """Set up the backend for an existing run being loaded. No-op here."""

    def setup_on_new_run(self) -> None:
        """Record backend bookkeeping for a newly created run.

        The default creates the (still empty) results table in the main
        database so that the run is recognisable as a :class:`.DataSet` (rather
        than a ``DataSetInMem``) even before it is started. Subclasses that keep
        no results table in the main database override this.
        """
        _create_run_table(self.results_conn, self._dataset.table_name)

    def create_results_table(self) -> None:
        """Populate this backend's results table when the run is started.

        The default adds the parameter columns to the table created in
        :meth:`setup_on_new_run`; subclasses that store results elsewhere
        override this (e.g. to create a per-dataset file).
        """
        ds = self._dataset
        # Add all columns in a single transaction to avoid one commit (fsync)
        # per parameter.
        with atomic(self.results_conn) as conn:
            for spec in ds._rundescriber.interdeps.paramspecs:
                insert_column(conn, ds.table_name, spec.name, spec.type)

    def close(self) -> None:
        """Close any resources owned by the backend. No-op here."""

    # -- results-table operations ----------------------------------------

    def results_table_exists(self) -> bool:
        """Whether the physical results table currently exists."""
        return _check_if_table_found(self.results_conn, self._dataset.table_name)

    def number_of_results(self) -> int:
        """Number of rows in the results table (``0`` if it does not exist)."""
        if not self.results_table_exists():
            return 0
        sql = f'SELECT COUNT(*) FROM "{self._dataset.table_name}"'
        cursor = atomic_transaction(self.results_conn, sql)
        return one(cursor, "COUNT(*)")

    def results_length(self) -> int:
        """Length of the results table (max row id, ``0`` if it does not
        exist)."""
        if not self.results_table_exists():
            return 0
        return length(self.results_conn, self._dataset.table_name)

    def insert_results(
        self, param_names: Sequence[str], values: Sequence[Sequence[VALUE]]
    ) -> None:
        """Insert rows of results directly (used for non-background writes)."""
        insert_many_values(
            self.results_conn, self._dataset.table_name, list(param_names), values
        )

    @property
    def results_db_path(self) -> str | None:
        """Path of the separate file holding the results table, or ``None`` when
        results live in the main database.

        Lets the ``DataSet`` route background writes to the right file without
        the backend knowing about the background-writer's queue-item format.
        """
        return None

    def read_parameter_data(
        self,
        valid_param_names: list[str],
        start: int | None,
        end: int | None,
        callback: Callable[[float], None] | None,
    ) -> ParameterData:
        """Read parameter data for the given parameters from the backend."""
        return get_parameter_data(
            self.results_conn,
            self._dataset.table_name,
            valid_param_names,
            start,
            end,
            callback,
        )


class MainDatabaseResultsBackend(ResultsBackend):
    """Store results in the main QCoDeS database (the default).

    The reference behaviour: it adds nothing, as the base class's generic
    implementation is exactly what a main-database dataset needs.
    """


class SeparateSqliteFileResultsBackend(ResultsBackend):
    """Store results in an individual per-dataset SQLite file.

    All metadata stays in the main database; the path to the per-dataset file
    is recorded in the ``raw_data_db_path`` column of the ``runs`` table (not in
    the user-facing metadata), which is also how such runs are recognised when
    loading.
    """

    def __init__(self, dataset: DataSet) -> None:
        super().__init__(dataset)
        self._conn: AtomicConnection | None = None
        self._db_path: str | None = None

    @property
    def results_conn(self) -> AtomicConnection:
        # Before the dataset is started the per-dataset file does not exist yet,
        # so fall back to the main connection. The main connection has no
        # results table either, which the operation methods handle via
        # ``results_table_exists``.
        if self._conn is not None:
            return self._conn
        return self._dataset.conn

    def setup_on_load(self, *, read_only: bool) -> None:
        ds = self._dataset
        # The path is stored in a dedicated runs-table column, not in the
        # user-facing metadata.
        raw_db_path = get_raw_data_db_path_for_run(ds.conn, ds.run_id)
        self._db_path = raw_db_path
        if raw_db_path is None:
            return
        if Path(raw_db_path).is_file():
            self._conn = connect_to_raw_data_db(raw_db_path, read_only=read_only)
        elif ds._started:
            raise FileNotFoundError(
                f"Raw data file for dataset {ds.guid} not found at "
                f"'{raw_db_path}'. The per-dataset SQLite file may "
                f"have been moved or deleted."
            )
        # else: the dataset was never started, so the raw data file has not been
        # created yet - there is simply no data to connect to.

    def setup_on_new_run(self) -> None:
        ds = self._dataset
        # Record the raw-data backend location up front. This marks the run as a
        # split-storage dataset (so it can be told apart from a DataSetInMem run,
        # which also has no results table) even before it is started and before
        # the raw data file is created.
        raw_path_str = str(get_raw_data_db_path(ds.guid))
        self._db_path = raw_path_str
        with atomic(ds.conn) as aconn:
            set_raw_data_db_path_for_run(aconn, ds.run_id, raw_path_str)

    def create_results_table(self) -> None:
        ds = self._dataset
        # The raw-data path was already recorded at creation time; reuse it so
        # both locations stay in sync.
        raw_path_str = self._db_path or str(get_raw_data_db_path(ds.guid))
        self._conn = create_raw_data_db(
            Path(raw_path_str),
            ds.table_name,
            ds._rundescriber.interdeps.paramspecs,
        )
        if self._db_path != raw_path_str:
            self._db_path = raw_path_str
            with atomic(ds.conn) as aconn:
                set_raw_data_db_path_for_run(aconn, ds.run_id, raw_path_str)

    @property
    def results_db_path(self) -> str | None:
        return self._db_path

    def read_parameter_data(
        self,
        valid_param_names: list[str],
        start: int | None,
        end: int | None,
        callback: Callable[[float], None] | None,
    ) -> ParameterData:
        ds = self._dataset
        if self._conn is None:
            # Not started yet / no separate file: defer to the default reader,
            # which will find no results table and return empty data.
            return super().read_parameter_data(valid_param_names, start, end, callback)
        # When results live in a separate DB, bypass get_parameter_data (which
        # looks up the rundescriber from the main DB) and call the lower-level
        # function directly with the rundescriber we already hold.
        output: ParameterData = {}
        for param_name in valid_param_names:
            output[param_name] = get_shaped_parameter_data_for_one_paramtree(
                self._conn,
                ds.table_name,
                ds._rundescriber,
                param_name,
                start,
                end,
                callback,
            )
        return output

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()


def select_results_backend_for_new_run(dataset: DataSet) -> ResultsBackend:
    """Choose the results backend for a newly created dataset based on config."""
    if is_raw_data_storage_enabled():
        return SeparateSqliteFileResultsBackend(dataset)
    return MainDatabaseResultsBackend(dataset)


def select_results_backend_for_existing_run(
    dataset: DataSet, conn: AtomicConnection, run_id: int
) -> ResultsBackend:
    """Choose the results backend for an existing run.

    A run whose ``raw_data_db_path`` column is set stores its results in a
    separate SQLite file; otherwise the results live in the main database.
    """
    if get_raw_data_db_path_for_run(conn, run_id) is not None:
        return SeparateSqliteFileResultsBackend(dataset)
    return MainDatabaseResultsBackend(dataset)
