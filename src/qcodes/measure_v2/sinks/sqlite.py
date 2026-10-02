"""Default storage sink: writes the event stream to a SQLite dataset.

Wraps the existing :py:class:`~qcodes.dataset.measurements.Measurement` /
:py:class:`~qcodes.dataset.measurements.DataSaver` plumbing. The bulk of
the work is the schema-registration lifecycle:

- On ``RunStarted``: build a ``Measurement``, register each setpoint and
  measured parameter (with setpoint dependencies for measured params),
  enter the ``Runner`` context manager (which is where the station
  snapshot is taken and the ``DataSaver`` is created).
- On ``RowEmitted``: forward the row via ``DataSaver.add_result``. All
  array fan-out, type coercion, and write-batching logic stays in
  ``DataSaver``.
- On ``RunStopped``: exit the ``Runner`` context manager (success or
  error path); the dataset's completion timestamp is set.

**Threading.** SQLite connections are bound to the thread that created
them (``check_same_thread=True``). Since this sink runs on the engine's
publisher thread, it opens its own connection lazily on first use and
creates experiments against it. Datasets opened by this sink can be
read back from other threads via their in-memory caches, or re-loaded
via ``load_by_id`` on the consuming thread.

The sink also implements :py:meth:`dataset_for` so the engine's publisher
can resolve ``RunHandle.dataset`` to the created dataset right after
``RunStarted`` is processed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import qcodes as qc
from qcodes.dataset.experiment_container import load_or_create_experiment
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.sqlite.database import connect
from qcodes.measure_v2.events import RowEmitted, RunStarted, RunStopped

if TYPE_CHECKING:
    from uuid import UUID

    from qcodes.dataset.data_set_protocol import DataSetProtocol
    from qcodes.dataset.measurements import DataSaver
    from qcodes.dataset.sqlite.connection import AtomicConnection
    from qcodes.measure_v2.events import Event


_LOG = logging.getLogger(__name__)


class SqliteSink:
    """Critical sink that writes the event stream to a QCoDeS SQLite dataset.

    Tracer-bullet scope: handles 1D scans of scalar-valued parameters.
    Array-valued parameters work transparently through ``DataSaver`` but
    are not yet covered by tests.

    Args:
        experiment_name: Name of the experiment all runs from this sink
            belong to. The experiment is loaded or created on the publisher
            thread on first ``RunStarted``.
        sample_name: Sample name for the experiment.

    """

    critical: bool = True

    def __init__(
        self,
        *,
        experiment_name: str = "measure_v2",
        sample_name: str = "default",
    ) -> None:
        self._experiment_name = experiment_name
        self._sample_name = sample_name
        self._conn: AtomicConnection | None = None
        self._runners: dict[UUID, Any] = {}
        self._savers: dict[UUID, DataSaver] = {}
        self._datasets: dict[UUID, DataSetProtocol] = {}

    def __call__(self, event: Event) -> None:
        if isinstance(event, RunStarted):
            self._open(event)
        elif isinstance(event, RowEmitted):
            self._add_row(event)
        elif isinstance(event, RunStopped):
            self._close(event)

    def dataset_for(self, run_id: UUID) -> DataSetProtocol | None:
        """Return the dataset associated with ``run_id``, if one was opened."""
        return self._datasets.get(run_id)

    # ------------------------------------------------------------------
    # Internal lifecycle
    # ------------------------------------------------------------------

    def _ensure_connection(self) -> AtomicConnection:
        """Open a SQLite connection on the current (publisher) thread.

        SQLite connections are thread-bound, so this connection is only
        usable from the same thread that calls this method first.
        """
        if self._conn is None:
            db_path = qc.config["core"]["db_location"]
            self._conn = connect(db_path)
        return self._conn

    def _open(self, event: RunStarted) -> None:
        descriptor = event.descriptor
        conn = self._ensure_connection()

        # Load or create the experiment on the publisher thread so its
        # connection is usable here. Note: if event.exp was provided from
        # the main thread, we ignore its identity and use ours; this is a
        # tracer limitation.
        exp_name = self._experiment_name
        sample_name = self._sample_name
        if event.exp is not None:
            exp_name = event.exp.name
            sample_name = event.exp.sample_name
        exp = load_or_create_experiment(
            experiment_name=exp_name,
            sample_name=sample_name,
            conn=conn,
        )

        meas = Measurement(name=event.name or "results", exp=exp)
        if event.write_period is not None:
            meas.write_period = event.write_period

        # Register setpoints first; measured params reference them.
        for p in descriptor.setpoints:
            meas.register_parameter(p)
        for p in descriptor.measured:
            meas.register_parameter(p, setpoints=descriptor.setpoints)

        # meas.run() returns a Runner (the actual context manager).
        # Runner.__enter__ returns the DataSaver; Runner.__exit__ commits.
        runner = meas.run()
        saver: DataSaver = runner.__enter__()
        self._runners[event.run_id] = runner
        self._savers[event.run_id] = saver
        self._datasets[event.run_id] = saver.dataset

    def _add_row(self, event: RowEmitted) -> None:
        saver = self._savers.get(event.run_id)
        if saver is None:
            _LOG.warning("RowEmitted for unknown run %s; dropping.", event.run_id)
            return
        # snapshot is {ParameterBase: value} → tuples for add_result.
        saver.add_result(*event.snapshot.items())

    def _close(self, event: RunStopped) -> None:
        runner = self._runners.pop(event.run_id, None)
        self._savers.pop(event.run_id, None)
        # Keep the dataset reference around so handle.dataset.result() stays valid.
        if runner is None:
            return
        exc_type = type(event.error) if event.error is not None else None
        try:
            runner.__exit__(exc_type, event.error, None)
        except BaseException:
            _LOG.exception("Error closing dataset for run %s", event.run_id)
            raise
