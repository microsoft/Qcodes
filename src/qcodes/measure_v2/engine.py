"""The measurement engine.

A :py:class:`MeasurementEngine` owns a dedicated worker thread that
iterates plan generators and dispatches their messages, and a publisher
thread that fans out events to sinks. The user-facing API is small:

- ``engine.submit(plan)`` returns a :py:class:`RunHandle`.
- ``handle.wait()`` blocks until the run completes.
- ``handle.cancel()`` requests graceful cancellation (plan's ``finally``
  runs).

Tracer-bullet scope (current):

- One run at a time (concurrent ``submit`` raises ``RuntimeError``).
- Sequential reads (no thread pool grouping by underlying instrument).
- Engine identifies parameters by ``ParameterBase`` object identity;
  ``register_name`` uniqueness is checked at submit (via the descriptor
  in ``OpenRun``).
- ``handle.dataset`` is a :py:class:`Future` but resolves to ``None`` —
  populated by the SQLite sink in a later layer.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

from qcodes.measure_v2.events import (
    Descriptor,
    RowEmitted,
    RunResult,
    RunStarted,
    RunStopped,
)
from qcodes.measure_v2.exceptions import CancelRequested, PlanError
from qcodes.measure_v2.messages import (
    CloseRun,
    Emit,
    OpenRun,
    Read,
    Set,
    Sleep,
)
from qcodes.measure_v2.sinks import is_critical
from qcodes.measure_v2.sinks.memory import MemorySink

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from qcodes.dataset.data_set_protocol import DataSetProtocol
    from qcodes.measure_v2.events import Event, RunStopReason
    from qcodes.measure_v2.messages import Msg
    from qcodes.measure_v2.sinks import DataSink
    from qcodes.parameters import ParameterBase


_LOG = logging.getLogger(__name__)

# Sentinels used to signal thread shutdown via the internal queues.
_SUBMIT_SHUTDOWN = object()
_PUBLISH_SHUTDOWN = object()


# ----------------------------------------------------------------------------
# Public types
# ----------------------------------------------------------------------------


RunStatus = Literal["queued", "running", "cancelling", "done", "error"]


class RunHandle:
    """User-facing handle for a submitted run.

    The handle is returned immediately from :py:meth:`MeasurementEngine.submit`
    and exposes futures for the dataset (resolved when a SQLite sink opens it)
    and the run result (resolved when the run completes).
    """

    def __init__(
        self,
        run_id: UUID,
        cancel_event: threading.Event,
        future: Future[RunResult],
        dataset_future: Future[DataSetProtocol | None],
    ) -> None:
        self.run_id = run_id
        self._cancel_event = cancel_event
        self._cancel_reason_box: list[str] = []
        self.future = future
        self.dataset: Future[DataSetProtocol | None] = dataset_future

    def cancel(self, reason: str = "user") -> None:
        """Request graceful cancellation.

        Sets the cancel flag; the engine throws
        :py:class:`CancelRequested` into the plan at the next yield point.
        The plan's ``try/finally`` cleanup runs before the run is reported
        stopped.
        """
        if not self._cancel_reason_box:
            self._cancel_reason_box.append(reason)
        self._cancel_event.set()

    def wait(self, timeout: float | None = None) -> RunResult:
        """Block until the run completes; return the :py:class:`RunResult`."""
        return self.future.result(timeout=timeout)

    @property
    def status(self) -> RunStatus:
        if self.future.done():
            return "error" if self.future.exception() is not None else "done"
        if self._cancel_event.is_set():
            return "cancelling"
        return "running"


# ----------------------------------------------------------------------------
# Internal submission state
# ----------------------------------------------------------------------------


@dataclass
class _Submission:
    run_id: UUID
    plan: Generator[Msg, Any, None]
    cancel_event: threading.Event
    cancel_reason_box: list[str]
    future: Future[RunResult]
    dataset_future: Future[DataSetProtocol | None]
    descriptor: Descriptor | None = None
    state: dict[ParameterBase, Any] = field(default_factory=dict)
    n_rows: int = 0
    _next_seq: int = 0
    started_at: float = 0.0

    def next_seq(self) -> int:
        s = self._next_seq
        self._next_seq += 1
        return s


# ----------------------------------------------------------------------------
# The engine
# ----------------------------------------------------------------------------


class MeasurementEngine:
    """Drives plans on a dedicated worker thread, publishes events to sinks.

    Args:
        name: Engine name (for logging).
        sinks: Sequence of sinks the engine publishes events to. Defaults
            to a single :py:class:`MemorySink` — useful for tests; real
            usage typically passes a :py:class:`SqliteSink`.
        queue_maxsize: Bounded capacity of the publish queue. When full,
            the engine thread blocks on event publication (back-pressure).

    """

    def __init__(
        self,
        *,
        name: str = "default",
        sinks: Sequence[DataSink] | None = None,
        queue_maxsize: int = 1024,
    ) -> None:
        self.name = name
        # Sinks: critical first (so SQLite-style sinks process events before
        # observers); within each criticality bucket, preserve registration order.
        provided = list(sinks) if sinks is not None else [MemorySink()]
        self._sinks: list[DataSink] = sorted(
            provided, key=lambda s: 0 if is_critical(s) else 1
        )

        self._submit_queue: queue.Queue[_Submission | object] = queue.Queue()
        self._publish_queue: queue.Queue[Event | object] = queue.Queue(
            maxsize=queue_maxsize
        )

        # Submissions live here until the publisher has finalized them (i.e.,
        # delivered RunStopped to all sinks). _current_sub is only cleared by
        # the publisher, so handle.wait() returning is the signal that data
        # has been durably written to all sinks.
        self._current_sub: _Submission | None = None
        self._subs_by_id: dict[UUID, _Submission] = {}
        self._current_lock = threading.Lock()
        self._shutdown = threading.Event()

        self._engine_thread = threading.Thread(
            target=self._engine_loop,
            name=f"measure_v2-engine-{name}",
            daemon=True,
        )
        self._publisher_thread = threading.Thread(
            target=self._publisher_loop,
            name=f"measure_v2-publisher-{name}",
            daemon=True,
        )
        self._engine_thread.start()
        self._publisher_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        plan: Generator[Msg, Any, None],
        *,
        name: str = "",
    ) -> RunHandle:
        """Submit a plan for execution.

        Returns immediately with a :py:class:`RunHandle`. The plan is
        executed on the engine's worker thread. In tracer scope, concurrent
        submission raises ``RuntimeError`` (no queue yet). The ``name``
        kwarg is accepted for API symmetry with v1 but currently ignored.

        Raises:
            RuntimeError: If a run is already in progress.

        """
        del name  # accepted for API symmetry; v1 will use it
        if self._shutdown.is_set():
            raise RuntimeError(f"Engine {self.name!r} is shut down")

        with self._current_lock:
            if self._current_sub is not None:
                raise RuntimeError(
                    f"Engine {self.name!r} is already running a plan. "
                    "Tracer-scope engines run one plan at a time; queueing is deferred to v1."
                )

        run_id = uuid4()
        cancel_event = threading.Event()
        future: Future[RunResult] = Future()
        dataset_future: Future[DataSetProtocol | None] = Future()
        sub = _Submission(
            run_id=run_id,
            plan=plan,
            cancel_event=cancel_event,
            cancel_reason_box=[],
            future=future,
            dataset_future=dataset_future,
        )
        handle = RunHandle(
            run_id=run_id,
            cancel_event=cancel_event,
            future=future,
            dataset_future=dataset_future,
        )
        sub.cancel_reason_box = handle._cancel_reason_box  # share reason box
        with self._current_lock:
            self._current_sub = sub
            self._subs_by_id[run_id] = sub
        self._submit_queue.put(sub)
        return handle

    def shutdown(self, *, wait: bool = True, timeout: float = 30.0) -> None:
        """Stop the engine.

        Cancels any in-flight run and waits up to ``timeout`` seconds for
        the engine and publisher threads to exit. After the deadline:
        logs and returns; cleanup may continue running in the background.
        """
        if self._shutdown.is_set():
            return
        self._shutdown.set()

        # Cancel any in-flight run.
        with self._current_lock:
            sub = self._current_sub
        if sub is not None:
            if not sub.cancel_reason_box:
                sub.cancel_reason_box.append("engine_shutdown")
            sub.cancel_event.set()

        # Signal both threads to exit after their current work.
        self._submit_queue.put(_SUBMIT_SHUTDOWN)

        if wait:
            self._engine_thread.join(timeout=timeout)
            if self._engine_thread.is_alive():
                _LOG.warning(
                    "Engine thread did not exit within %.1fs; abandoning.", timeout
                )
            # Publisher thread is signaled by the engine thread after it exits.
            remaining = max(
                0.1, timeout - (timeout if self._engine_thread.is_alive() else 0)
            )
            self._publisher_thread.join(timeout=remaining)
            if self._publisher_thread.is_alive():
                _LOG.warning(
                    "Publisher thread did not exit within %.1fs; abandoning.",
                    timeout,
                )

    # ------------------------------------------------------------------
    # Engine thread
    # ------------------------------------------------------------------

    def _engine_loop(self) -> None:
        try:
            while True:
                item = self._submit_queue.get()
                if item is _SUBMIT_SHUTDOWN:
                    break
                sub = item  # type: ignore[assignment]
                assert isinstance(sub, _Submission)
                try:
                    self._run_one_plan(sub)
                except BaseException:
                    _LOG.exception("Unhandled error driving plan %s", sub.run_id)
        finally:
            self._publish_queue.put(_PUBLISH_SHUTDOWN)

    def _run_one_plan(self, sub: _Submission) -> None:
        it = iter(sub.plan)
        in_cleanup = False
        send_value: Any = None
        reason: RunStopReason = "completed"
        error: BaseException | None = None
        sub.started_at = time.time()
        cancel_request_time: float | None = None

        # ``pending_msg`` lets us inject a message (e.g., the first message
        # returned by a cleanup throw) into the next dispatch iteration
        # without re-entering the get-next-message phase.
        pending_msg: Msg | None = None

        try:
            while True:
                # Phase 1: obtain the next message.
                if pending_msg is not None:
                    msg = pending_msg
                    pending_msg = None
                else:
                    cancel_now = sub.cancel_event.is_set() and not in_cleanup
                    if cancel_now:
                        in_cleanup = True
                        reason = "cancelled"
                        cancel_request_time = time.time()
                        try:
                            msg = it.throw(CancelRequested(self._reason(sub)))
                        except (StopIteration, CancelRequested):
                            break
                    else:
                        try:
                            msg = it.send(send_value)
                        except StopIteration:
                            break
                        except CancelRequested:
                            if reason == "completed":
                                reason = "interrupted"
                            break

                    # Re-check cancel between send and dispatch (critique fix).
                    if sub.cancel_event.is_set() and not in_cleanup:
                        in_cleanup = True
                        reason = "cancelled"
                        cancel_request_time = time.time()
                        try:
                            msg = it.throw(CancelRequested(self._reason(sub)))
                        except (StopIteration, CancelRequested):
                            break

                # Phase 2: dispatch the message.
                try:
                    send_value = self._dispatch(msg, sub)
                except BaseException as exc:
                    if not in_cleanup:
                        error = exc
                        reason = "error"
                        in_cleanup = True
                    # Inject the error into the plan so its finally runs.
                    try:
                        pending_msg = it.throw(type(exc), exc)
                        send_value = None
                    except (StopIteration, CancelRequested):
                        break
        finally:
            # Engine guarantees the generator is closed.
            try:
                it.close()
            except BaseException:
                _LOG.exception("Error closing plan %s", sub.run_id)
            stopped_at = time.time()
            cancel_latency = (
                stopped_at - cancel_request_time
                if cancel_request_time is not None
                else None
            )
            self._publish_run_stopped(
                sub, reason, error, sub.started_at, stopped_at, cancel_latency
            )
            self._complete_submission(sub, reason, error, stopped_at, cancel_latency)

    @staticmethod
    def _reason(sub: _Submission) -> str:
        return sub.cancel_reason_box[0] if sub.cancel_reason_box else "cancel"

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, msg: Msg, sub: _Submission) -> Any:
        # Note: cancellable_sleep is the only place we honor cancellation
        # within a message dispatch. All other dispatches are atomic.
        match msg:
            case Set(param, value):
                param.set(value)
                sub.state[param] = value
                return None
            case Read(params):
                results: dict[ParameterBase, Any] = {}
                # Sequential reads — tracer scope. v1 will group by underlying_instrument.
                for p in params:
                    results[p] = p.get()
                sub.state.update(results)
                return results
            case Sleep(seconds):
                self._cancellable_sleep(seconds, sub)
                return None
            case Emit(overrides):
                self._handle_emit(sub, overrides)
                return None
            case OpenRun() as o:
                self._handle_open_run(sub, o)
                return None
            case CloseRun():
                # Sink receives RunStopped via _publish_run_stopped at end of run;
                # CloseRun on its own is a no-op for the engine.
                return None
            case _:
                raise TypeError(f"Unknown plan message: {msg!r}")

    def _handle_open_run(self, sub: _Submission, msg: OpenRun) -> None:
        # Build the descriptor and validate.
        descriptor = Descriptor(
            setpoints=msg.setpoint_params,
            measured=msg.measured_params,
            shapes=msg.shapes,
        )
        self._check_no_duplicate_register_names(descriptor)
        sub.descriptor = descriptor
        self._publish(
            RunStarted(
                run_id=sub.run_id,
                name=msg.name,
                descriptor=descriptor,
                exp=msg.exp,
                write_period=msg.write_period,
                started_at=sub.started_at,
            )
        )

    def _handle_emit(
        self, sub: _Submission, overrides: dict[ParameterBase, Any] | Any
    ) -> None:
        descriptor = sub.descriptor
        if descriptor is None:
            raise PlanError("Emit yielded before OpenRun")

        declared: set[ParameterBase] = set(descriptor.setpoints) | set(
            descriptor.measured
        )

        # Validate overrides reference only declared params.
        for p in overrides:
            if p not in declared:
                raise PlanError(
                    f"Emit overrides parameter {p.register_name!r} "
                    "not declared in the run descriptor"
                )

        # Build the snapshot row.
        snapshot: dict[ParameterBase, Any] = {}
        for p in descriptor.setpoints:
            if p in overrides:
                snapshot[p] = overrides[p]
            elif p in sub.state:
                snapshot[p] = sub.state[p]
            else:
                raise PlanError(
                    f"Emit before setpoint {p.register_name!r} was set; "
                    "engine state has no value for it."
                )
        for p in descriptor.measured:
            if p in overrides:
                snapshot[p] = overrides[p]
            elif p in sub.state:
                snapshot[p] = sub.state[p]
            else:
                raise PlanError(
                    f"Emit before measured parameter {p.register_name!r} "
                    "was read; engine state has no value for it."
                )

        seq = sub.next_seq()
        sub.n_rows += 1
        self._publish(RowEmitted(run_id=sub.run_id, snapshot=snapshot, seq=seq))

    def _cancellable_sleep(self, seconds: float, sub: _Submission) -> None:
        if seconds <= 0:
            return
        deadline = time.monotonic() + seconds
        # Chunked checks: ~100ms granularity, so cancel during a long sleep
        # is bounded by that chunk.
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            if sub.cancel_event.is_set():
                return  # Let the main loop pick up the cancel on next iteration.
            time.sleep(min(remaining, 0.1))

    @staticmethod
    def _check_no_duplicate_register_names(descriptor: Descriptor) -> None:
        seen: dict[str, ParameterBase] = {}
        for p in (*descriptor.setpoints, *descriptor.measured):
            name = p.register_name
            if name in seen and seen[name] is not p:
                raise PlanError(
                    f"Two distinct parameters share register_name {name!r}: "
                    f"{seen[name]} and {p}."
                )
            seen[name] = p

    # ------------------------------------------------------------------
    # Publish & sink lifecycle
    # ------------------------------------------------------------------

    def _publish(self, event: Event) -> None:
        # Blocks if the publish queue is full (back-pressure). Acceptable
        # default for DAQ — better than dropping data.
        self._publish_queue.put(event)

    def _publish_run_stopped(
        self,
        sub: _Submission,
        reason: RunStopReason,
        error: BaseException | None,
        started_at: float,
        stopped_at: float,
        cancel_latency: float | None,
    ) -> None:
        # Always emit RunStopped, even if OpenRun was never published (e.g.,
        # the plan errored before yielding OpenRun). Sinks must handle this
        # gracefully.
        self._publish_queue.put(
            RunStopped(
                run_id=sub.run_id,
                reason=reason,
                error=error,
                started_at=started_at,
                stopped_at=stopped_at,
                cancel_latency=cancel_latency,
                n_rows_emitted=sub.n_rows,
            )
        )

    def _complete_submission(
        self,
        sub: _Submission,
        reason: RunStopReason,
        error: BaseException | None,
        stopped_at: float,
        cancel_latency: float | None,
    ) -> None:
        # No-op: the publisher thread now owns final-state transition so
        # that handle.wait() returning is a guarantee that all sinks have
        # processed RunStopped. See _finalize_after_run_stopped.
        del sub, reason, error, stopped_at, cancel_latency

    def _finalize_after_run_stopped(self, event: RunStopped) -> None:
        """Called by the publisher after RunStopped has been dispatched.

        Constructs the :py:class:`RunResult`, resolves the dataset future
        (to ``None`` if no sink claimed it), completes the run future,
        and releases the engine slot.
        """
        with self._current_lock:
            sub = self._subs_by_id.pop(event.run_id, None)
            if (
                self._current_sub is not None
                and self._current_sub.run_id == event.run_id
            ):
                self._current_sub = None
        if sub is None:
            return  # already finalized (defensive)
        if not sub.dataset_future.done():
            sub.dataset_future.set_result(None)
        if not sub.future.done():
            sub.future.set_result(
                RunResult(
                    run_id=event.run_id,
                    reason=event.reason,
                    error=event.error,
                    started_at=event.started_at,
                    stopped_at=event.stopped_at,
                    cancel_latency=event.cancel_latency,
                    n_rows_emitted=event.n_rows_emitted,
                )
            )

    def _publisher_loop(self) -> None:
        while True:
            item = self._publish_queue.get()
            if item is _PUBLISH_SHUTDOWN:
                break
            event = item  # type: ignore[assignment]
            for sink in self._sinks:
                try:
                    sink(event)  # type: ignore[arg-type]
                except BaseException:
                    if is_critical(sink):
                        _LOG.exception(
                            "Critical sink %r raised on %s; "
                            "run integrity may be compromised.",
                            sink,
                            type(event).__name__,
                        )
                        # Tracer scope: log only. v1 will escalate to abort the run.
                    else:
                        _LOG.exception(
                            "Non-critical sink %r raised on %s; continuing.",
                            sink,
                            type(event).__name__,
                        )
            # After RunStarted is delivered, ask any dataset-providing sink
            # (e.g., SqliteSink) for the dataset and resolve handle.dataset.
            if isinstance(event, RunStarted):
                self._resolve_dataset_for(event.run_id)
            # Finalize after all sinks have seen RunStopped — this is what
            # makes handle.wait() return only when data is durable.
            if isinstance(event, RunStopped):
                self._finalize_after_run_stopped(event)

    def _resolve_dataset_for(self, run_id: UUID) -> None:
        """Resolve ``handle.dataset`` from the first sink that has one."""
        with self._current_lock:
            sub = self._subs_by_id.get(run_id)
        if sub is None or sub.dataset_future.done():
            return
        for sink in self._sinks:
            provider = getattr(sink, "dataset_for", None)
            if provider is None:
                continue
            try:
                ds = provider(run_id)
            except BaseException:
                _LOG.exception("Sink %r raised in dataset_for; ignoring.", sink)
                continue
            if ds is not None:
                sub.dataset_future.set_result(ds)
                return
