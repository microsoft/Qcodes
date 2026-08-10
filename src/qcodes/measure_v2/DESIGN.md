# QCoDeS Async Measurement Architecture — Design Exploration

**Status:** exploration / design draft, no implementation
**Scope:** a parallel measurement API for QCoDeS that decouples plan description
from execution, enabling non-blocking measurements, adaptive scans, live data
access, and cancellation with safe cleanup.

**Revision history:**
- **rev2** (post-critique): dropped lazy schema discovery; snapshot taken on
  `OpenRun` unconditionally; introduced sink criticality; added cancel re-check
  after `.send()`; added shutdown deadline; engine canonicalizes parameter
  identity by `register_name`; cross-param parallelism preserves today's
  `underlying_instrument` grouping; `get_after_set` explicit in plan-builders;
  multi-stream `Emit(stream=...)` removed from core vocabulary (deferred);
  softened convenience-layer compatibility claims; documented behavior for
  empty plans and `dataset.cache.data()` thread-safety.

---

## 1. Problem Statement

QCoDeS today performs measurements synchronously on the main thread. The
`Measurement.run()` context manager and `dond(...)` family use nested for-loops
that block the calling kernel for the duration of the scan. This makes several
classes of work awkward or impossible:

- Keeping a Jupyter kernel responsive while a long scan runs.
- Adaptive sweeps that decide their next setpoint from the previous reading.
- Live data consumption (plotters, ML feedback, dashboards) beyond the
  existing dataset subscriber mechanism.
- Pause/resume of in-flight measurements.
- Multiple concurrent measurements on independent rigs in one process.
- Programmatic cancellation with guaranteed safe cleanup of instrument state.

The existing model conflates four distinct concerns into one for-loop:
*describing* the scan, *executing* it on instruments, *writing* it to disk,
and *displaying* it. This document proposes splitting them.

## 2. Goals & Non-Goals

### Goals

- A parallel API surface (not a replacement) usable from notebooks today.
- Plans as data — describable, inspectable, composable, testable without
  instruments.
- Non-blocking submission with handles for status / cancel / pause / resume.
- Live data access via a uniform sink/event model.
- Cancellation that always runs user-defined cleanup.
- Reuse of existing parameter and dataset infrastructure (no driver changes
  required for correctness).
- Clean unit testability of plan logic.

### Non-goals (for v1)

- Replacing or deprecating any existing API.
- Multi-process or remote execution (out-of-process services).
- Async/await user-facing API.
- Driver-level cancel-during-set support.
- Crash recovery / replay of partial runs.
- A measurement queue UI / scheduler beyond simple FIFO.
- Modeling after any specific existing framework (bluesky, pymeasure, Labber).
  Where designs converge it's coincidence; where they diverge it's intentional.

## 3. Architectural Overview

Four components, each owning one concern:

```
                                              +--> SqliteSink (default)
+----------+   +----------------+   +-------+ |
|   Plan   |-->| Measurement    |-->|Publisher|+--> LiveMatplotlibSink
| (gener-  |   | Engine         |   | Thread  | |
| ator of  |<--| (worker thread |   |         | +--> TqdmSink
| Msg)     |   |  + queue)      |   +-------+ |
+----------+   +----------------+              +--> user sinks ...
   ^
   |
+-----------------+
| Plan-builders & |
| Convenience API |
| (qc.scan, ...)  |
+-----------------+
```

- **Plan** — a generator yielding typed messages. Pure data; doesn't perform
  I/O itself. Receives results back via `.send()`.
- **MeasurementEngine** — owns one worker thread; iterates plans; dispatches
  messages to instruments; emits events.
- **Sinks** — callables that consume events. Default sink writes to SQLite by
  wrapping `DataSaver`. Other sinks attach for live viz, network, etc.
- **Convenience API** — `qc.scan(...)` etc., constructs plans + sinks and
  submits them to a default engine.

Strict layering: plans depend only on the message vocabulary and
`ParameterBase`. The engine depends on the message vocabulary. Sinks depend on
the event vocabulary. The convenience layer depends on all three.

## 4. Plan Model

### 4.1 Message vocabulary

Seven message types (plus one descriptor message for opt-in metadata):

```python
@dataclass(frozen=True)
class Set:
    param: ParameterBase
    value: Any

@dataclass(frozen=True)
class Read:
    params: tuple[ParameterBase, ...]
    # Engine returns dict[ParameterBase, Any] via .send()

@dataclass(frozen=True)
class Sleep:
    seconds: float

@dataclass(frozen=True)
class Call:
    fn: Callable[[], Any]
    # Engine returns fn's result via .send()

@dataclass(frozen=True)
class Emit:
    overrides: Mapping[ParameterBase, Any] = field(default_factory=dict)
    # overrides may only carry params already in the descriptor.
    # Adding new params at Emit time is forbidden (no lazy registration).

@dataclass(frozen=True)
class OpenRun:
    name: str
    exp: Experiment | None
    descriptor: Descriptor           # required — no lazy mode
    write_period: float | None = None

@dataclass(frozen=True)
class CloseRun:
    pass

@dataclass(frozen=True)
class Describe:
    """First-message marker; consumed by the run() decorator to build OpenRun
    when the caller doesn't pass explicit args to run(...). One of explicit
    run() args or Describe MUST be provided — schema is never lazy."""
    setpoints: tuple[ParameterBase, ...]
    measured: tuple[ParameterBase, ...]
    shapes: Shapes | None = None
```

`Emit(stream=...)` (multi-stream within one run) is deliberately not in v1.
Per-stream schemas and SqliteSink mapping are non-trivial; defer until a
concrete use case demands it.

Deliberately absent:

- No `BreakIf` / `Checkpoint` — both reduce to `Call`.
- No `Subscribe` — sink registration is engine API, not plan content.
- No `Pause` / `Resume` — engine API only.
- No `Wait(condition)` — composes from `Sleep` + `Read` + plan-level loop.

### 4.2 Plan as generator

```python
Plan = Generator[Msg, Any, None]
```

A plan is a generator function. The engine drives iteration with `it.send(...)`,
where the value sent is the result of the previous message:

| Message | What `.send()` returns next iteration |
|---|---|
| `Set`, `Sleep`, `Emit`, `OpenRun`, `CloseRun`, `Describe` | `None` |
| `Read(params)` | `dict[ParameterBase, Any]` |
| `Call(fn)` | return value of `fn` |

### 4.3 Composition

Plans compose with `yield from`. Python's generator protocol propagates both
sent values and thrown exceptions correctly through nested generators, so
adaptive plans built from sub-plans work without engine support.

```python
def outer(...):
    yield Set(g1, 1.0)
    try:
        yield from inner(...)
    finally:
        yield Set(g1, 0.0)
```

### 4.4 Adaptivity

Adaptive plans use the `.send()` channel to receive read results and decide
the next setpoint:

```python
def bisect_transition(gate, drain_i, *, lo, hi, threshold, tol, settle):
    while abs(hi - lo) > tol:
        mid = 0.5 * (hi + lo)
        yield Set(gate, mid)
        yield Sleep(settle)
        r = yield Read((drain_i,))
        yield Emit()
        if r[drain_i] < threshold:
            hi = mid
        else:
            lo = mid
```

External adaptive libraries integrate cleanly because plan-side code is
synchronous between yields:

```python
def adaptive_scan(knob, signal, bounds, *, loss_goal, settle):
    learner = Learner1D(function=None, bounds=bounds)
    while learner.loss() > loss_goal:
        (x,), _ = learner.ask(1)
        yield Set(knob, x)
        yield Sleep(settle)
        r = yield Read((signal,))
        yield Emit()
        learner.tell(x, float(r[signal]))
```

## 5. Engine

### 5.1 Threading

One dedicated worker thread per `MeasurementEngine` instance. The engine owns
this thread and instantiates it lazily.

Rationale:

- Driver code stays synchronous. Per-instrument thread-safety constraints
  (pyvisa, NI-DAQmx, ZI) are satisfied by always touching an instrument from
  the same thread.
- GIL is irrelevant — work is I/O bound.
- Cancellation is cooperative checks between yields, not thread interruption.
- The engine can run an internal `ThreadPoolExecutor` for parallel `Read`
  reads without exposing parallelism to plans.

For multi-rig scenarios, instantiate multiple engines, each with its own
disjoint instrument set.

### 5.2 Dispatch loop (sketch)

```python
def _run_plan(self, plan: Plan) -> None:
    it = iter(plan)
    send_value: Any = None
    in_cleanup = False
    while True:
        try:
            if self._cancel_pending and not in_cleanup:
                in_cleanup = True
                msg = it.throw(CancelRequested(self._cancel_reason))
            else:
                self._paused.wait()
                msg = it.send(send_value)
        except StopIteration:
            return
        # Re-check cancel between .send() and dispatch: a cancel arriving in
        # this window must not cause one extra Set/Read to land on hardware.
        if self._cancel_pending and not in_cleanup:
            in_cleanup = True
            try:
                msg = it.throw(CancelRequested(self._cancel_reason))
            except StopIteration:
                return
        send_value = self._dispatch(msg)

def _dispatch(self, msg: Msg) -> Any:
    match msg:
        case Set(p, v):
            p.set(v)
            self._state[p.register_name] = v
        case Read(params):
            out = self._read_pool.read(params)   # grouped by underlying_instrument
            self._state.update({p.register_name: v for p, v in out.items()})
            return out
        case Sleep(s):
            self._cancellable_sleep(s)
        case Call(fn):
            return fn()
        case Emit(overrides):
            snapshot = {**self._state, **{p.register_name: v for p, v in overrides.items()}}
            # overrides containing un-declared params raise before publish
            self._validate_against_descriptor(overrides)
            self._publish(RowEmitted(self._run_id, snapshot, self._next_seq()))
        case OpenRun(name, exp, descriptor, wp):
            self._publish(RunStarted(self._run_id, name, exp, descriptor, wp, time.time()))
        case CloseRun():
            self._publish(RunStopped(...))
```

State is keyed by `register_name`, not parameter object identity, to match
how the dataset identifies parameters (see §10).

### 5.3 Engine API

```python
class MeasurementEngine:
    def __init__(self, *, name: str = "default",
                 sinks: Sequence[DataSink] = (DEFAULT_SQLITE_SINK,),
                 read_pool_size: int | None = None) -> None: ...

    def submit(self, plan: Plan, *, name: str = "",
               sinks: Sequence[DataSink] = ()) -> RunHandle: ...

    def cancel(self, h: RunHandle) -> None: ...
    def pause(self, h: RunHandle) -> None: ...
    def resume(self, h: RunHandle) -> None: ...
    def shutdown(self, *, wait: bool = True) -> None: ...

class RunHandle:
    uuid: UUID
    dataset: Future[DataSetProtocol]    # resolves once SQLite sink opens
    future: Future[RunResult]           # resolves on RunStopped
    @property
    def status(self) -> RunStatus: ...
    def wait(self, timeout: float | None = None) -> RunResult: ...
    def cancel(self) -> None: ...
```

### 5.4 Concurrent submissions

Concurrent submits to a single engine are queued FIFO. The engine processes
one plan at a time. Cancelling a queued-but-not-started run removes it from
the queue and emits `RunStopped(reason="cancelled_before_start")` to sinks
without iterating the plan.

For parallel execution across rigs, use multiple engine instances.

## 6. Sinks & Events

### 6.1 Event vocabulary

```python
@dataclass(frozen=True)
class RunStarted:
    run_id: UUID
    name: str
    exp: Experiment | None
    descriptor: Descriptor | None
    write_period: float | None
    started_at: float

@dataclass(frozen=True)
class RowEmitted:
    run_id: UUID
    stream: str
    snapshot: Mapping[ParameterBase, Any]
    seq: int

@dataclass(frozen=True)
class RunStopped:
    run_id: UUID
    reason: Literal["completed", "cancelled", "interrupted",
                    "error", "engine_shutdown", "cancelled_before_start"]
    error: BaseException | None
    stopped_at: float
    cancel_latency: float | None
    n_rows_emitted: int
```

### 6.2 Sink protocol

```python
class DataSink(Protocol):
    def __call__(self, event: Event) -> None: ...
```

A sink is just a callable. Trivial to implement; symmetric across writer,
plotter, network publisher, test fakes.

### 6.3 Publisher thread and sink criticality

One publisher thread per engine. Engine emits events into a bounded queue;
publisher drains and fans out to sinks in **criticality order**: critical
sinks first (so durable storage runs before non-critical observers), then
non-critical sinks.

```
[engine thread] --put--> [Queue, bounded] --get--> [publisher thread]
                                                       +--> [critical] sqlite_sink(event)
                                                       +--> [non-crit] plotter_sink(event)
                                                       +--> [non-crit] user_sink(event)
```

**Sink criticality.** Each sink declares `critical: bool` (default `False`):

- **Critical sinks** (e.g., the default `SqliteSink`): an exception aborts
  the run. On `RunStarted` failure, the engine throws `SinkOpenFailed` into
  the plan generator (which runs its `finally`) and emits
  `RunStopped(reason="error")`. On mid-run or `RunStopped` failure, the
  exception is logged, the dataset is marked errored, and propagates to
  `RunResult.error`.
- **Non-critical sinks** (plotters, network publishers): exceptions are
  caught and logged. The sink is not unregistered. Other sinks continue.

Rationale: silent data loss on a "completed" run is unacceptable for the
default storage path. Plotters losing a frame is fine.

**Ordering guarantees:**
- Events delivered to all sinks in emission order.
- Critical sinks run before non-critical sinks for each event.
- Engine thread never blocks on disk I/O except via backpressure (§6.4).
- `handle.dataset` future resolves immediately after the critical SqliteSink
  finishes processing `RunStarted` for that run.
- `handle.future` (a `Future[RunResult]`) resolves only after the publisher
  has processed `RunStopped` through **all** sinks — i.e., after the dataset
  is closed on disk. This means `handle.wait()` returning is a guarantee that
  data is durably committed.

### 6.4 Backpressure

Bounded queue between engine and publisher.

- Default queue size: tuned for ~1s of typical event throughput.
- Queue holds event objects (cheap); large array payloads inside `RowEmitted`
  are referenced, not copied. Memory cost is bounded by in-flight Emit count.
- On overflow: **engine thread blocks**. Acquisition pauses until writer
  catches up. Logged as warning. Correct default for DAQ — better to slow the
  sweep than to drop data.
- Future: opt-in lossy sinks for pure live-viz that can drop intermediate
  events but never `RunStarted`/`RunStopped`/critical `RowEmitted`.
- Invariant: `RunStopped` is never dropped, even on cancel or shutdown.
  Publisher fully drains before reporting run finished.

### 6.5 Default SQLite sink

The default sink wraps existing `DataSaver`. The dataclass-to-`add_result`
conversion is trivial, but the **schema-registration lifecycle is where the
real integration work lives**:

```python
class SqliteSink:
    critical = True

    def __call__(self, event: Event) -> None:
        match event:
            case RunStarted(run_id, name, exp, descriptor, wp, _):
                meas = Measurement(name=name, exp=exp)
                # Register all setpoints from descriptor
                for p in descriptor.setpoints:
                    meas.register_parameter(p)
                # Register measured params with their setpoint dependencies
                for p in descriptor.measured:
                    meas.register_parameter(p, setpoints=descriptor.setpoints,
                                            shapes=descriptor.shapes)
                if wp is not None:
                    meas.write_period = wp
                saver = meas.run().__enter__()        # snapshot taken here
                self._savers[run_id] = saver
                self._resolve_dataset_future(run_id, saver.dataset)
            case RowEmitted(run_id, snapshot, _):
                self._savers[run_id].add_result(*snapshot.items())
            case RunStopped(run_id, reason, error, *_):
                exc_type = type(error) if error else None
                self._savers.pop(run_id).__exit__(exc_type, error, None)
```

Key points:

- The descriptor is required (no lazy mode), so `RunStarted` is the only
  moment the sink registers parameters. No "register at first Emit" complexity.
- `meas.run().__enter__()` is where the station snapshot is taken — exactly
  matches today's behavior. The plan's setup actions (warm-up, ramping) run
  *after* this snapshot, also matching today's behavior with `enter_actions`.
- `handle.dataset` is resolved in this method, before any `RowEmitted` can
  arrive on the queue (publisher is single-threaded).
- For `ParameterWithSetpoints` / `MultiParameter`, all existing array
  fan-out and shape-validation logic in `DataSaver.add_result` is reused
  unchanged. The sink itself stays small.

The non-trivial work is not in the sink — it's in ensuring the descriptor
passed in `RunStarted` is correct. That's the responsibility of the
plan-builder (via `Describe`) or the user (via explicit `run()` args).

## 7. Run Lifecycle / `run()` Decorator

Plans don't open their own runs. The `run(...)` decorator wraps a plan-builder
to inject `OpenRun`/`CloseRun` around it. The descriptor is mandatory — it
comes from either explicit args or a `Describe` first message:

```python
def run(*, name: str = "", exp: Experiment | None = None,
        setpoints: tuple[ParameterBase, ...] | None = None,
        measured: tuple[ParameterBase, ...] | None = None,
        shapes: Shapes | None = None,
        write_period: float | None = None) -> Callable[[Plan], Plan]:
    def wrap(inner: Plan) -> Plan:
        try:
            first = next(inner)
        except StopIteration:
            # Empty plan: no run, no events. Returns immediately.
            # RunResult will report reason="completed", n_rows_emitted=0,
            # dataset future resolved to None.
            return

        described = first if isinstance(first, Describe) else None
        first_passthrough = None if described is not None else first

        explicit = (setpoints is not None or measured is not None)
        if explicit and described is not None:
            raise PlanError("Both run() args and Describe given; pick one.")
        if not explicit and described is None:
            raise PlanError(
                "Plan has no schema. Provide either run(setpoints=..., "
                "measured=...) args or yield Describe(...) as the first "
                "message of the plan. Lazy schema discovery is not supported."
            )

        if explicit:
            descr = Descriptor(setpoints=setpoints or (),
                               measured=measured or (),
                               shapes=shapes)
        else:
            descr = Descriptor(setpoints=described.setpoints,
                               measured=described.measured,
                               shapes=described.shapes)

        yield OpenRun(name=name, exp=exp, descriptor=descr, write_period=write_period)
        try:
            if first_passthrough is not None:
                yield first_passthrough
            yield from inner
        finally:
            yield CloseRun()
    return wrap
```

Properties:

- Plans without an explicit schema (neither `run()` args nor `Describe`)
  fail fast at decoration time with a clear error.
- Empty plans (zero yielded messages) are valid and produce no run.
- Plans composable via `yield from` — an outer plan can wrap sub-plans
  without re-opening runs, since only the outermost `run(...)` decorator
  yields `OpenRun`/`CloseRun`.

## 8. Schema Declaration

Every run has a `Descriptor` declared **before** execution begins. It carries:

- `setpoints: tuple[ParameterBase, ...]` — parameters the plan sweeps via `Set`.
- `measured: tuple[ParameterBase, ...]` — parameters the plan reads via `Read`.
- `shapes: Shapes | None` — optional per-measured-param shape hints.

Two modes only:

1. **Explicit `run(...)` args** — for callers (e.g., the convenience layer)
   that know everything about the run.

   ```python
   plan = run(
       name="g1_vs_g2",
       setpoints=(g1, g2),
       measured=(current,),
       shapes=(11, 11),
   )(scan_inner_outer(LinSweep(g1, 0, 1, 11), LinSweep(g2, 0, 1, 11), [current]))
   ```

2. **`Describe` as the first yielded message** — for plan-builders whose
   shape is known to the builder but not to the caller. The `run(...)`
   decorator consumes the `Describe` to construct the descriptor.

   ```python
   def scan_inner_outer(outer, inner, measured) -> Plan:
       yield Describe(
           setpoints=(outer.param, inner.param),
           measured=tuple(measured),
           shapes=(outer.num_points, inner.num_points),
       )
       for v_outer in outer.get_setpoints():
           ...
   ```

The two modes are mutually exclusive — passing both raises `PlanError`.
Providing neither also raises `PlanError`. Schema is never inferred from
observed `Set`/`Read` calls (this was considered and dropped — see
revision history).

**Validation at runtime:**

- A `Set(p, v)` whose `p.register_name` isn't in `descriptor.setpoints`
  is allowed (intermediate setpoints, warm-up, etc. — these execute but
  aren't written to the dataset).
- A `Read((p1, p2, …))` of a param not in `descriptor.measured` raises.
- An `Emit(overrides={p: v})` whose `p` isn't in
  `descriptor.setpoints | descriptor.measured` raises.

These rules give the SQLite sink a fixed, declared schema before the first
row is written, removing the lifecycle-coordination problem that lazy mode
introduced.

## 9. Cancellation & Cleanup Contract

### 9.1 Single mechanism

All cancellation sources funnel into one path: engine throws `CancelRequested`
into the plan generator at the next yield point (re-checked between `.send()`
and dispatch — see §5.2). The plan's `try/finally` runs. Cleanup messages are
dispatched. `RunStopped(reason=…)` is delivered to sinks. Different sources
differ only in the `reason` string.

| Source | Trigger |
|---|---|
| User explicit | `handle.cancel()` |
| KeyboardInterrupt | Ctrl-C while in `handle.wait()` |
| Engine shutdown | `engine.shutdown()` cancels all live runs |
| Plan-internal | Plan raises `BreakConditionInterrupt` |
| Process termination | `atexit` calls `shutdown(wait=True)` with a deadline |

### 9.1a Shutdown deadline (escape valve)

`engine.shutdown(wait=True, timeout=30.0)` carries a deadline (default 30 s).
On expiry, the engine logs loudly and returns; in-flight cleanups continue
on the engine thread until they finish or the process exits. `atexit`
invokes `shutdown(wait=True, timeout=30.0)` — the deadline prevents
interpreter shutdown from hanging on a stuck driver.

This is the only way the "cleanup always runs" rule has an escape: not by
skipping cleanup, but by detaching the caller after a bounded wait.

### 9.2 Latency

Cancel takes effect at the next yield point. Bounded by:

| Currently dispatching | Cancel latency |
|---|---|
| `Set` (no ramp) | ≈ instrument I/O time |
| `Set` with `step`/`inter_delay` ramp | full ramp duration |
| `Read` | slowest `get` |
| `Sleep(s)` | ~100 ms (engine chunks sleep with cancel-flag checks) |
| `Call(fn)` | `fn`'s runtime |
| Cleanup messages | same rules; second cancel ignored |

There is **no `immediate=True` mode**. Cleanup always runs. If cleanup hangs,
the only escape is killing the process — same as today.

### 9.3 Plan-author rules

1. **Cleanup goes in `try/finally`.** The `finally` block can yield messages;
   they're dispatched normally.
2. **If you catch `CancelRequested`, re-raise.** Swallowing makes the plan
   un-cancellable.
3. **Cleanup must not hang or raise.** Slow cleanup is the user's call but
   counts against cancel latency. Raising in cleanup masks the original
   exception.

### 9.4 Canonical safe-bias example

```python
def biased_sweep(bias, current, targets, *, max_step=0.1, settle=10e-3):
    def safe_ramp(target):
        current_val = bias.cache.get() if bias.cache.valid else 0.0
        n = max(1, int(abs(target - current_val) / max_step) + 1)
        for step in np.linspace(current_val, target, n):
            yield Set(bias, step)
            yield Sleep(0.02)

    yield from safe_ramp(targets[0])
    try:
        for v in targets:
            yield Set(bias, v)
            yield Sleep(settle)
            yield Read((current,))
            yield Emit()
    finally:
        yield from safe_ramp(0.0)
```

Cancel mid-sweep: `CancelRequested` is thrown at the current yield, the inner
`finally` runs `safe_ramp(0.0)`, run is reported stopped only after the ramp
completes.

### 9.5 Composition

`yield from` propagates exceptions naturally. Nested plans' `finally` blocks
run in reverse-nested order on cancel. No engine support needed beyond
Python's generator protocol.

### 9.6 Ctrl-C handling

```python
def wait(self, timeout=None) -> RunResult:
    try:
        return self._future.result(timeout=timeout)
    except KeyboardInterrupt:
        if self._first_interrupt:
            self._first_interrupt = False
            self.engine.cancel(self)
            return self._future.result()
        raise          # second Ctrl-C escapes
```

Matches today's `catch_interrupts`: first Ctrl-C cancels gracefully; second
escapes (cleanup may still run on engine thread).

### 9.7 RunResult

```python
@dataclass(frozen=True)
class RunResult:
    run_id: UUID
    reason: Literal[...]
    error: BaseException | None
    started_at: float
    stopped_at: float
    cancel_latency: float | None       # surfaces slow cleanups
    n_rows_emitted: int
```

## 10. Parameter Contract

The engine demands almost nothing from `ParameterBase`:

| Engine action | Method called |
|---|---|
| `Set(p, v)` | `p.set(v)` |
| `Read((p,))` | `p.get()` |
| State cache key | `p.register_name` (canonical identity — see below) |

Everything else (`validators`, `step`, `inter_delay`, `scale`, `cache`,
`snapshot`, `setpoints`, etc.) is internal to `set`/`get` or used only by the
sink, and is reused unchanged.

### Canonical parameter identity

The engine identifies parameters by `register_name`, not by object identity.
Rationale: `DataSaver.add_result` validates by name and rejects unknown
names, so two distinct `Parameter` objects sharing a name would pass engine
identity checks but fail at the sink. Failing fast at submission is better.

At `engine.submit(plan)`:
- The descriptor's `setpoints` and `measured` are checked for duplicate
  `register_name` — duplicates raise `PlanError`.
- The state cache uses `register_name` keys for both reads and emits.
- Two `Parameter` objects in the same descriptor with the same name → error.

Sink-side reuse (default `SqliteSink` via `DataSaver`):

- `p.full_name`, `p.label`, `p.unit` for registration
- `p.shapes`, `p.setpoints` for `ParameterWithSetpoints`
- `p.snapshot()` for run snapshot (taken on `RunStarted` / `meas.run().__enter__()`)
- All existing array fan-out logic

### Thread-safety contract — preserves today's behavior

The engine reuses today's `underlying_instrument` grouping for parallel reads:

- `Read((p1, p2, …))` partitions params by `p.underlying_instrument`. Params
  sharing an underlying instrument are read sequentially; params on
  *different* underlying instruments are read in parallel via the engine's
  thread pool.
- This matches the existing behavior of `ThreadPoolParamsCaller` exactly.
  Drivers safe under today's `use_threads=True` are safe under the new
  engine; drivers unsafe under it remain unsafe.
- Cross-instrument coupling on shared resources (e.g., a shared VISA bus
  between two `Instrument`s) is still the driver author's problem. Workaround:
  configure the engine's `read_pool_size=1` or set `underlying_instrument` to
  return the shared resource.

### `get_after_set`: plan-builder responsibility

`AbstractSweep.get_after_set` is honored by plan-builders, not by the engine:

```python
# Inside scan_1d / scan_inner_outer:
yield Set(sweep.param, v)
yield Sleep(sweep.delay)
if sweep.get_after_set:
    yield Read((sweep.param,))      # overwrites state[register_name]
yield Read(tuple(measured))
yield Emit()
```

The engine has no special case. The plan-builder uses the existing
`AbstractSweep` API and emits the appropriate `Read` message. Appendix A.1
and the tracer's `scan_1d` both honor this contract.

### Cancellation during set/get

Not interruptible in v1. Cancel waits for the current `set`/`get` to return.
Driver authors who care could opt in to a cancel token in a future revision;
out of scope now.

### Parameter type coverage

All standard parameter types work without modification:

- `Parameter`, `ManualParameter`, `DelegateParameter`
- `ParameterWithSetpoints`, `MultiParameter`, `ArrayParameter`
- `GroupParameter`, `Function`
- Custom `ParameterBase` subclasses with standard `get_raw` / `set_raw`

Callable measurements (today's `param_meas` accepting bare functions): the
convenience layer wraps them in a one-off `Parameter` so the engine sees only
`ParameterBase`. Plan vocabulary stays typed.

### Live data access from main thread

`handle.dataset.cache.data()` is **eventually consistent** in v1. Multiple
threads read/write the dataset and its cache; there's no read lock. For
deterministic live data, attach a sink instead:

```python
class MyLiveSink:
    critical = False
    def __call__(self, event):
        if isinstance(event, RowEmitted):
            my_thread_safe_buffer.append(event.snapshot)
```

A thread-safe `LiveSnapshot` API may be added in a future revision.

## 11. User-Facing API

Three tiers.

### Tier 1: convenience surface (95% of users)

```python
import qcodes as qc

# N-d sweep, replaces do0d/do1d/do2d/dond
ds = qc.scan(
    LinSweep(g1, 0, 1, 11, delay=0.01),
    LinSweep(g2, 0, 1, 11, delay=0.01),
    measure=[current],
    name="g1g2",
)

# Single shot
ds = qc.measure(temperature, current, name="snapshot")

# Adaptive
ds = qc.scan_adaptive(
    knob=g1, signal=current,
    bounds=(-2.0, 0.0),
    loss_goal=0.01, max_points=200,
)

# Non-blocking — return a handle
handle = qc.scan(..., wait=False)
handle.dataset.cache.data()      # peek partial data
qc.live_plot(handle)             # attach live plotter
handle.cancel()
handle.wait()                    # confirm cleanup completed
```

Some kwargs map directly to today's `dond` and ship in v1:

- `name`, `exp`, `write_period`
- `show_progress`, `use_threads`
- `break_condition`
- `wait` (default `True`), `engine` (default `qc.default_engine()`)

Other `dond` features need explicit design and are **not** v1 of the
convenience layer. Users who need them keep using today's `dond`:

- `enter_actions` / `exit_actions` — designable, but their interaction with
  the `run()` decorator's `try/finally` needs thought.
- `before_inner_actions` / `after_inner_actions` — tied to loop structure;
  natural fit as plan-builder args once `scan_inner_outer` is finalized.
- `flush_columns` — engine policy or a new `Checkpoint` message; deferred.
- `additional_setpoints` — needs a dedicated design (registered but not
  swept; today a `Read` at start would express this).
- Grouped measurements / multi-dataset outputs — deferred with multi-stream.
- `live_plot=True` — needs a `LiveMatplotlibSink`; deferred to live-viz
  workstream.

`dond` / `do0d` / `do1d` / `do2d` remain in `qcodes.dataset.dond` unchanged
during v1. They are not yet shims over the engine; a future migration step
may convert them.

### Tier 2: engine surface

```python
eng = qc.default_engine()
# or
eng = qc.MeasurementEngine(name="rig_b", sinks=[my_custom_sink])

handle = eng.submit(my_plan, name="run1")
handle.subscribe(my_callback)        # add sink for this run
handle.pause()
handle.resume()
handle.cancel()
status = handle.status
result = handle.future.result()
```

### Tier 3: plan-builder authoring (library authors)

```python
def find_pinchoff(gate, drain_i, *, threshold=1e-9, ...) -> Plan:
    """Reusable plan-builder. No OpenRun — caller wraps with run(...)."""
    v = 0.0
    while v >= -2.0:
        yield Set(gate, v)
        yield Sleep(20e-3)
        r = yield Read((drain_i,))
        yield Emit()
        if r[drain_i] < threshold:
            return v
        v -= 0.005

# Used via convenience:
ds = qc.run_plan(find_pinchoff(g1, current), name="pinchoff")
```

### Default engine

`qc.default_engine()` lazy-instantiates a process-wide singleton. No
interaction with `Station` — the engine doesn't own instruments, it executes
plans against parameters. Multi-rig users instantiate explicit engines.

## 12. Testing Strategy

Three test levels.

### L1: message-stream tests (no engine)

Drive a plan generator manually; assert the message sequence.

```python
def drive_plan(plan, *, on_read=None, on_call=None) -> list[Msg]:
    out = []
    send_value = None
    try:
        msg = next(plan)
        while True:
            out.append(msg)
            match msg:
                case Read(params): send_value = on_read(params) if on_read else {}
                case Call(fn):     send_value = on_call(fn) if on_call else None
                case _:            send_value = None
            msg = plan.send(send_value)
    except StopIteration:
        pass
    return out

def test_bisect_converges():
    gate = Parameter("gate")
    drain_i = Parameter("drain_i")
    msgs = drive_plan(
        bisect_transition(gate, drain_i, lo=-2, hi=0, threshold=1e-9, tol=0.01),
        on_read=lambda _: {drain_i: 1e-6 if some_condition else 1e-12},
    )
    last_set = [m for m in msgs if isinstance(m, Set)][-1]
    assert abs(last_set.value - (-1.0)) < 0.02
```

Exercises plan logic in isolation. No engine, no sinks, no instruments.

### L2: engine-driven tests (with `MemorySink`)

```python
class MemorySink:
    def __init__(self): self.events = []
    def __call__(self, event): self.events.append(event)

def test_scan_writes_correct_rows():
    g = Parameter("g", initial_value=0.0, set_cmd=None)
    i = Parameter("i", get_cmd=lambda: g.cache.get() ** 2)
    sink = MemorySink()
    eng = MeasurementEngine(sinks=[sink])
    plan = run(name="t")(scan_inner_outer(LinSweep(g, 0, 1, 5), [i]))
    eng.submit(plan).wait()
    eng.shutdown()

    rows = [e for e in sink.events if isinstance(e, RowEmitted)]
    assert len(rows) == 5
    assert rows[-1].snapshot[i] == pytest.approx(1.0)
```

Tests engine + plan + parameter integration without a database.

### L3: end-to-end (with `SqliteSink` and instrument simulators)

Existing pattern; unchanged.

### Time

Plan-builders accept their delays as parameters; tests pass ~0. For tests
that need to assert on timestamps or `cancel_latency`, an opt-in `FakeClock`
abstraction can be passed to the engine.

### Layout

```
tests/
  engine/         # submit, queue, cancel, shutdown, schema discovery
  plans/          # message vocabulary, run() decorator, composition
  sinks/          # memory, sqlite, tee
  builders/       # individual plan-builders, level 1
  integration/    # end-to-end + cancel safety
```

Today's `tests/dataset/test_dond_*.py` keep working unchanged because `dond`
becomes a shim.

## 13. Decisions Log

Decisions superseded by rev2 are struck through. New rev2 decisions are
marked `★`.

| # | Decision |
|---|---|
| 1 | Seven-message vocabulary plus `Describe` (no `Emit(stream=...)` in v1 — ★ rev2 removed) |
| 2 | Single-threaded engine per instance |
| 3 | Plans are generators; `.send()` for adaptivity |
| 4 | DataSink = callable; one publisher thread per engine |
| 5 | Default SQLite sink wraps existing `DataSaver` (schema lifecycle is the real work — ★ rev2) |
| 6 | Bounded queue, block on overflow, never drop `RowEmitted`/`Run*` |
| 7 | `OpenRun`/`CloseRun` injected by `run(...)` decorator, not in builders |
| 8 | New parallel API; defer migration concerns |
| 9 | Driver thread-safety opt-in (no required changes) |
| 10 | Designed on own merits; no specific framework as model |
| 11 | ~~Schema discovery: explicit args > `Describe` > lazy~~ → **★ rev2: lazy mode dropped. Schema is mandatory via `run()` args or `Describe`** |
| 12 | Public API: `qc.scan` / `qc.measure` / `qc.run_plan` + engine for advanced |
| 13 | Default engine = lazy global singleton; multi-engine via explicit instantiation |
| 14 | Concurrent submits to one engine = FIFO queue |
| 15 | Engine surface on params: `set` + `get` only |
| 16 | Sink reuses existing `DataSaver` introspection |
| 17 | Per-param serialization guaranteed; cross-param parallelism follows today's `underlying_instrument` grouping (★ rev2 clarified) |
| 18 | Cancel during `set`/`get` not interruptible in v1 |
| 19 | Callables-as-measurements wrapped at convenience layer |
| 20 | ~~Snapshot on first `Emit` in lazy mode~~ → **★ rev2: snapshot always taken on `OpenRun` (`meas.run().__enter__()`), unconditionally** |
| 21 | Single cancel mechanism: `it.throw(CancelRequested)` |
| 22 | Cleanup runs in plan `try/finally`; engine dispatches yielded messages |
| 23 | Cancel-during-cleanup ignored (in_cleanup flag) |
| 24 | No `immediate=True` mode |
| 25 | First Ctrl-C = cancel; second = re-raise |
| 26 | Cancel latency unbounded (driver-dependent); not a v1 target |
| 27 | Engine `shutdown()` cancels all live runs; `atexit` invokes it |
| 28 | Queued-but-not-started runs cancel without iterating plan |
| 29 | `BreakConditionInterrupt` reuses cancel mechanism |
| 30 | `RunResult.cancel_latency` exposed |
| 31 | Three test levels: message-stream, engine-driven, instrument |
| 32 | `drive_plan(...)` test helper |
| 33 | `MemorySink` as standard test sink |
| 34 | Plan-builders parametrize delays for fast tests; `FakeClock` opt-in |
| 35 | Plan vocabulary module is engine-independent |
| 36 | New test directories alongside existing dond tests |
| ★37 | Cancel flag re-checked after `it.send()` and before `_dispatch` |
| ★38 | Sink criticality: critical sinks (SQLite default) abort run on failure; non-critical sinks (plotters) only log |
| ★39 | Critical sinks run before non-critical sinks per event |
| ★40 | `handle.dataset` resolves after SQLite sink processes `RunStarted`; `handle.future` resolves only after **all** sinks process `RunStopped` (data is durably committed when `wait()` returns) |
| ★41 | Engine identifies parameters by `register_name`, not object identity; duplicate names rejected at submit |
| ★42 | `get_after_set` is a plan-builder concern (yields explicit `Read` after `Set`); engine has no special case |
| ★43 | `engine.shutdown(wait=True, timeout=30.0)` has a deadline; `atexit` uses it |
| ★44 | Empty plans (zero yielded messages) produce no run, no events; `RunResult(reason="completed", n_rows_emitted=0)` |
| ★45 | `dataset.cache.data()` is documented as eventually-consistent; deterministic live data goes through a sink |
| ★46 | Convenience layer v1 ships a reduced kwarg set (see §11); `dond` is NOT yet a shim |

## 14. Open Questions / Future Work

- **Streaming acquisition ergonomics.** F2/F3 cases (chunked Alazar-style)
  work via `Call` + `Emit(overrides=...)` but the boilerplate is real. If
  driver authors hit this often, introduce a `Stream` message or sugar.
- **Multi-stream within one run.** Deliberately removed from v1 vocabulary
  (no `Emit(stream=...)`); needs per-stream schema design plus SqliteSink
  mapping. Re-introduce when a concrete use case justifies it.
- **Lazy schema discovery.** Considered and dropped (see rev2). May
  reconsider if usage patterns show consistent friction with mandatory
  `Describe` for ad-hoc plans.
- **Derived/metadata writes from plans.** `Annotate(...)` message vs
  per-stream `Emit`. Defer.
- **Cancel-aware long sets.** Optional cancel token in `ParameterBase.set`
  for drivers that opt in.
- **FakeClock implementation details.** Engine indirection through a `Clock`
  abstraction.
- **Queue inspection / priority.** Beyond simple FIFO + cancel, no design.
- **Crash recovery / replay.** Out of scope for v1.
- **Out-of-process measurement service.** Could layer on top of this
  architecture (sink protocol is naturally serializable) but explicitly out
  of scope.
- **Static lint** for plan-builders that swallow `CancelRequested`.
- **Thread-safe live `dataset.cache` view.** v1 documents
  `cache.data()` as eventually-consistent; a proper `LiveSnapshot` wrapper
  with appropriate locking can be added later.
- **`enter_actions` / `exit_actions` / inner-loop actions.** Need design
  for how user-provided callables interact with the `run()` decorator's
  `try/finally` semantics.

## 15. Out of Scope

Documented to prevent scope creep:

- async/await user-facing API
- Driver migration to async or to a new base class
- Multi-process execution
- Remote / network measurement service
- Crash recovery, replay, or partial-run resume
- Adaptive-only specialized engine
- Replacing `Measurement` / `dond` / `DataSaver` (they stay alongside; not
  yet shims)
- A queue UI or scheduler beyond FIFO + cancel
- Modeling after any specific external framework
- Lazy schema discovery (considered, dropped in rev2)
- Multi-stream `Emit(stream=...)` (deferred from v1 vocabulary)
- Cancel-during-`set`/`get` interruption (driver-level cancel tokens)
- 2D-mechanical-extension claim: `scan_inner_outer` for v1 is a fresh
  design, not a mechanical lift from `do_nd` (real semantics around
  `set_before_sweep`, inner actions, `flush_columns`, `additional_setpoints`
  need explicit treatment in the plan-builder)

## Appendix A: Plan-Builder Worked Examples

### A.1 N-d scan (NOT a drop-in replacement for dond/do2d)

This is a fresh design for v1, not a mechanical lift. Several `dond`
features (`set_before_sweep`, `flush_columns`, `before/after_inner_actions`,
`additional_setpoints`, `enter_actions`, `exit_actions`) need explicit
treatment and are not in this minimal builder.

```python
def scan_inner_outer(outer, inner, measured) -> Plan:
    yield Describe(
        setpoints=(outer.param, inner.param),
        measured=tuple(measured),
        shapes=(outer.num_points, inner.num_points),
    )
    for v_outer in outer.get_setpoints():
        yield Set(outer.param, v_outer)
        yield Sleep(outer.delay)
        if outer.get_after_set:
            yield Read((outer.param,))
        for v_inner in inner.get_setpoints():
            yield Set(inner.param, v_inner)
            yield Sleep(inner.delay)
            if inner.get_after_set:
                yield Read((inner.param,))
            yield Read(tuple(measured))
            yield Emit()
```

### A.2 Pinchoff (stop-at-condition)

```python
def find_pinchoff(gate, drain_i, *, threshold, settle, step) -> Plan:
    yield Describe(setpoints=(gate,), measured=(drain_i,))
    v = 0.0
    while True:
        yield Set(gate, v)
        yield Sleep(settle)
        r = yield Read((drain_i,))
        yield Emit()
        if r[drain_i] < threshold:
            break
        v += step
```

### A.3 Adaptive (with python-adaptive)

```python
def adaptive_scan(knob, signal, bounds, *, loss_goal, settle):
    yield Describe(setpoints=(knob,), measured=(signal,))
    learner = Learner1D(function=None, bounds=bounds)
    while learner.loss() > loss_goal:
        (x,), _ = learner.ask(1)
        yield Set(knob, x)
        yield Sleep(settle)
        r = yield Read((signal,))
        yield Emit()
        learner.tell(x, float(r[signal]))
```

### A.4 Tracking a 2D feature

```python
def track_resonance(B, f, signal, B_setpoints, *, f_window, f_n, f0_initial, settle):
    yield Describe(setpoints=(B, f), measured=(signal,),
                   shapes=(len(B_setpoints), f_n))
    f_center = f0_initial
    for b in B_setpoints:
        yield Set(B, b)
        yield Sleep(settle)
        f_grid = np.linspace(f_center - f_window/2, f_center + f_window/2, f_n)
        sweep_results = []
        for f_val in f_grid:
            yield Set(f, f_val)
            yield Sleep(settle)
            r = yield Read((signal,))
            yield Emit()
            sweep_results.append(r[signal])
        f_center = f_grid[int(np.argmin(sweep_results))]
```

### A.5 Safe biased sweep with ramp-down

```python
def biased_sweep(bias, current, targets, *, max_step=0.1, settle=10e-3):
    yield Describe(setpoints=(bias,), measured=(current,))

    def safe_ramp(target):
        cv = bias.cache.get() if bias.cache.valid else 0.0
        n = max(1, int(abs(target - cv) / max_step) + 1)
        for step in np.linspace(cv, target, n):
            yield Set(bias, step)
            yield Sleep(0.02)

    yield from safe_ramp(targets[0])
    try:
        for v in targets:
            yield Set(bias, v)
            yield Sleep(settle)
            yield Read((current,))
            yield Emit()
    finally:
        yield from safe_ramp(0.0)
```

## Appendix B: Glossary

- **Plan** — generator of `Msg` objects describing a measurement.
- **Plan-builder** — function returning a plan.
- **Message** — one of `Set`, `Read`, `Sleep`, `Call`, `Emit`, `OpenRun`,
  `CloseRun`, `Describe`.
- **Engine** — `MeasurementEngine`, owns a worker thread and dispatches plans.
- **Sink** — callable accepting `Event` objects.
- **Event** — one of `RunStarted`, `RowEmitted`, `RunStopped`.
- **Run** — one execution of a plan, identified by a UUID, optionally
  associated with a `DataSet`.
- **Descriptor** — declared schema of a run (setpoints, measured, shapes).
- **Convenience layer** — `qc.scan`, `qc.measure`, `qc.run_plan`, etc.
- **Default engine** — process-wide lazy singleton accessed via
  `qc.default_engine()`.
