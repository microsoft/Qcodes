"""L1 tests for the ``scan_1d`` plan-builder.

These tests exercise the plan-builder in isolation from any engine, using
the ``drive_plan`` helper. The contract under test:

- Yields exactly ``num_points`` of (Set, [Sleep], Read, Emit) cycles.
- ALWAYS yields ``Set(sweep.param, 0.0)`` in its ``finally`` block — this
  is the cancel-safety guarantee that acceptance criterion #2 relies on.
"""

from __future__ import annotations

import numpy as np
import pytest

from qcodes.dataset.dond.sweeps import LinSweep
from qcodes.measure_v2 import Emit, Read, Set, Sleep, run, scan_1d
from qcodes.measure_v2.testing import drive_plan
from qcodes.parameters import Parameter


def test_scan_1d_yields_expected_message_count() -> None:
    g = Parameter("g")
    i = Parameter("i")
    sweep = LinSweep(g, start=0.0, stop=1.0, num_points=5)

    result = drive_plan(scan_1d(sweep, [i]))

    sets = [m for m in result.messages if isinstance(m, Set)]
    reads = [m for m in result.messages if isinstance(m, Read)]
    emits = [m for m in result.messages if isinstance(m, Emit)]
    # 5 sweep Sets + 1 cleanup Set in finally
    assert len(sets) == 6
    assert len(reads) == 5
    assert len(emits) == 5


def test_scan_1d_set_values_match_linspace() -> None:
    g = Parameter("g")
    i = Parameter("i")
    sweep = LinSweep(g, start=-1.0, stop=1.0, num_points=11)

    result = drive_plan(scan_1d(sweep, [i]))

    sets = [m for m in result.messages if isinstance(m, Set) and m.param is g]
    sweep_values = [s.value for s in sets[:-1]]  # last is cleanup
    expected = np.linspace(-1.0, 1.0, 11)
    assert sweep_values == pytest.approx(list(expected))


def test_scan_1d_cleanup_set_is_zero() -> None:
    g = Parameter("g")
    i = Parameter("i")
    sweep = LinSweep(g, start=0.0, stop=5.0, num_points=3)

    result = drive_plan(scan_1d(sweep, [i]))

    last_set = next(m for m in reversed(result.messages) if isinstance(m, Set))
    assert last_set.param is g
    assert last_set.value == 0.0


def test_scan_1d_no_sleep_when_delay_zero() -> None:
    g = Parameter("g")
    i = Parameter("i")
    sweep = LinSweep(g, start=0.0, stop=1.0, num_points=3, delay=0.0)

    result = drive_plan(scan_1d(sweep, [i]))

    assert not any(isinstance(m, Sleep) for m in result.messages)


def test_scan_1d_sleeps_when_delay_positive() -> None:
    g = Parameter("g")
    i = Parameter("i")
    sweep = LinSweep(g, start=0.0, stop=1.0, num_points=3, delay=0.05)

    result = drive_plan(scan_1d(sweep, [i]))

    sleeps = [m for m in result.messages if isinstance(m, Sleep)]
    assert len(sleeps) == 3
    assert all(s.seconds == pytest.approx(0.05) for s in sleeps)


def test_scan_1d_cleanup_runs_on_cancel() -> None:
    """THE acceptance contract: even a mid-sweep cancel must ramp to 0.0."""
    g = Parameter("g")
    i = Parameter("i")
    sweep = LinSweep(g, start=0.0, stop=1.0, num_points=1001)

    # Cancel after the second Set (cancel_after=2 → after [Set(0.0), Read]
    # of the very first point's cycle, partway through the loop).
    result = drive_plan(scan_1d(sweep, [i]), cancel_after=2)

    assert result.cancelled
    last_set = next(m for m in reversed(result.messages) if isinstance(m, Set))
    assert last_set.param is g
    assert last_set.value == 0.0, (
        "scan_1d MUST yield Set(sweep.param, 0.0) in its finally. "
        "This is the contract acceptance criterion #2 depends on."
    )


def test_scan_1d_multiple_measured_params_one_read_msg() -> None:
    """Multiple measured params are batched into one Read message per point."""
    g = Parameter("g")
    i = Parameter("i")
    j = Parameter("j")
    sweep = LinSweep(g, start=0.0, stop=1.0, num_points=3)

    result = drive_plan(scan_1d(sweep, [i, j]))

    reads = [m for m in result.messages if isinstance(m, Read)]
    assert len(reads) == 3
    assert all(r.params == (i, j) for r in reads)


def test_scan_1d_under_run_decorator_emits_lifecycle() -> None:
    """End-to-end L1: run() + scan_1d together produces a valid run."""
    g = Parameter("g")
    i = Parameter("i")
    sweep = LinSweep(g, start=0.0, stop=1.0, num_points=5)

    plan = run(name="t", setpoints=(g,), measured=(i,), shapes={"i": (5,)})(
        scan_1d(sweep, [i])
    )

    result = drive_plan(plan)

    # OpenRun + 5*(Set + Read + Emit) + cleanup Set + CloseRun
    assert len(result.messages) == 1 + 5 * 3 + 1 + 1
    # Last message must always be CloseRun (decorator's finally)
    from qcodes.measure_v2 import CloseRun, OpenRun  # noqa: PLC0415

    assert isinstance(result.messages[0], OpenRun)
    assert isinstance(result.messages[-1], CloseRun)
