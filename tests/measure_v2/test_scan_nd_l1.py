"""L1 tests for ``scan_nd``."""

from __future__ import annotations

import numpy as np
import pytest

from qcodes.dataset.dond.sweeps import LinSweep
from qcodes.measure_v2 import Emit, Read, Set, Sleep, scan_nd
from qcodes.measure_v2.testing import drive_plan
from qcodes.parameters import Parameter


def test_scan_nd_requires_at_least_one_sweep() -> None:
    with pytest.raises(ValueError, match="at least one sweep"):
        list(scan_nd(measured=[]))


def test_scan_nd_1d_equivalent_to_scan_1d() -> None:
    """A single-sweep scan_nd matches the structure of scan_1d."""
    from qcodes.measure_v2 import scan_1d  # noqa: PLC0415

    g = Parameter("g")
    i = Parameter("i")
    sweep = LinSweep(g, 0.0, 1.0, 5)

    a = drive_plan(scan_nd(sweep, measured=[i]))
    b = drive_plan(scan_1d(sweep, [i]))

    # Same message types in the same positions (values may differ trivially).
    assert [type(m) for m in a.messages] == [type(m) for m in b.messages]


def test_scan_nd_2d_message_counts() -> None:
    g = Parameter("g")
    h = Parameter("h")
    i = Parameter("i")
    outer = LinSweep(g, 0.0, 1.0, 3)
    inner = LinSweep(h, -1.0, 1.0, 4)

    result = drive_plan(scan_nd(outer, inner, measured=[i]))

    sets = [m for m in result.messages if isinstance(m, Set)]
    reads = [m for m in result.messages if isinstance(m, Read)]
    emits = [m for m in result.messages if isinstance(m, Emit)]

    # 3 outer Sets + 3*4 inner Sets + 2 cleanup Sets (g, h)
    assert len(sets) == 3 + 3 * 4 + 2
    # One Read per innermost point
    assert len(reads) == 3 * 4
    # One Emit per innermost point
    assert len(emits) == 3 * 4


def test_scan_nd_3d_message_counts() -> None:
    a = Parameter("a")
    b = Parameter("b")
    c = Parameter("c")
    i = Parameter("i")
    s1 = LinSweep(a, 0.0, 1.0, 2)
    s2 = LinSweep(b, 0.0, 1.0, 3)
    s3 = LinSweep(c, 0.0, 1.0, 4)

    result = drive_plan(scan_nd(s1, s2, s3, measured=[i]))

    emits = [m for m in result.messages if isinstance(m, Emit)]
    reads = [m for m in result.messages if isinstance(m, Read)]
    assert len(emits) == 2 * 3 * 4
    assert len(reads) == 2 * 3 * 4


def test_scan_nd_outer_set_appears_before_inner_sets() -> None:
    """For each outer point, all inner points run before the next outer Set."""
    g = Parameter("g")  # outer
    h = Parameter("h")  # inner
    i = Parameter("i")
    outer = LinSweep(g, 0.0, 1.0, 3)
    inner = LinSweep(h, 0.0, 1.0, 4)

    result = drive_plan(scan_nd(outer, inner, measured=[i]))

    # Walk the messages: each outer Set should be followed by 4 inner Sets
    # before the next outer Set.
    g_sets = [
        idx
        for idx, m in enumerate(result.messages)
        if isinstance(m, Set) and m.param is g
    ]
    # Three sweep Sets + one cleanup Set
    assert len(g_sets) == 4
    # Between consecutive sweep Sets of g, h should be Set 4 times
    for k in range(3):
        between = result.messages[g_sets[k] + 1 : g_sets[k + 1]]
        h_sets_between = [m for m in between if isinstance(m, Set) and m.param is h]
        if k < 3 - 1:
            # interior gap: 4 inner sets
            assert len(h_sets_between) == 4
        # The last gap (after last outer sweep Set) contains the inner sweep
        # plus the cleanup h Set — drop check; tested elsewhere.


def test_scan_nd_cleanup_ramps_all_sweep_params_to_zero() -> None:
    g = Parameter("g")
    h = Parameter("h")
    i = Parameter("i")
    outer = LinSweep(g, 0.0, 5.0, 4)
    inner = LinSweep(h, 0.0, 5.0, 4)

    result = drive_plan(scan_nd(outer, inner, measured=[i]))

    # Last two Sets in the stream must be (g→0, h→0) in outer-to-inner order.
    sets_at_zero = [m for m in result.messages if isinstance(m, Set) and m.value == 0.0]
    # Among those, the LAST two are the cleanup.
    cleanup = sets_at_zero[-2:]
    assert cleanup[0].param is g
    assert cleanup[1].param is h


def test_scan_nd_cleanup_runs_on_cancel() -> None:
    g = Parameter("g")
    h = Parameter("h")
    i = Parameter("i")
    outer = LinSweep(g, 0.0, 1.0, 100)
    inner = LinSweep(h, 0.0, 1.0, 100)

    result = drive_plan(scan_nd(outer, inner, measured=[i]), cancel_after=5)

    assert result.cancelled
    # The last messages must be the two cleanup Sets.
    last_two_sets = [m for m in result.messages if isinstance(m, Set)][-2:]
    assert last_two_sets[0].param is g
    assert last_two_sets[0].value == 0.0
    assert last_two_sets[1].param is h
    assert last_two_sets[1].value == 0.0


def test_scan_nd_sleeps_on_each_sweep_with_positive_delay() -> None:
    g = Parameter("g")
    h = Parameter("h")
    i = Parameter("i")
    # Outer has delay; inner has no delay
    outer = LinSweep(g, 0.0, 1.0, 3, delay=0.05)
    inner = LinSweep(h, 0.0, 1.0, 4, delay=0.0)

    result = drive_plan(scan_nd(outer, inner, measured=[i]))

    sleeps = [m for m in result.messages if isinstance(m, Sleep)]
    # Sleep only after outer Set: 3 of them, each 0.05s
    assert len(sleeps) == 3
    assert all(s.seconds == pytest.approx(0.05) for s in sleeps)


def test_scan_nd_get_after_set_inserts_readback() -> None:
    """When sweep.get_after_set is True, scan_nd yields Read((param,)) after Set."""
    g = Parameter("g")
    i = Parameter("i")
    sweep = LinSweep(g, 0.0, 1.0, 3, get_after_set=True)

    result = drive_plan(scan_nd(sweep, measured=[i]))

    # Each sweep iteration: Set(g) → Read((g,)) → Read((i,)) → Emit
    # So we have 3 Reads of (g,) and 3 Reads of (i,) — 6 total.
    g_reads = [m for m in result.messages if isinstance(m, Read) and m.params == (g,)]
    i_reads = [m for m in result.messages if isinstance(m, Read) and m.params == (i,)]
    assert len(g_reads) == 3
    assert len(i_reads) == 3


def test_scan_nd_get_after_set_off_no_readback() -> None:
    g = Parameter("g")
    i = Parameter("i")
    sweep = LinSweep(g, 0.0, 1.0, 3, get_after_set=False)

    result = drive_plan(scan_nd(sweep, measured=[i]))

    # No standalone (g,) Read should appear.
    g_only_reads = [
        m for m in result.messages if isinstance(m, Read) and m.params == (g,)
    ]
    assert g_only_reads == []


def test_scan_nd_set_values_are_correct() -> None:
    """Verify the actual sweep values land in the Set messages."""
    g = Parameter("g")
    h = Parameter("h")
    i = Parameter("i")
    outer = LinSweep(g, 0.0, 1.0, 3)
    inner = LinSweep(h, -1.0, 1.0, 2)

    result = drive_plan(scan_nd(outer, inner, measured=[i]))

    # The last Set of each sweep param is the cleanup ramp to 0.0;
    # everything before that is the sweep itself.
    g_sets = [m for m in result.messages if isinstance(m, Set) and m.param is g]
    h_sets = [m for m in result.messages if isinstance(m, Set) and m.param is h]
    # 3 sweep points + 1 cleanup for g
    g_sweep_values = [s.value for s in g_sets[:-1]]
    # 3 outer * 2 inner = 6 sweep points + 1 cleanup for h
    h_sweep_values = [s.value for s in h_sets[:-1]]
    expected_g = list(np.linspace(0.0, 1.0, 3))
    expected_h = list(np.linspace(-1.0, 1.0, 2)) * 3
    assert g_sweep_values == pytest.approx(expected_g)
    assert h_sweep_values == pytest.approx(expected_h)
