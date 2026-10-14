"""Smoke tests for the plan message vocabulary.

Verifies that the dataclasses can be constructed and are hashable/frozen.
Plan-builder integration tests live in test_scan_1d_l1.py.
"""

from __future__ import annotations

import pytest

from qcodes.measure_v2 import (
    Emit,
    OpenRun,
    Read,
    Set,
)
from qcodes.parameters import Parameter


def test_messages_are_frozen() -> None:
    g = Parameter("g")
    s = Set(g, 0.5)
    with pytest.raises(Exception):
        s.value = 1.0  # type: ignore[misc]


def test_messages_are_hashable() -> None:
    g = Parameter("g")
    s = Set(g, 0.5)
    # Frozen dataclasses with hashable fields are hashable. Useful for
    # sets/dict keys (e.g., deduplicating expected messages in tests).
    assert hash(s) == hash(Set(g, 0.5))


def test_read_takes_tuple() -> None:
    i = Parameter("i")
    r = Read((i,))
    assert r.params == (i,)


def test_emit_default_overrides_empty() -> None:
    e = Emit()
    assert e.overrides == {}


def test_open_run_carries_descriptor() -> None:
    g = Parameter("g")
    i = Parameter("i")
    o = OpenRun(
        name="t",
        setpoint_params=(g,),
        measured_params=(i,),
        shapes={i.register_name: (5,)},
    )
    assert o.setpoint_params == (g,)
    assert o.measured_params == (i,)
    assert o.shapes is not None
