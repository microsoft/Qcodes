"""L1 tests for the ``run(...)`` decorator.

The decorator owns run-lifecycle responsibility: it must yield ``OpenRun``
at the start and ``CloseRun`` at the end (success, error, or cancel) of
the wrapped plan. It also validates the schema at decoration time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from qcodes.measure_v2 import (
    CloseRun,
    Emit,
    Msg,
    OpenRun,
    PlanError,
    Read,
    Set,
    run,
)
from qcodes.measure_v2.testing import drive_plan
from qcodes.parameters import Parameter

if TYPE_CHECKING:
    from collections.abc import Generator


def _trivial_plan(g, i) -> Generator[Msg, Any, None]:
    yield Set(g, 1.0)
    yield Read((i,))
    yield Emit()


def test_run_yields_openrun_first_and_closerun_last() -> None:
    g = Parameter("g")
    i = Parameter("i")
    plan = run(name="t", setpoints=(g,), measured=(i,))(_trivial_plan(g, i))

    result = drive_plan(plan)

    assert isinstance(result.messages[0], OpenRun)
    assert isinstance(result.messages[-1], CloseRun)
    assert result.messages[0].name == "t"
    assert result.messages[0].setpoint_params == (g,)
    assert result.messages[0].measured_params == (i,)


def test_run_preserves_inner_messages_in_order() -> None:
    g = Parameter("g")
    i = Parameter("i")
    plan = run(setpoints=(g,), measured=(i,))(_trivial_plan(g, i))

    result = drive_plan(plan)

    # OpenRun + (Set + Read + Emit) + CloseRun = 5 messages
    assert len(result.messages) == 5
    assert isinstance(result.messages[1], Set)
    assert isinstance(result.messages[2], Read)
    assert isinstance(result.messages[3], Emit)


def test_run_without_schema_raises_plan_error() -> None:
    g = Parameter("g")
    i = Parameter("i")
    plan = run(name="t")(_trivial_plan(g, i))
    with pytest.raises(PlanError, match="explicit setpoints"):
        drive_plan(plan)


def test_run_duplicate_register_names_raises() -> None:
    g1 = Parameter("dup")
    g2 = Parameter("dup")
    i = Parameter("i")
    plan = run(setpoints=(g1, g2), measured=(i,))(_trivial_plan(g1, i))
    with pytest.raises(PlanError, match="register_name"):
        drive_plan(plan)


def test_run_empty_inner_plan_emits_only_lifecycle() -> None:
    """Empty plans yield only the OpenRun + CloseRun bookends.

    (Prior design noted that empty plans produced no events; that was
    incompatible with correct exception forwarding through the decorator.
    Current contract: every run() execution produces an OpenRun and a
    CloseRun, even with no data.)
    """

    def _empty() -> Generator[Msg, Any, None]:
        if False:
            yield  # pragma: no cover

    g = Parameter("g")
    i = Parameter("i")
    plan = run(setpoints=(g,), measured=(i,))(_empty())

    result = drive_plan(plan)

    assert len(result.messages) == 2
    assert isinstance(result.messages[0], OpenRun)
    assert isinstance(result.messages[-1], CloseRun)
    assert not result.cancelled


def test_run_cancel_propagates_to_inner_plan_finally() -> None:
    """REGRESSION: the decorator must forward CancelRequested to inner.

    A naive decorator that manually iterates ``inner.send(...) / yield msg``
    will run its own finally on cancel but won't throw into the inner
    plan — so the inner plan's finally never runs and cleanup is silently
    skipped. ``yield from`` is what gives us correct propagation.
    """
    g = Parameter("g")
    i = Parameter("i")
    inner_finally_ran = []

    def _inner() -> Generator[Msg, Any, None]:
        try:
            for _ in range(1000):
                yield Set(g, 1.0)
                yield Read((i,))
                yield Emit()
        finally:
            inner_finally_ran.append(True)
            yield Set(g, 0.0)  # cleanup message

    plan = run(setpoints=(g,), measured=(i,))(_inner())
    result = drive_plan(plan, cancel_after=4)

    assert inner_finally_ran == [True], (
        "Inner plan's finally must run when the decorator is cancelled"
    )
    # The cleanup Set(g, 0.0) must appear in the message stream
    cleanup_sets = [
        m
        for m in result.messages
        if isinstance(m, Set) and m.value == 0.0 and m.param is g
    ]
    assert len(cleanup_sets) == 1
    assert isinstance(result.messages[-1], CloseRun)


def test_run_closerun_emitted_even_on_cancel() -> None:
    """CloseRun MUST appear in the message stream after a cancel."""
    g = Parameter("g")
    i = Parameter("i")

    def _long_plan() -> Generator[Msg, Any, None]:
        for v in range(1000):
            yield Set(g, float(v))
            yield Read((i,))
            yield Emit()

    plan = run(setpoints=(g,), measured=(i,))(_long_plan())

    result = drive_plan(plan, cancel_after=5)

    assert result.cancelled
    assert isinstance(result.messages[-1], CloseRun)


def test_run_composition_via_yield_from_single_run() -> None:
    """A plan that yield-froms an inner plan still produces ONE run."""
    g = Parameter("g")
    i = Parameter("i")

    def _inner() -> Generator[Msg, Any, None]:
        yield Set(g, 0.0)
        yield Read((i,))
        yield Emit()

    def _outer() -> Generator[Msg, Any, None]:
        yield Set(g, 1.0)
        yield from _inner()
        yield Set(g, 2.0)
        yield Read((i,))
        yield Emit()

    plan = run(setpoints=(g,), measured=(i,))(_outer())
    result = drive_plan(plan)

    opens = [m for m in result.messages if isinstance(m, OpenRun)]
    closes = [m for m in result.messages if isinstance(m, CloseRun)]
    assert len(opens) == 1
    assert len(closes) == 1


def test_run_passes_send_values_through_decorator() -> None:
    """The .send() value (Read result) must reach the inner plan."""
    g = Parameter("g")
    i = Parameter("i")
    seen: list[Any] = []

    def _adaptive() -> Generator[Msg, Any, None]:
        for _ in range(3):
            yield Set(g, 0.0)
            r = yield Read((i,))
            seen.append(r)
            yield Emit()

    plan = run(setpoints=(g,), measured=(i,))(_adaptive())

    drive_plan(plan, on_read=lambda params: {p: 7.0 for p in params})

    assert seen == [{i: 7.0}, {i: 7.0}, {i: 7.0}]
