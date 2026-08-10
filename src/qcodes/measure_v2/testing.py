"""Testing utilities for the measure_v2 plan/engine API.

These helpers are public-API so users writing custom plan-builders can
unit-test them without instantiating an engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from qcodes.measure_v2.exceptions import CancelRequested
from qcodes.measure_v2.messages import Msg, Read

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from qcodes.parameters import ParameterBase


@dataclass
class DrivePlanResult:
    """Result of :py:func:`drive_plan`."""

    messages: list[Msg] = field(default_factory=list)
    cancelled: bool = False
    error: BaseException | None = None


def drive_plan(
    plan: Generator[Msg, Any, None],
    *,
    on_read: Callable[[tuple[ParameterBase, ...]], dict[ParameterBase, Any]]
    | None = None,
    cancel_after: int | None = None,
    cancel_reason: str = "test",
) -> DrivePlanResult:
    """Drive a plan to completion, synthesizing responses to ``Read`` messages.

    A miniature, in-memory engine for unit-testing plan-builders. Records
    every message the plan yields. Optionally simulates cancellation by
    throwing :py:class:`~qcodes.measure_v2.exceptions.CancelRequested` into
    the plan at a chosen message index.

    Args:
        plan: The plan generator to drive.
        on_read: Optional callback to synthesize ``Read`` responses. Called
            with the params tuple from the ``Read`` message; must return a
            dict mapping each param to its value. Defaults to returning 0.0
            for every param.
        cancel_after: If set, throw :py:class:`CancelRequested` after this
            many messages have been yielded. The plan's ``finally`` blocks
            should still run; their yielded messages are appended to the
            result.
        cancel_reason: The reason string to attach to the thrown
            :py:class:`CancelRequested`.

    Returns:
        A :py:class:`DrivePlanResult` with the message log and the cancel
        state. Well-behaved plans either complete normally (``cancelled=False``)
        or re-raise the ``CancelRequested`` after their finally
        (``cancelled=True``).

    """
    result = DrivePlanResult()
    try:
        msg = next(plan)
        while True:
            result.messages.append(msg)
            if cancel_after is not None and len(result.messages) == cancel_after:
                result.cancelled = True
                msg = plan.throw(CancelRequested(cancel_reason))
                continue
            send_value: Any = None
            if isinstance(msg, Read):
                send_value = (
                    on_read(msg.params)
                    if on_read is not None
                    else dict.fromkeys(msg.params, 0.0)
                )
            msg = plan.send(send_value)
    except StopIteration:
        pass
    except CancelRequested:
        # Plan re-raised after its finally; this is the well-behaved path.
        result.cancelled = True
    return result
