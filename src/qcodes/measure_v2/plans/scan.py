"""Scan plan-builders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qcodes.measure_v2.messages import Emit, Msg, Read, Set, Sleep

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from qcodes.dataset.dond.sweeps import AbstractSweep
    from qcodes.parameters import ParameterBase


def scan_1d(
    sweep: AbstractSweep,
    measured: Sequence[ParameterBase],
    *,
    ramp_down_value: float = 0.0,
) -> Generator[Msg, Any, None]:
    """Sweep one parameter, reading measured parameters at each setpoint.

    Yields one ``Set``/(optional ``Sleep``)/``Read``/``Emit`` cycle per
    setpoint. In its ``finally`` block, **always** yields a final ``Set``
    that drives the swept parameter to ``ramp_down_value`` (default 0.0).
    This guarantee is the basis of the tracer's cancel-safety contract:
    a cancelled scan returns the swept parameter to a known state.

    The plan-builder does **not** yield ``OpenRun``/``CloseRun``; wrap with
    :py:func:`qcodes.measure_v2.run` to open a run for the dataset.

    Args:
        sweep: The :py:class:`~qcodes.dataset.dond.sweeps.AbstractSweep`
            describing the swept parameter and its setpoint values.
        measured: Parameters to read at each setpoint.
        ramp_down_value: Value to set the swept parameter to on exit
            (success, error, or cancel). Defaults to 0.0.

    Yields:
        Plan messages.

    """
    try:
        for v in sweep.get_setpoints():
            yield Set(sweep.param, float(v))
            if sweep.delay > 0:
                yield Sleep(sweep.delay)
            yield Read(tuple(measured))
            yield Emit()
    finally:
        # Mandatory cleanup: leave the swept parameter at a known value.
        yield Set(sweep.param, ramp_down_value)
