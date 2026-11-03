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

    Thin wrapper around :py:func:`scan_nd` for the 1D case. See
    :py:func:`scan_nd` for the full contract; the only difference is that
    ``scan_1d`` takes the sweep as a positional argument (matching the
    single-sweep idiom) instead of as a variadic.

    The mandatory ramp-to-``ramp_down_value`` cleanup on exit (success,
    error, or cancel) is the architecture's safety guarantee — see DESIGN.md.
    """
    yield from scan_nd(sweep, measured=measured, ramp_down_value=ramp_down_value)


def scan_nd(
    *sweeps: AbstractSweep,
    measured: Sequence[ParameterBase],
    ramp_down_value: float = 0.0,
) -> Generator[Msg, Any, None]:
    """Nested sweep over one or more parameters.

    Sweep ordering is outermost-first: ``scan_nd(outer, inner, measured=[i])``
    drives ``outer`` once per ``inner`` cycle. At each innermost point the
    plan reads every measured parameter and emits one dataset row.

    Per-sweep behavior at every setpoint:

    1. ``Set(sweep.param, value)`` — drive the instrument
    2. ``Sleep(sweep.delay)`` if ``sweep.delay > 0``
    3. ``Read((sweep.param,))`` if ``sweep.get_after_set`` — readback wins
       over the requested value in the dataset row.

    On the innermost sweep, after the optional readback:

    4. ``Read(measured)`` — single batched read of all measured parameters
    5. ``Emit()`` — produce one row from the engine's state cache

    Mandatory cleanup (``finally``): every swept parameter is set to
    ``ramp_down_value`` (default ``0.0``), in outer→inner order. This is
    the architecture's cancel-safety contract: on cancel (or error), the
    instrument state is restored to a known value before the run is
    reported stopped.

    Args:
        *sweeps: One or more :py:class:`~qcodes.dataset.dond.sweeps.AbstractSweep`
            instances. The first argument is the outermost loop; the last
            is the innermost.
        measured: Parameters to read at each innermost setpoint.
        ramp_down_value: Value each swept parameter is set to on exit.

    Raises:
        ValueError: If no sweeps are provided.

    """
    if not sweeps:
        raise ValueError("scan_nd requires at least one sweep")

    measured_tuple = tuple(measured)
    try:
        yield from _scan_recursive(sweeps, measured_tuple, depth=0)
    finally:
        # Ramp every swept parameter to the safe value, outer-to-inner.
        # If the plan is being cancelled, these Sets are dispatched in
        # order by the engine before RunStopped is published.
        for sweep in sweeps:
            yield Set(sweep.param, ramp_down_value)


def _scan_recursive(
    sweeps: tuple[AbstractSweep, ...],
    measured: tuple[ParameterBase, ...],
    *,
    depth: int,
) -> Generator[Msg, Any, None]:
    sweep = sweeps[depth]
    is_innermost = depth == len(sweeps) - 1
    for v in sweep.get_setpoints():
        yield Set(sweep.param, float(v))
        if sweep.delay > 0:
            yield Sleep(sweep.delay)
        if sweep.get_after_set:
            # Readback overwrites state[sweep.param] with the actual
            # instrument value, so the dataset row reflects reality.
            yield Read((sweep.param,))

        if is_innermost:
            yield Read(measured)
            yield Emit()
        else:
            yield from _scan_recursive(sweeps, measured, depth=depth + 1)
