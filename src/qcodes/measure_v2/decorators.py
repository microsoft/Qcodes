"""The ``run(...)`` decorator.

Wraps a plan-builder so that the resulting plan opens and closes a run
around the inner messages. Plan-builders themselves never yield
``OpenRun``/``CloseRun``; this is the only place those messages originate.

Tracer scope: explicit-args mode only. ``Describe`` first-message support
is deferred to v1. The schema (setpoints, measured params, optional shapes)
must be provided as kwargs to ``run(...)`` — failing to provide them raises
:py:class:`~qcodes.measure_v2.exceptions.PlanError` at decoration time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qcodes.measure_v2.exceptions import PlanError
from qcodes.measure_v2.messages import CloseRun, Msg, OpenRun

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from qcodes.dataset.descriptions.versioning.rundescribertypes import Shapes
    from qcodes.dataset.experiment_container import Experiment
    from qcodes.parameters import ParameterBase


Plan = "Generator[Msg, Any, None]"  # alias for documentation


def run(
    *,
    name: str = "",
    exp: Experiment | None = None,
    setpoints: tuple[ParameterBase, ...] | None = None,
    measured: tuple[ParameterBase, ...] | None = None,
    shapes: Shapes | None = None,
    write_period: float | None = None,
) -> Callable[[Generator[Msg, Any, None]], Generator[Msg, Any, None]]:
    """Wrap a plan to open and close a run around it.

    Args:
        name: Run name passed to the dataset.
        exp: Optional :py:class:`~qcodes.dataset.experiment_container.Experiment`
            the run belongs to.
        setpoints: Tuple of parameters the plan sweeps via ``Set``. Required
            in tracer scope.
        measured: Tuple of parameters the plan reads via ``Read``. Required
            in tracer scope.
        shapes: Optional per-measured-param shape hints, in the form
            ``{register_name: (n0, n1, ...)}``.
        write_period: Optional override for the dataset write period.

    Returns:
        A decorator that wraps a plan-generator and yields ``OpenRun`` at
        the start and ``CloseRun`` at the end (success, error, or cancel).

    Raises:
        PlanError: If neither ``setpoints`` nor ``measured`` are provided.
            Lazy schema discovery is not supported in v1.

    """

    def wrap(
        inner: Generator[Msg, Any, None],
    ) -> Generator[Msg, Any, None]:
        return _decorated(
            inner,
            name=name,
            exp=exp,
            setpoints=setpoints,
            measured=measured,
            shapes=shapes,
            write_period=write_period,
        )

    return wrap


def _decorated(
    inner: Generator[Msg, Any, None],
    *,
    name: str,
    exp: Experiment | None,
    setpoints: tuple[ParameterBase, ...] | None,
    measured: tuple[ParameterBase, ...] | None,
    shapes: Shapes | None,
    write_period: float | None,
) -> Generator[Msg, Any, None]:
    if setpoints is None and measured is None:
        raise PlanError(
            "run(...) requires explicit setpoints=... and/or measured=... "
            "kwargs in v1 (lazy schema discovery is not supported). "
            "Pass tuples of ParameterBase instances."
        )

    setpoints = setpoints or ()
    measured = measured or ()

    _check_no_duplicate_register_names(setpoints, measured)

    yield OpenRun(
        name=name,
        setpoint_params=setpoints,
        measured_params=measured,
        exp=exp,
        shapes=shapes,
        write_period=write_period,
    )
    try:
        # ``yield from`` is critical here: it transparently forwards
        # ``.send()`` and ``.throw()`` to the inner generator. A manual
        # ``inner.send(...) / yield msg`` loop would NOT propagate
        # ``CancelRequested`` to the inner plan's ``finally`` blocks.
        yield from inner
    finally:
        yield CloseRun()


def _check_no_duplicate_register_names(
    setpoints: tuple[ParameterBase, ...],
    measured: tuple[ParameterBase, ...],
) -> None:
    seen: dict[str, ParameterBase] = {}
    for p in (*setpoints, *measured):
        name = p.register_name
        if name in seen and seen[name] is not p:
            raise PlanError(
                f"Two distinct parameters share register_name {name!r}: "
                f"{seen[name]} and {p}. Engine identity is canonicalized "
                f"by register_name; duplicates are rejected at submission."
            )
        seen[name] = p
