from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from opentelemetry import trace

from .do_nd import DondKWargs, dond
from .sweeps import LinSweep

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.dataset.dond.do_nd_utils import (
        AxesTupleListWithDataSet,
        ParamMeasT,
    )
    from qcodes.parameters import ParameterBase

LOG = logging.getLogger(__name__)
TRACER = trace.get_tracer(__name__)


@TRACER.start_as_current_span("qcodes.dataset.do1d")
def do1d(
    param_set: ParameterBase,
    start: float,
    stop: float,
    num_points: int,
    delay: float,
    *param_meas: ParamMeasT,
    **kwargs: Unpack[DondKWargs],
) -> AxesTupleListWithDataSet:
    """
    Perform a 1D scan of ``param_set`` from ``start`` to ``stop`` in
    ``num_points`` measuring param_meas at each step. In case param_meas is
    an ArrayParameter this is effectively a 2d scan.

    Args:
        param_set: The QCoDeS parameter to sweep over
        start: Starting point of sweep
        stop: End point of sweep
        num_points: Number of points in sweep
        delay: Delay after setting parameter before measurement is performed
        *param_meas: Parameter(s) to measure at each step or functions that
          will be called at each step. The function should take no arguments.
          The parameters and functions are called in the order they are
          supplied.
        **kwargs: kwargs are the same as for :func:`dond` and forwarded directly to :func:`dond`.

    Returns:
        The QCoDeS dataset.

    """
    kwargs.setdefault("log_info", "Using 'qcodes.dataset.do1d'")

    return cast(
        "AxesTupleListWithDataSet",
        dond(
            LinSweep(
                param=param_set,
                start=start,
                stop=stop,
                delay=delay,
                num_points=num_points,
            ),
            *param_meas,
            **kwargs,
            squeeze=True,
        ),
    )
