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


@TRACER.start_as_current_span("qcodes.dataset.do2d")
def do2d(
    param_set1: ParameterBase,
    start1: float,
    stop1: float,
    num_points1: int,
    delay1: float,
    param_set2: ParameterBase,
    start2: float,
    stop2: float,
    num_points2: int,
    delay2: float,
    *param_meas: ParamMeasT,
    **kwargs: Unpack[DondKWargs],
) -> AxesTupleListWithDataSet:
    """
    Perform a 1D scan of ``param_set1`` from ``start1`` to ``stop1`` in
    ``num_points1`` and ``param_set2`` from ``start2`` to ``stop2`` in
    ``num_points2`` measuring param_meas at each step.

    Args:
        param_set1: The QCoDeS parameter to sweep over in the outer loop
        start1: Starting point of sweep in outer loop
        stop1: End point of sweep in the outer loop
        num_points1: Number of points to measure in the outer loop
        delay1: Delay after setting parameter in the outer loop
        param_set2: The QCoDeS parameter to sweep over in the inner loop
        start2: Starting point of sweep in inner loop
        stop2: End point of sweep in the inner loop
        num_points2: Number of points to measure in the inner loop
        delay2: Delay after setting parameter before measurement is performed
        param_meas: Parameter(s) to measure at each step or functions that
          will be called at each step. The function should take no arguments.
          The parameters and functions are called in the order they are
          supplied.
        **kwargs: kwargs are the same as for dond and forwarded directly to dond.

    Returns:
        The QCoDeS dataset.

    """
    return cast(
        "AxesTupleListWithDataSet",
        dond(
            LinSweep(
                param=param_set1,
                start=start1,
                stop=stop1,
                delay=delay1,
                num_points=num_points1,
            ),
            LinSweep(
                param=param_set2,
                start=start2,
                stop=stop2,
                delay=delay2,
                num_points=num_points2,
            ),
            *param_meas,
            **kwargs,
            squeeze=True,
        ),
    )
