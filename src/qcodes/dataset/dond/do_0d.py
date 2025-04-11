from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, overload

from opentelemetry import trace

from .do_nd import DondKWargs, dond

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from .do_nd_utils import (
        AxesTupleListWithDataSet,
        MultiAxesTupleListWithDataSet,
        ParamMeasT,
    )

LOG = logging.getLogger(__name__)
TRACER = trace.get_tracer(__name__)


@overload
def do0d(
    *param_meas: ParamMeasT, squeeze: Literal[False], **kwargs: Unpack[DondKWargs]
) -> MultiAxesTupleListWithDataSet: ...


@overload
def do0d(
    *param_meas: ParamMeasT, squeeze: Literal[True], **kwargs: Unpack[DondKWargs]
) -> AxesTupleListWithDataSet | MultiAxesTupleListWithDataSet: ...


@overload
def do0d(
    *param_meas: ParamMeasT, squeeze: bool = True, **kwargs: Unpack[DondKWargs]
) -> AxesTupleListWithDataSet | MultiAxesTupleListWithDataSet: ...


@TRACER.start_as_current_span("qcodes.dataset.do0d")
def do0d(
    *param_meas: ParamMeasT, squeeze: bool = True, **kwargs: Unpack[DondKWargs]
) -> AxesTupleListWithDataSet | MultiAxesTupleListWithDataSet:
    """
    Perform a measurement of a single parameter. This is probably most
    useful for a ParameterWithSetpoints that already returns an array of data points.

    Args:
        *param_meas: Parameter(s) to measure at each step or functions that
          will be called at each step. The function should take no arguments.
          The parameters and functions are called in the order they are
          supplied.
        squeeze: If True, will return a tuple of QCoDeS DataSet, Matplotlib axis,
            Matplotlib colorbar if only one group of measurements was performed
            and a tuple of tuples of these if more than one group of measurements
            was performed. If False, will always return a tuple where the first
            member is a tuple of QCoDeS DataSet(s) and the second member is a tuple
            of Matplotlib axis(es) and the third member is a tuple of Matplotlib
            colorbar(s).
        **kwargs: kwargs are the same as for dond and forwarded directly to dond.

    Returns:
        The QCoDeS dataset.

    """
    return dond(
        *param_meas,
        squeeze=squeeze,
        **kwargs,
    )
