from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from opentelemetry import trace

from .do_nd import DondKWargs, dond

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from .do_nd_utils import (
        AxesTupleListWithDataSet,
        ParamMeasT,
    )

LOG = logging.getLogger(__name__)
TRACER = trace.get_tracer(__name__)


@TRACER.start_as_current_span("qcodes.dataset.do0d")
def do0d(
    *param_meas: ParamMeasT, **kwargs: Unpack[DondKWargs]
) -> AxesTupleListWithDataSet:
    """
    Perform a measurement of a single parameter. This is probably most
    useful for a ParameterWithSetpoints that already returns an array of data points.

    Args:
        *param_meas: Parameter(s) to measure at each step or functions that
          will be called at each step. The function should take no arguments.
          The parameters and functions are called in the order they are
          supplied.
        **kwargs: kwargs are the same as for :func:`dond` and forwarded directly to :func:`dond`.

    Returns:
        The QCoDeS dataset.

    """

    kwargs.setdefault("log_info", "Using 'qcodes.dataset.do0d'")

    # since we only support entering parameters
    # as a simple list or args we are sure to always
    # get back a AxesTupleListWithDataSet and cast is safe
    return cast(
        "AxesTupleListWithDataSet",
        dond(*param_meas, **kwargs, squeeze=True),
    )
