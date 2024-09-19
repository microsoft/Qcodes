from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from opentelemetry import trace

from qcodes import config
from qcodes.parameters import ParameterBase

from ..descriptions.detect_shapes import detect_shape_of_measurement
from ..measurements import Measurement
from ..threading import process_params_meas
from .do_nd_utils import _handle_plotting, _register_parameters, _set_write_period

LOG = logging.getLogger(__name__)
TRACER = trace.get_tracer(__name__)

if TYPE_CHECKING:
    from ..descriptions.versioning.rundescribertypes import Shapes
    from ..experiment_container import Experiment
    from .do_nd_utils import AxesTupleListWithDataSet, ParamMeasT


@TRACER.start_as_current_span("qcodes.dataset.do0d")
def do0d(
    *param_meas: ParamMeasT,
    write_period: float | None = None,
    measurement_name: str = "",
    exp: Experiment | None = None,
    do_plot: bool | None = None,
    use_threads: bool | None = None,
    log_info: str | None = None,
) -> AxesTupleListWithDataSet:
    """
    Perform a measurement of a single parameter. This is probably most
    useful for an ArrayParameter that already returns an array of data points

    Args:
        *param_meas: Parameter(s) to measure at each step or functions that
          will be called at each step. The function should take no arguments.
          The parameters and functions are called in the order they are
          supplied.
        write_period: The time after which the data is actually written to the
            database.
        measurement_name: Name of the measurement. This will be passed down to
            the dataset produced by the measurement. If not given, a default
            value of 'results' is used for the dataset.
        exp: The experiment to use for this measurement.
        do_plot: should png and pdf versions of the images be saved after the
            run. If None the setting will be read from ``qcodesrc.json``
        use_threads: If True measurements from each instrument will be done on
            separate threads. If you are measuring from several instruments
            this may give a significant speedup.
        log_info: Message that is logged during the measurement. If None a default
            message is used.

    Returns:
        The QCoDeS dataset.
    """
    if do_plot is None:
        do_plot = cast(bool, config.dataset.dond_plot)
    meas = Measurement(name=measurement_name, exp=exp)
    if log_info is not None:
        meas._extra_log_info = log_info
    else:
        meas._extra_log_info = "Using 'qcodes.dataset.do0d'"

    measured_parameters = tuple(
        param for param in param_meas if isinstance(param, ParameterBase)
    )

    try:
        shapes: Shapes | None = detect_shape_of_measurement(
            measured_parameters,
        )
    except TypeError:
        LOG.exception(
            f"Could not detect shape of {measured_parameters} "
            f"falling back to unknown shape."
        )
        shapes = None

    _register_parameters(meas, param_meas, shapes=shapes)
    _set_write_period(meas, write_period)

    with meas.run() as datasaver:
        datasaver.add_result(*process_params_meas(param_meas, use_threads=use_threads))
        dataset = datasaver.dataset

    return _handle_plotting(dataset, do_plot)
