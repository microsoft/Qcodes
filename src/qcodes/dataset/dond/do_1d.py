from __future__ import annotations

import logging
import sys
import time
from typing import TYPE_CHECKING, cast

import numpy as np
from opentelemetry import trace
from tqdm.auto import tqdm

from qcodes import config
from qcodes.dataset.descriptions.detect_shapes import detect_shape_of_measurement
from qcodes.dataset.dond.do_nd_utils import (
    BreakConditionInterrupt,
    _handle_plotting,
    _register_actions,
    _register_parameters,
    _set_write_period,
    catch_interrupts,
)
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.threading import (
    SequentialParamsCaller,
    ThreadPoolParamsCaller,
    process_params_meas,
)
from qcodes.parameters import ParameterBase

LOG = logging.getLogger(__name__)
TRACER = trace.get_tracer(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qcodes.dataset.descriptions.versioning.rundescribertypes import Shapes
    from qcodes.dataset.dond.do_nd_utils import (
        ActionsT,
        AxesTupleListWithDataSet,
        BreakConditionT,
        ParamMeasT,
    )
    from qcodes.dataset.experiment_container import Experiment


@TRACER.start_as_current_span("qcodes.dataset.do1d")
def do1d(
    param_set: ParameterBase,
    start: float,
    stop: float,
    num_points: int,
    delay: float,
    *param_meas: ParamMeasT,
    enter_actions: ActionsT = (),
    exit_actions: ActionsT = (),
    write_period: float | None = None,
    measurement_name: str = "",
    exp: Experiment | None = None,
    do_plot: bool | None = None,
    use_threads: bool | None = None,
    additional_setpoints: Sequence[ParameterBase] = tuple(),
    show_progress: bool | None = None,
    log_info: str | None = None,
    break_condition: BreakConditionT | None = None,
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
        param_meas: Parameter(s) to measure at each step or functions that
          will be called at each step. The function should take no arguments.
          The parameters and functions are called in the order they are
          supplied.
        enter_actions: A list of functions taking no arguments that will be
            called before the measurements start
        exit_actions: A list of functions taking no arguments that will be
            called after the measurements ends
        write_period: The time after which the data is actually written to the
            database.
        additional_setpoints: A list of setpoint parameters to be registered in
            the measurement but not scanned.
        measurement_name: Name of the measurement. This will be passed down to
            the dataset produced by the measurement. If not given, a default
            value of 'results' is used for the dataset.
        exp: The experiment to use for this measurement.
        do_plot: should png and pdf versions of the images be saved after the
            run. If None the setting will be read from ``qcodesrc.json``
        use_threads: If True measurements from each instrument will be done on
            separate threads. If you are measuring from several instruments
            this may give a significant speedup.
        show_progress: should a progress bar be displayed during the
            measurement. If None the setting will be read from ``qcodesrc.json``
        log_info: Message that is logged during the measurement. If None a default
            message is used.
        break_condition: Callable that takes no arguments. If returned True,
            measurement is interrupted.

    Returns:
        The QCoDeS dataset.
    """
    if do_plot is None:
        do_plot = cast(bool, config.dataset.dond_plot)
    if show_progress is None:
        show_progress = config.dataset.dond_show_progress

    meas = Measurement(name=measurement_name, exp=exp)
    if log_info is not None:
        meas._extra_log_info = log_info
    else:
        meas._extra_log_info = "Using 'qcodes.dataset.do1d'"

    all_setpoint_params = (param_set,) + tuple(s for s in additional_setpoints)

    measured_parameters = tuple(
        param for param in param_meas if isinstance(param, ParameterBase)
    )
    try:
        loop_shape = (num_points,) + tuple(1 for _ in additional_setpoints)
        shapes: Shapes | None = detect_shape_of_measurement(
            measured_parameters, loop_shape
        )
    except TypeError:
        LOG.exception(
            f"Could not detect shape of {measured_parameters} "
            f"falling back to unknown shape."
        )
        shapes = None

    _register_parameters(meas, all_setpoint_params)
    _register_parameters(meas, param_meas, setpoints=all_setpoint_params, shapes=shapes)
    _set_write_period(meas, write_period)
    _register_actions(meas, enter_actions, exit_actions)

    if use_threads is None:
        use_threads = config.dataset.use_threads

    param_meas_caller = (
        ThreadPoolParamsCaller(*param_meas)
        if use_threads
        else SequentialParamsCaller(*param_meas)
    )

    # do1D enforces a simple relationship between measured parameters
    # and set parameters. For anything more complicated this should be
    # reimplemented from scratch
    with catch_interrupts() as interrupted, meas.run() as datasaver, param_meas_caller as call_param_meas:
        dataset = datasaver.dataset
        additional_setpoints_data = process_params_meas(additional_setpoints)
        setpoints = np.linspace(start, stop, num_points)

        # flush to prevent unflushed print's to visually interrupt tqdm bar
        # updates
        sys.stdout.flush()
        sys.stderr.flush()

        for set_point in tqdm(setpoints, disable=not show_progress):
            param_set.set(set_point)
            time.sleep(delay)
            datasaver.add_result(
                (param_set, set_point), *call_param_meas(), *additional_setpoints_data
            )

            if callable(break_condition):
                if break_condition():
                    raise BreakConditionInterrupt("Break condition was met.")

    return _handle_plotting(dataset, do_plot, interrupted())
