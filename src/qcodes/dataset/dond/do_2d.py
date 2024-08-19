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
    set_before_sweep: bool | None = True,
    enter_actions: ActionsT = (),
    exit_actions: ActionsT = (),
    before_inner_actions: ActionsT = (),
    after_inner_actions: ActionsT = (),
    write_period: float | None = None,
    measurement_name: str = "",
    exp: Experiment | None = None,
    flush_columns: bool = False,
    do_plot: bool | None = None,
    use_threads: bool | None = None,
    additional_setpoints: Sequence[ParameterBase] = tuple(),
    show_progress: bool | None = None,
    log_info: str | None = None,
    break_condition: BreakConditionT | None = None,
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
        set_before_sweep: if True the outer parameter is set to its first value
            before the inner parameter is swept to its next value.
        enter_actions: A list of functions taking no arguments that will be
            called before the measurements start
        exit_actions: A list of functions taking no arguments that will be
            called after the measurements ends
        before_inner_actions: Actions executed before each run of the inner loop
        after_inner_actions: Actions executed after each run of the inner loop
        write_period: The time after which the data is actually written to the
            database.
        measurement_name: Name of the measurement. This will be passed down to
            the dataset produced by the measurement. If not given, a default
            value of 'results' is used for the dataset.
        exp: The experiment to use for this measurement.
        flush_columns: The data is written after a column is finished
            independent of the passed time and write period.
        additional_setpoints: A list of setpoint parameters to be registered in
            the measurement but not scanned.
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
        meas._extra_log_info = "Using 'qcodes.dataset.do2d'"
    all_setpoint_params = (
        param_set1,
        param_set2,
    ) + tuple(s for s in additional_setpoints)

    measured_parameters = tuple(
        param for param in param_meas if isinstance(param, ParameterBase)
    )

    try:
        loop_shape = (num_points1, num_points2) + tuple(1 for _ in additional_setpoints)
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

    with (
        catch_interrupts() as interrupted,
        meas.run() as datasaver,
        param_meas_caller as call_param_meas,
    ):
        dataset = datasaver.dataset
        additional_setpoints_data = process_params_meas(additional_setpoints)
        setpoints1 = np.linspace(start1, stop1, num_points1)
        for set_point1 in tqdm(setpoints1, disable=not show_progress):
            if set_before_sweep:
                param_set2.set(start2)

            param_set1.set(set_point1)

            for action in before_inner_actions:
                action()

            time.sleep(delay1)

            setpoints2 = np.linspace(start2, stop2, num_points2)

            # flush to prevent unflushed print's to visually interrupt tqdm bar
            # updates
            sys.stdout.flush()
            sys.stderr.flush()
            for set_point2 in tqdm(setpoints2, disable=not show_progress, leave=False):
                # skip first inner set point if `set_before_sweep`
                if set_point2 == start2 and set_before_sweep:
                    pass
                else:
                    param_set2.set(set_point2)
                    time.sleep(delay2)

                datasaver.add_result(
                    (param_set1, set_point1),
                    (param_set2, set_point2),
                    *call_param_meas(),
                    *additional_setpoints_data,
                )

                if callable(break_condition):
                    if break_condition():
                        raise BreakConditionInterrupt("Break condition was met.")

            for action in after_inner_actions:
                action()
            if flush_columns:
                datasaver.flush_data_to_database()

    return _handle_plotting(dataset, do_plot, interrupted())
