import logging
import os
from contextlib import contextmanager
from typing import Callable, Iterator, List, Optional, Sequence, Tuple, Union

import matplotlib
import numpy as np

from qcodes import config
from qcodes.dataset.data_set import DataSet
from qcodes.dataset.descriptions.detect_shapes import \
    detect_shape_of_measurement
from qcodes.dataset.descriptions.versioning.rundescribertypes import Shapes
from qcodes.dataset.measurements import Measurement, res_type
from qcodes.dataset.plotting import plot_dataset
from qcodes.instrument.base import _BaseParameter

ActionsT = Sequence[Callable[[], None]]

ParamMeasT = Union[_BaseParameter, Callable[[], None]]

AxesTuple = Tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar]
AxesTupleList = Tuple[List[matplotlib.axes.Axes],
                      List[Optional[matplotlib.colorbar.Colorbar]]]
AxesTupleListWithDataSet = Tuple[DataSet, List[matplotlib.axes.Axes],
                                 List[Optional[matplotlib.colorbar.Colorbar]]]

OutType = List[res_type]

LOG = logging.getLogger(__name__)


def _process_params_meas(param_meas: Sequence[ParamMeasT]) -> OutType:
    output: OutType = []
    for parameter in param_meas:
        if isinstance(parameter, _BaseParameter):
            output.append((parameter, parameter.get()))
        elif callable(parameter):
            parameter()
    return output


def _register_parameters(
        meas: Measurement,
        param_meas: Sequence[ParamMeasT],
        setpoints: Optional[Sequence[_BaseParameter]] = None,
        shapes: Shapes = None
        ) -> None:
    for parameter in param_meas:
        if isinstance(parameter, _BaseParameter):
            meas.register_parameter(parameter,
                                    setpoints=setpoints)
    meas.set_shapes(shapes=shapes)


def _register_actions(
        meas: Measurement,
        enter_actions: ActionsT,
        exit_actions: ActionsT) -> None:
    for action in enter_actions:
        # this omits the possibility of passing
        # argument to enter and exit actions.
        # Do we want that?
        meas.add_before_run(action, ())
    for action in exit_actions:
        meas.add_after_run(action, ())


def _set_write_period(
        meas: Measurement,
        write_period: Optional[float] = None) -> None:
    if write_period is not None:
        meas.write_period = write_period


@contextmanager
def _catch_keyboard_interrupts() -> Iterator[Callable[[], bool]]:
    interrupted = False

    def has_been_interrupted() -> bool:
        nonlocal interrupted
        return interrupted

    try:
        yield has_been_interrupted
    except KeyboardInterrupt:
        interrupted = True


def do0d(
        *param_meas: ParamMeasT,
        write_period: Optional[float] = None,
        do_plot: bool = True
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
        do_plot: should png and pdf versions of the images be saved after the
            run.

    Returns:
        The QCoDeS dataset.
    """
    meas = Measurement()

    measured_parameters = tuple(param for param in param_meas
                                if isinstance(param, _BaseParameter))

    try:
        shapes: Shapes = detect_shape_of_measurement(
            measured_parameters,
        )
    except TypeError:
        LOG.exception(
            f"Could not detect shape of {measured_parameters} "
            f"falling back to unknown shape.")
        shapes = None

    _register_parameters(meas, param_meas, shapes=shapes)
    _set_write_period(meas, write_period)

    with meas.run() as datasaver:
        datasaver.add_result(*_process_params_meas(param_meas))
        dataset = datasaver.dataset

    return _handle_plotting(dataset, do_plot)


def do1d(
        param_set: _BaseParameter, start: float, stop: float,
        num_points: int, delay: float,
        *param_meas: ParamMeasT,
        enter_actions: ActionsT = (),
        exit_actions: ActionsT = (),
        write_period: Optional[float] = None,
        do_plot: bool = True,
        additional_setpoints: Sequence[ParamMeasT] = tuple(),
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
        enter_actions: A list of functions taking no arguments that will be
            called before the measurements start
        exit_actions: A list of functions taking no arguments that will be
            called after the measurements ends
        write_period: The time after which the data is actually written to the
            database.
        additional_setpoints: A list of setpoint parameters to be registered in
            the measurement but not scanned.
        do_plot: should png and pdf versions of the images be saved after the
            run.

    Returns:
        The QCoDeS dataset.
    """
    meas = Measurement()

    all_setpoint_params = (param_set,) + tuple(
        s for s in additional_setpoints)

    measured_parameters = tuple(param for param in param_meas
                                if isinstance(param, _BaseParameter))
    try:
        loop_shape = tuple(1 for _ in additional_setpoints) + (num_points,)
        shapes: Shapes = detect_shape_of_measurement(
            measured_parameters,
            loop_shape
        )
    except TypeError:
        LOG.exception(
            f"Could not detect shape of {measured_parameters} "
            f"falling back to unknown shape.")
        shapes = None

    _register_parameters(meas, all_setpoint_params)
    _register_parameters(meas, param_meas, setpoints=all_setpoint_params,
                         shapes=shapes)
    _set_write_period(meas, write_period)
    _register_actions(meas, enter_actions, exit_actions)
    param_set.post_delay = delay

    # do1D enforces a simple relationship between measured parameters
    # and set parameters. For anything more complicated this should be
    # reimplemented from scratch
    with _catch_keyboard_interrupts() as interrupted, meas.run() as datasaver:
        additional_setpoints_data = _process_params_meas(additional_setpoints)
        for set_point in np.linspace(start, stop, num_points):
            param_set.set(set_point)
            datasaver.add_result((param_set, set_point),
                                 *_process_params_meas(param_meas),
                                 *additional_setpoints_data)
        dataset = datasaver.dataset
    return _handle_plotting(dataset, do_plot, interrupted())


def do2d(
        param_set1: _BaseParameter, start1: float, stop1: float,
        num_points1: int, delay1: float,
        param_set2: _BaseParameter, start2: float, stop2: float,
        num_points2: int, delay2: float,
        *param_meas: ParamMeasT,
        set_before_sweep: Optional[bool] = True,
        enter_actions: ActionsT = (),
        exit_actions: ActionsT = (),
        before_inner_actions: ActionsT = (),
        after_inner_actions: ActionsT = (),
        write_period: Optional[float] = None,
        flush_columns: bool = False,
        do_plot: bool = True,
        additional_setpoints: Sequence[ParamMeasT] = tuple(),
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
        *param_meas: Parameter(s) to measure at each step or functions that
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
        flush_columns: The data is written after a column is finished
            independent of the passed time and write period.
        additional_setpoints: A list of setpoint parameters to be registered in
            the measurement but not scanned.
        do_plot: should png and pdf versions of the images be saved after the
            run.

    Returns:
        The QCoDeS dataset.
    """

    meas = Measurement()
    all_setpoint_params = (param_set1, param_set2,) + tuple(
            s for s in additional_setpoints)


    measured_parameters = tuple(param for param in param_meas
                                if isinstance(param, _BaseParameter))

    try:
        loop_shape = tuple(
            1 for _ in additional_setpoints
        ) + (num_points1, num_points2)
        shapes: Shapes = detect_shape_of_measurement(
            measured_parameters,
            loop_shape
        )
    except TypeError:
        LOG.exception(
            f"Could not detect shape of {measured_parameters} "
            f"falling back to unknown shape.")
        shapes = None

    _register_parameters(meas, all_setpoint_params)
    _register_parameters(meas, param_meas, setpoints=all_setpoint_params,
                         shapes=shapes)
    _set_write_period(meas, write_period)
    _register_actions(meas, enter_actions, exit_actions)

    param_set1.post_delay = delay1
    param_set2.post_delay = delay2

    with _catch_keyboard_interrupts() as interrupted, meas.run() as datasaver:
        additional_setpoints_data = _process_params_meas(additional_setpoints)
        for set_point1 in np.linspace(start1, stop1, num_points1):
            if set_before_sweep:
                param_set2.set(start2)

            param_set1.set(set_point1)
            for action in before_inner_actions:
                action()
            for set_point2 in np.linspace(start2, stop2, num_points2):
                # skip first inner set point if `set_before_sweep`
                if set_point2 == start2 and set_before_sweep:
                    pass
                else:
                    param_set2.set(set_point2)

                datasaver.add_result((param_set1, set_point1),
                                     (param_set2, set_point2),
                                     *_process_params_meas(param_meas),
                                     *additional_setpoints_data)
            for action in after_inner_actions:
                action()
            if flush_columns:
                datasaver.flush_data_to_database()
        dataset = datasaver.dataset
    return _handle_plotting(dataset, do_plot, interrupted())


def _handle_plotting(
        data: DataSet,
        do_plot: bool = True,
        interrupted: bool = False) -> AxesTupleListWithDataSet:
    """
    Save the plots created by datasaver as pdf and png

    Args:
        datasaver: a measurement datasaver that contains a dataset to be saved
            as plot.
            :param do_plot:

    """
    if do_plot:
        res = plot(data)
    else:
        res = data, [None], [None]

    if interrupted:
        raise KeyboardInterrupt

    return res


def plot(data: DataSet, save_pdf: bool = True,
         save_png: bool = True) -> Tuple[DataSet, list, list]:
    """
    The utility function to plot results and save the figures either in pdf or
    png or both formats.

    Args:
        data: The QCoDeS dataset to be plotted.
        save_pdf: Save figure in pdf format.
        save_png: Save figure in png format.
    """
    dataid = data.run_id
    axes, cbs = plot_dataset(data)
    mainfolder = config.user.mainfolder
    experiment_name = data.exp_name
    sample_name = data.sample_name
    storage_dir = os.path.join(mainfolder, experiment_name, sample_name)
    os.makedirs(storage_dir, exist_ok=True)
    png_dir = os.path.join(storage_dir, 'png')
    pdf_dif = os.path.join(storage_dir, 'pdf')
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dif, exist_ok=True)
    for i, ax in enumerate(axes):
        if save_pdf:
            full_path = os.path.join(pdf_dif, f'{dataid}_{i}.pdf')
            ax.figure.savefig(full_path, dpi=500)
        if save_png:
            full_path = os.path.join(png_dir, f'{dataid}_{i}.png')
            ax.figure.savefig(full_path, dpi=500)
    res = data, axes, cbs
    return res
