from typing import Callable, Sequence, Union, Tuple, List, Optional
import os
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from qcodes.dataset.measurements import Measurement
from qcodes.instrument.base import _BaseParameter
from qcodes.dataset.plotting import plot_by_id
from qcodes import config

AxesTuple = Tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar]
AxesTupleList = Tuple[List[matplotlib.axes.Axes],
                      List[Optional[matplotlib.colorbar.Colorbar]]]
AxesTupleListWithRunId = Tuple[int, List[matplotlib.axes.Axes],
                      List[Optional[matplotlib.colorbar.Colorbar]]]
number = Union[float, int]


def do0d(*param_meas:  Union[_BaseParameter, Callable[[], None]],
         do_plot: bool = True) -> AxesTupleListWithRunId:
    """
    Perform a measurement of a single parameter. This is probably most
    useful for an ArrayParamter that already returns an array of data points

    Args:
        *param_meas: Parameter(s) to measure at each step or functions that
          will be called at each step. The function should take no arguments.
          The parameters and functions are called in the order they are
          supplied.
        do_plot: should png and pdf versions of the images be saved after the
            run.

    Returns:
        The run_id of the DataSet created
    """
    meas = Measurement()
    output = []

    for parameter in param_meas:
        meas.register_parameter(parameter)
        output.append([parameter, None])

    with meas.run() as datasaver:

        for i, parameter in enumerate(param_meas):
            if isinstance(parameter, _BaseParameter):
                output[i][1] = parameter.get()
            elif callable(parameter):
                parameter()
        datasaver.add_result(*output)
    dataid = datasaver.run_id

    if do_plot is True:
        ax, cbs = _save_image(datasaver)
    else:
        ax = None,
        cbs = None

    return dataid, ax, cbs


def do1d(param_set: _BaseParameter, start: number, stop: number,
         num_points: int, delay: number,
         *param_meas: Union[_BaseParameter, Callable[[], None]],
         enter_actions: Sequence[Callable[[], None]] = (),
         exit_actions: Sequence[Callable[[], None]] = (),
         do_plot: bool = True) \
        -> AxesTupleListWithRunId:
    """
    Perform a 1D scan of ``param_set`` from ``start`` to ``stop`` in
    ``num_points`` measuring param_meas at each step. In case param_meas is
    an ArrayParameter this is effectively a 2d scan.

    Args:
        param_set: The QCoDeS parameter to sweep over
        start: Starting point of sweep
        stop: End point of sweep
        num_points: Number of points in sweep
        delay: Delay after setting paramter before measurement is performed
        *param_meas: Parameter(s) to measure at each step or functions that
          will be called at each step. The function should take no arguments.
          The parameters and functions are called in the order they are
          supplied.
        enter_actions: A list of functions taking no arguments that will be
            called before the measurements start
        exit_actions: A list of functions taking no arguments that will be
            called after the measurements ends
        do_plot: should png and pdf versions of the images be saved after the
            run.

    Returns:
        The run_id of the DataSet created
    """
    meas = Measurement()
    meas.register_parameter(
        param_set)  # register the first independent parameter
    output = []
    param_set.post_delay = delay
    interrupted = False

    for action in enter_actions:
        # this omits the posibility of passing
        # argument to enter and exit actions.
        # Do we want that?
        meas.add_before_run(action, ())
    for action in exit_actions:
        meas.add_after_run(action, ())

    # do1D enforces a simple relationship between measured parameters
    # and set parameters. For anything more complicated this should be
    # reimplemented from scratch
    for parameter in param_meas:
        if isinstance(parameter, _BaseParameter):
            meas.register_parameter(parameter, setpoints=(param_set,))
            output.append([parameter, None])

    try:
        with meas.run() as datasaver:

            for set_point in np.linspace(start, stop, num_points):
                param_set.set(set_point)
                output = []
                for parameter in param_meas:
                    if isinstance(parameter, _BaseParameter):
                        output.append((parameter, parameter.get()))
                    elif callable(parameter):
                        parameter()
                datasaver.add_result((param_set, set_point),
                                      *output)
    except KeyboardInterrupt:
        interrupted = True

    dataid = datasaver.run_id  # convenient to have for plotting

    if do_plot is True:
        ax, cbs = _save_image(datasaver)
    else:
        ax = None,
        cbs = None

    if interrupted:
        raise KeyboardInterrupt
    return dataid, ax, cbs


def do2d(param_set1: _BaseParameter, start1: number, stop1: number,
         num_points1: int, delay1: number,
         param_set2: _BaseParameter, start2: number, stop2: number,
         num_points2: int, delay2: number,
         *param_meas: Union[_BaseParameter, Callable[[], None]],
         set_before_sweep: Optional[bool] = False,
         enter_actions: Sequence[Callable[[], None]] = (),
         exit_actions: Sequence[Callable[[], None]] = (),
         before_inner_actions: Sequence[Callable[[], None]] = (),
         after_inner_actions: Sequence[Callable[[], None]] = (),
         do_plot: bool=True) -> AxesTupleListWithRunId:

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
        delay2: Delay after setting paramter before measurement is performed
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
        do_plot: should png and pdf versions of the images be saved after the
            run.

    Returns:
        The run_id of the DataSet created
    """

    meas = Measurement()
    meas.register_parameter(param_set1)
    param_set1.post_delay = delay1
    meas.register_parameter(param_set2)
    param_set2.post_delay = delay2
    interrupted = False
    for action in enter_actions:
        # this omits the possibility of passing
        # argument to enter and exit actions.
        # Do we want that?
        meas.add_before_run(action, ())

    for action in exit_actions:
        meas.add_after_run(action, ())

    for parameter in param_meas:
        if isinstance(parameter, _BaseParameter):
            meas.register_parameter(parameter,
                                    setpoints=(param_set1, param_set2))
    try:
        with meas.run() as datasaver:
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
                    output = []
                    for parameter in param_meas:
                        if isinstance(parameter, _BaseParameter):
                            output.append((parameter, parameter.get()))
                        elif callable(parameter):
                            parameter()
                    datasaver.add_result((param_set1, set_point1),
                                         (param_set2, set_point2),
                                         *output)
                for action in after_inner_actions:
                    action()
    except KeyboardInterrupt:
        interrupted = True

    dataid = datasaver.run_id

    if do_plot is True:
        ax, cbs = _save_image(datasaver)
    else:
        ax = None,
        cbs = None
    if interrupted:
        raise KeyboardInterrupt

    return dataid, ax, cbs


def _save_image(datasaver) -> AxesTupleList:
    """
    Save the plots created by datasaver as pdf and png

    Args:
        datasaver: a measurement datasaver that contains a dataset to be saved
            as plot.

    """
    plt.ioff()
    dataid = datasaver.run_id
    start = time.time()
    axes, cbs = plot_by_id(dataid)
    stop = time.time()
    print(f"plot by id took {stop-start}")

    mainfolder = config.user.mainfolder
    experiment_name = datasaver._dataset.exp_name
    sample_name = datasaver._dataset.sample_name

    storage_dir = os.path.join(mainfolder, experiment_name, sample_name)
    os.makedirs(storage_dir, exist_ok=True)

    png_dir = os.path.join(storage_dir, 'png')
    pdf_dif = os.path.join(storage_dir, 'pdf')

    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dif, exist_ok=True)

    save_pdf = True
    save_png = True

    for i, ax in enumerate(axes):
        if save_pdf:
            full_path = os.path.join(pdf_dif, f'{dataid}_{i}.pdf')
            ax.figure.savefig(full_path, dpi=500)
        if save_png:
            full_path = os.path.join(png_dir, f'{dataid}_{i}.png')
            ax.figure.savefig(full_path, dpi=500)
    plt.ion()
    return axes, cbs
