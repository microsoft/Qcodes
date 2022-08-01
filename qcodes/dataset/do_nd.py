import logging
import sys
import time
from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import matplotlib.axes
import matplotlib.colorbar
import numpy as np
from tqdm.auto import tqdm
from typing_extensions import TypedDict

from qcodes import config
from qcodes.dataset.data_set_protocol import DataSetProtocol, res_type
from qcodes.dataset.descriptions.detect_shapes import detect_shape_of_measurement
from qcodes.dataset.descriptions.versioning.rundescribertypes import Shapes
from qcodes.dataset.experiment_container import Experiment
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.plotting import plot_and_save_image
from qcodes.parameters import ParameterBase
from .threading import (
    SequentialParamsCaller,
    ThreadPoolParamsCaller,
    process_params_meas,
)

ActionsT = Sequence[Callable[[], None]]
BreakConditionT = Callable[[], bool]

ParamMeasT = Union[ParameterBase, Callable[[], None]]

AxesTuple = Tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar]
AxesTupleList = Tuple[
    List[matplotlib.axes.Axes], List[Optional[matplotlib.colorbar.Colorbar]]
]
AxesTupleListWithDataSet = Tuple[
    DataSetProtocol,
    List[matplotlib.axes.Axes],
    List[Optional[matplotlib.colorbar.Colorbar]],
]
MultiAxesTupleListWithDataSet = Tuple[
    Tuple[DataSetProtocol, ...],
    Tuple[List[matplotlib.axes.Axes], ...],
    Tuple[List[Optional[matplotlib.colorbar.Colorbar]], ...],
]

LOG = logging.getLogger(__name__)


class ParameterGroup(TypedDict):
    params: Tuple[ParamMeasT, ...]
    meas_name: str
    measured_params: List[res_type]


class UnsafeThreadingException(Exception):
    pass


class BreakConditionInterrupt(Exception):
    pass


MeasInterruptT = Union[KeyboardInterrupt, BreakConditionInterrupt, None]


def _register_parameters(
    meas: Measurement,
    param_meas: Sequence[ParamMeasT],
    setpoints: Optional[Sequence[ParameterBase]] = None,
    shapes: Shapes = None,
) -> None:
    for parameter in param_meas:
        if isinstance(parameter, ParameterBase):
            meas.register_parameter(parameter, setpoints=setpoints)
    meas.set_shapes(shapes=shapes)


def _register_actions(
    meas: Measurement, enter_actions: ActionsT, exit_actions: ActionsT
) -> None:
    for action in enter_actions:
        # this omits the possibility of passing
        # argument to enter and exit actions.
        # Do we want that?
        meas.add_before_run(action, ())
    for action in exit_actions:
        meas.add_after_run(action, ())


def _set_write_period(meas: Measurement, write_period: Optional[float] = None) -> None:
    if write_period is not None:
        meas.write_period = write_period


@contextmanager
def _catch_interrupts() -> Iterator[Callable[[], MeasInterruptT]]:
    interrupt_exception = None

    def get_interrupt_exception() -> MeasInterruptT:
        nonlocal interrupt_exception
        return interrupt_exception

    try:
        yield get_interrupt_exception
    except (KeyboardInterrupt, BreakConditionInterrupt) as e:
        interrupt_exception = e


def do0d(
    *param_meas: ParamMeasT,
    write_period: Optional[float] = None,
    measurement_name: str = "",
    exp: Optional[Experiment] = None,
    do_plot: Optional[bool] = None,
    use_threads: Optional[bool] = None,
    log_info: Optional[str] = None,
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
        do_plot = config.dataset.dond_plot
    meas = Measurement(name=measurement_name, exp=exp)
    if log_info is not None:
        meas._extra_log_info = log_info
    else:
        meas._extra_log_info = "Using 'qcodes.dataset.do0d'"

    measured_parameters = tuple(
        param for param in param_meas if isinstance(param, ParameterBase)
    )

    try:
        shapes: Shapes = detect_shape_of_measurement(
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


def do1d(
    param_set: ParameterBase,
    start: float,
    stop: float,
    num_points: int,
    delay: float,
    *param_meas: ParamMeasT,
    enter_actions: ActionsT = (),
    exit_actions: ActionsT = (),
    write_period: Optional[float] = None,
    measurement_name: str = "",
    exp: Optional[Experiment] = None,
    do_plot: Optional[bool] = None,
    use_threads: Optional[bool] = None,
    additional_setpoints: Sequence[ParameterBase] = tuple(),
    show_progress: Optional[None] = None,
    log_info: Optional[str] = None,
    break_condition: Optional[BreakConditionT] = None,
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
        do_plot = config.dataset.dond_plot
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
        shapes: Shapes = detect_shape_of_measurement(measured_parameters, loop_shape)
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
    with _catch_interrupts() as interrupted, meas.run() as datasaver, param_meas_caller as call_param_meas:
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
    set_before_sweep: Optional[bool] = True,
    enter_actions: ActionsT = (),
    exit_actions: ActionsT = (),
    before_inner_actions: ActionsT = (),
    after_inner_actions: ActionsT = (),
    write_period: Optional[float] = None,
    measurement_name: str = "",
    exp: Optional[Experiment] = None,
    flush_columns: bool = False,
    do_plot: Optional[bool] = None,
    use_threads: Optional[bool] = None,
    additional_setpoints: Sequence[ParameterBase] = tuple(),
    show_progress: Optional[None] = None,
    log_info: Optional[str] = None,
    break_condition: Optional[BreakConditionT] = None,
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
        do_plot = config.dataset.dond_plot
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
        shapes: Shapes = detect_shape_of_measurement(measured_parameters, loop_shape)
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

    with _catch_interrupts() as interrupted, meas.run() as datasaver, param_meas_caller as call_param_meas:
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


class AbstractSweep(ABC):
    """
    Abstract sweep class that defines an interface for concrete sweep classes.
    """

    @abstractmethod
    def get_setpoints(self) -> np.ndarray:
        """
        Returns an array of setpoint values for this sweep.
        """
        pass

    @property
    @abstractmethod
    def param(self) -> ParameterBase:
        """
        Returns the Qcodes sweep parameter.
        """
        pass

    @property
    @abstractmethod
    def delay(self) -> float:
        """
        Delay between two consecutive sweep points.
        """
        pass

    @property
    @abstractmethod
    def num_points(self) -> int:
        """
        Number of sweep points.
        """
        pass

    @property
    @abstractmethod
    def post_actions(self) -> ActionsT:
        """
        actions to be performed after setting param to its setpoint.
        """
        pass


class LinSweep(AbstractSweep):
    """
    Linear sweep.

    Args:
        param: Qcodes parameter to sweep.
        start: Sweep start value.
        stop: Sweep end value.
        num_points: Number of sweep points.
        delay: Time in seconds between two consequtive sweep points
    """

    def __init__(
        self,
        param: ParameterBase,
        start: float,
        stop: float,
        num_points: int,
        delay: float = 0,
        post_actions: ActionsT = (),
    ):
        self._param = param
        self._start = start
        self._stop = stop
        self._num_points = num_points
        self._delay = delay
        self._post_actions = post_actions

    def get_setpoints(self) -> np.ndarray:
        """
        Linear (evenly spaced) numpy array for supplied start, stop and
        num_points.
        """
        return np.linspace(self._start, self._stop, self._num_points)

    @property
    def param(self) -> ParameterBase:
        return self._param

    @property
    def delay(self) -> float:
        return self._delay

    @property
    def num_points(self) -> int:
        return self._num_points

    @property
    def post_actions(self) -> ActionsT:
        return self._post_actions


class LogSweep(AbstractSweep):
    """
    Logarithmic sweep.

    Args:
        param: Qcodes parameter for sweep.
        start: Sweep start value.
        stop: Sweep end value.
        num_points: Number of sweep points.
        delay: Time in seconds between two consequtive sweep points.
    """

    def __init__(
        self,
        param: ParameterBase,
        start: float,
        stop: float,
        num_points: int,
        delay: float = 0,
        post_actions: ActionsT = (),
    ):
        self._param = param
        self._start = start
        self._stop = stop
        self._num_points = num_points
        self._delay = delay
        self._post_actions = post_actions

    def get_setpoints(self) -> np.ndarray:
        """
        Logarithmically spaced numpy array for supplied start, stop and
        num_points.
        """
        return np.logspace(self._start, self._stop, self._num_points)

    @property
    def param(self) -> ParameterBase:
        return self._param

    @property
    def delay(self) -> float:
        return self._delay

    @property
    def num_points(self) -> int:
        return self._num_points

    @property
    def post_actions(self) -> ActionsT:
        return self._post_actions


class ArraySweep(AbstractSweep):
    """
    Sweep the values of a given array.

    Args:
        param: Qcodes parameter for sweep.
        array: array with values to sweep.
        delay: Time in seconds between two consecutive sweep points.
        post_actions: Actions to do after each sweep point.
    """

    def __init__(
        self,
        param: ParameterBase,
        array: Union[Sequence[float], np.ndarray],
        delay: float = 0,
        post_actions: ActionsT = (),
    ):
        self._param = param
        self._array = np.array(array)
        self._delay = delay
        self._post_actions = post_actions

    def get_setpoints(self) -> np.ndarray:
        return self._array

    @property
    def param(self) -> ParameterBase:
        return self._param

    @property
    def delay(self) -> float:
        return self._delay

    @property
    def num_points(self) -> int:
        return len(self._array)

    @property
    def post_actions(self) -> ActionsT:
        return self._post_actions


class _Sweeper:
    def __init__(
        self,
        sweeps: Sequence[AbstractSweep],
        additional_setpoints: Sequence[ParameterBase],
    ):
        self._sweeps = sweeps
        self._additional_setpoints = additional_setpoints
        self._nested_setpoints = self._make_nested_setpoints()

        self._post_delays = tuple(sweep.delay for sweep in sweeps)
        self._params_set = tuple(sweep.param for sweep in sweeps)
        self._post_actions = tuple(sweep.post_actions for sweep in sweeps)

    def _make_nested_setpoints(self) -> np.ndarray:
        """Create the cartesian product of all the setpoint values."""
        if len(self._sweeps) == 0:
            return np.array([[]])  # 0d sweep (do0d)
        setpoint_values = [sweep.get_setpoints() for sweep in self._sweeps]
        return self._flatten_setpoint_values(setpoint_values)

    @staticmethod
    def _flatten_setpoint_values(setpoint_values: Sequence[np.ndarray]) -> np.ndarray:
        setpoint_grids = np.meshgrid(*setpoint_values, indexing="ij")
        flat_setpoint_grids = [np.ravel(grid, order="C") for grid in setpoint_grids]
        return np.vstack(flat_setpoint_grids).T

    @property
    def nested_setpoints(self) -> np.ndarray:
        return self._nested_setpoints

    @property
    def all_setpoint_params(self) -> Tuple[ParameterBase, ...]:
        return tuple(sweep.param for sweep in self._sweeps) + tuple(
            s for s in self._additional_setpoints
        )

    @property
    def shape(self) -> Tuple[int, ...]:
        loop_shape = tuple(sweep.num_points for sweep in self._sweeps) + tuple(
            1 for _ in self._additional_setpoints
        )
        return loop_shape

    @property
    def post_delays(self) -> Tuple[float, ...]:
        return self._post_delays

    @property
    def params_set(self) -> Tuple[ParameterBase, ...]:
        return self._params_set

    @property
    def post_actions(self) -> Tuple[ActionsT, ...]:
        return self._post_actions


class _Measurements:
    def __init__(
        self,
        measurement_name: str,
        params_meas: Sequence[Union[ParamMeasT, Sequence[ParamMeasT]]],
    ):
        (
            self._measured_all,
            self._grouped_parameters,
            self._measured_parameters,
        ) = _extract_paramters_by_type_and_group(measurement_name, params_meas)

    @property
    def measured_all(self) -> Tuple[ParamMeasT, ...]:
        return self._measured_all

    @property
    def grouped_parameters(self) -> Dict[str, ParameterGroup]:
        return self._grouped_parameters

    @property
    def measured_parameters(self) -> Tuple[ParameterBase, ...]:
        return self._measured_parameters


def dond(
    *params: Union[AbstractSweep, Union[ParamMeasT, Sequence[ParamMeasT]]],
    write_period: Optional[float] = None,
    measurement_name: str = "",
    exp: Optional[Union[Experiment, Sequence[Experiment]]] = None,
    enter_actions: ActionsT = (),
    exit_actions: ActionsT = (),
    do_plot: Optional[bool] = None,
    show_progress: Optional[bool] = None,
    use_threads: Optional[bool] = None,
    additional_setpoints: Sequence[ParameterBase] = tuple(),
    log_info: Optional[str] = None,
    break_condition: Optional[BreakConditionT] = None,
) -> Union[AxesTupleListWithDataSet, MultiAxesTupleListWithDataSet]:
    """
    Perform n-dimentional scan from slowest (first) to the fastest (last), to
    measure m measurement parameters. The dimensions should be specified
    as sweep objects, and after them the parameters to measure should be passed.

    Args:
        params: Instances of n sweep classes and m measurement parameters,
            e.g. if linear sweep is considered:

            .. code-block::

                LinSweep(param_set_1, start_1, stop_1, num_points_1, delay_1), ...,
                LinSweep(param_set_n, start_n, stop_n, num_points_n, delay_n),
                param_meas_1, param_meas_2, ..., param_meas_m

            If multiple DataSets creation is needed, measurement parameters should
            be grouped, so one dataset will be created for each group. e.g.:

            .. code-block::

                LinSweep(param_set_1, start_1, stop_1, num_points_1, delay_1), ...,
                LinSweep(param_set_n, start_n, stop_n, num_points_n, delay_n),
                [param_meas_1, param_meas_2], ..., [param_meas_m]

        write_period: The time after which the data is actually written to the
            database.
        measurement_name: Name of the measurement. This will be passed down to
            the dataset produced by the measurement. If not given, a default
            value of 'results' is used for the dataset.
        exp: The experiment to use for this measurement. If you create multiple
            measurements using groups you may also supply multiple experiments.
        enter_actions: A list of functions taking no arguments that will be
            called before the measurements start.
        exit_actions: A list of functions taking no arguments that will be
            called after the measurements ends.
        do_plot: should png and pdf versions of the images be saved and plots
            are shown after the run. If None the setting will be read from
            ``qcodesrc.json``
        show_progress: should a progress bar be displayed during the
            measurement. If None the setting will be read from ``qcodesrc.json``
        use_threads: If True, measurements from each instrument will be done on
            separate threads. If you are measuring from several instruments
            this may give a significant speedup.
        additional_setpoints: A list of setpoint parameters to be registered in
            the measurement but not scanned/swept-over.
        log_info: Message that is logged during the measurement. If None a default
            message is used.
        break_condition: Callable that takes no arguments. If returned True,
            measurement is interrupted.

    Returns:
        A tuple of QCoDeS DataSet, Matplotlib axis, Matplotlib colorbar. If
        more than one group of measurement parameters is supplied, the output
        will be a tuple of tuple(QCoDeS DataSet), tuple(Matplotlib axis),
        tuple(Matplotlib colorbar), in which each element of each sub-tuple
        belongs to one group, and the order of elements is the order of
        the supplied groups.
    """
    if do_plot is None:
        do_plot = config.dataset.dond_plot
    if show_progress is None:
        show_progress = config.dataset.dond_show_progress

    sweep_instances, params_meas = _parse_dond_arguments(*params)

    sweeper = _Sweeper(sweep_instances, additional_setpoints)

    measurements = _Measurements(measurement_name, params_meas)

    LOG.info(
        "Starting a doNd with scan with\n setpoints: %s,\n measuring: %s",
        sweeper.all_setpoint_params,
        measurements.measured_all,
    )
    LOG.debug(
        "Measured parameters have been grouped into:\n " "%s",
        {
            name: group["params"]
            for name, group in measurements.grouped_parameters.items()
        },
    )
    try:
        loop_shape = sweeper.shape
        shapes: Shapes = detect_shape_of_measurement(
            measurements.measured_parameters, loop_shape
        )
        LOG.debug("Detected shapes to be %s", shapes)
    except TypeError:
        LOG.exception(
            f"Could not detect shape of {measurements.measured_parameters} "
            f"falling back to unknown shape."
        )
        shapes = None
    meas_list = _create_measurements(
        sweeper.all_setpoint_params,
        enter_actions,
        exit_actions,
        exp,
        measurements.grouped_parameters,
        shapes,
        write_period,
        log_info,
    )

    datasets = []
    plots_axes = []
    plots_colorbar = []
    if use_threads is None:
        use_threads = config.dataset.use_threads

    params_meas_caller = (
        ThreadPoolParamsCaller(*measurements.measured_all)
        if use_threads
        else SequentialParamsCaller(*measurements.measured_all)
    )

    try:
        with _catch_interrupts() as interrupted, ExitStack() as stack, params_meas_caller as call_params_meas:
            datasavers = [stack.enter_context(measure.run()) for measure in meas_list]
            additional_setpoints_data = process_params_meas(additional_setpoints)
            previous_setpoints = np.empty(len(sweep_instances))
            for setpoints in tqdm(sweeper.nested_setpoints, disable=not show_progress):

                active_actions, delays = _select_active_actions_delays(
                    sweeper.post_actions,
                    sweeper.post_delays,
                    setpoints,
                    previous_setpoints,
                )
                previous_setpoints = setpoints

                param_set_list = []
                for setpoint_param, setpoint, action, delay in zip(
                    sweeper.params_set,
                    setpoints,
                    active_actions,
                    delays,
                ):
                    _conditional_parameter_set(setpoint_param, setpoint)
                    param_set_list.append((setpoint_param, setpoint))
                    for act in action:
                        act()
                    time.sleep(delay)

                meas_value_pair = call_params_meas()
                for group in measurements.grouped_parameters.values():
                    group["measured_params"] = []
                    for measured in meas_value_pair:
                        if measured[0] in group["params"]:
                            group["measured_params"].append(measured)
                for ind, datasaver in enumerate(datasavers):
                    datasaver.add_result(
                        *param_set_list,
                        *measurements.grouped_parameters[f"group_{ind}"][
                            "measured_params"
                        ],
                        *additional_setpoints_data,
                    )

                if callable(break_condition):
                    if break_condition():
                        raise BreakConditionInterrupt("Break condition was met.")
    finally:

        for datasaver in datasavers:
            ds, plot_axis, plot_color = _handle_plotting(
                datasaver.dataset, do_plot, interrupted()
            )
            datasets.append(ds)
            plots_axes.append(plot_axis)
            plots_colorbar.append(plot_color)

    if len(measurements.grouped_parameters) == 1:
        return datasets[0], plots_axes[0], plots_colorbar[0]
    else:
        return tuple(datasets), tuple(plots_axes), tuple(plots_colorbar)


def _parse_dond_arguments(
    *params: Union[AbstractSweep, Union[ParamMeasT, Sequence[ParamMeasT]]]
) -> Tuple[List[AbstractSweep], List[Union[ParamMeasT, Sequence[ParamMeasT]]]]:
    """
    Parse supplied arguments into sweep objects and measurement parameters
    and their callables.
    """
    sweep_instances: List[AbstractSweep] = []
    params_meas: List[Union[ParamMeasT, Sequence[ParamMeasT]]] = []
    for par in params:
        if isinstance(par, AbstractSweep):
            sweep_instances.append(par)
        else:
            params_meas.append(par)
    return sweep_instances, params_meas


def _conditional_parameter_set(
    parameter: ParameterBase,
    value: Union[float, complex],
) -> None:
    """
    Reads the cache value of the given parameter and set the parameter to
    the given value if the value is different from the cache value.
    """
    if value != parameter.cache.get():
        parameter.set(value)


def _make_nested_setpoints(sweeps: Sequence[AbstractSweep]) -> np.ndarray:
    """Create the cartesian product of all the setpoint values."""
    if len(sweeps) == 0:
        return np.array([[]])  # 0d sweep (do0d)
    setpoint_values = [sweep.get_setpoints() for sweep in sweeps]
    setpoint_grids = np.meshgrid(*setpoint_values, indexing="ij")
    flat_setpoint_grids = [np.ravel(grid, order="C") for grid in setpoint_grids]
    return np.vstack(flat_setpoint_grids).T


def _select_active_actions_delays(
    actions: Sequence[ActionsT],
    delays: Sequence[float],
    setpoints: np.ndarray,
    previous_setpoints: np.ndarray,
) -> Tuple[List[ActionsT], List[float]]:
    """
    Select ActionT (Sequence[Callable]) and delays(Sequence[float]) from
    a Sequence of ActionsT and delays, respectively, if the corresponding
    setpoint has changed. Otherwise, select an empty Sequence for actions
    and zero for delays.
    """
    actions_list: List[ActionsT] = [()] * len(setpoints)
    setpoints_delay: List[float] = [0] * len(setpoints)
    for ind, (new_setpoint, old_setpoint) in enumerate(
        zip(setpoints, previous_setpoints)
    ):
        if new_setpoint != old_setpoint:
            actions_list[ind] = actions[ind]
            setpoints_delay[ind] = delays[ind]
    return (actions_list, setpoints_delay)


def _create_measurements(
    all_setpoint_params: Sequence[ParameterBase],
    enter_actions: ActionsT,
    exit_actions: ActionsT,
    experiments: Optional[Union[Experiment, Sequence[Experiment]]],
    grouped_parameters: Mapping[str, ParameterGroup],
    shapes: Shapes,
    write_period: Optional[float],
    log_info: Optional[str],
) -> Tuple[Measurement, ...]:
    meas_list: List[Measurement] = []
    if log_info is not None:
        _extra_log_info = log_info
    else:
        _extra_log_info = "Using 'qcodes.dataset.dond'"

    if not isinstance(experiments, Sequence):
        experiments_internal: Sequence[Optional[Experiment]] = [
            experiments for _ in grouped_parameters
        ]
    else:
        experiments_internal = experiments

    if len(experiments_internal) != len(grouped_parameters):
        raise ValueError(
            f"Inconsistent number of "
            f"parameter groups and experiments "
            f"got {len(grouped_parameters)} and {len(experiments_internal)}"
        )

    for group, exp in zip(grouped_parameters.values(), experiments_internal):
        meas_name = group["meas_name"]
        meas_params = group["params"]
        meas = Measurement(name=meas_name, exp=exp)
        meas._extra_log_info = _extra_log_info
        _register_parameters(meas, all_setpoint_params)
        _register_parameters(
            meas, meas_params, setpoints=all_setpoint_params, shapes=shapes
        )
        _set_write_period(meas, write_period)
        _register_actions(meas, enter_actions, exit_actions)
        meas_list.append(meas)
    return tuple(meas_list)


def _extract_paramters_by_type_and_group(
    measurement_name: str,
    params_meas: Sequence[Union[ParamMeasT, Sequence[ParamMeasT]]],
) -> Tuple[
    Tuple[ParamMeasT, ...], Dict[str, ParameterGroup], Tuple[ParameterBase, ...]
]:
    measured_parameters: List[ParameterBase] = []
    measured_all: List[ParamMeasT] = []
    single_group: List[ParamMeasT] = []
    multi_group: List[Sequence[ParamMeasT]] = []
    grouped_parameters: Dict[str, ParameterGroup] = {}
    for param in params_meas:
        if not isinstance(param, Sequence):
            single_group.append(param)
            measured_all.append(param)
            if isinstance(param, ParameterBase):
                measured_parameters.append(param)
        elif not isinstance(param, str):
            multi_group.append(param)
            for nested_param in param:
                measured_all.append(nested_param)
                if isinstance(nested_param, ParameterBase):
                    measured_parameters.append(nested_param)
    if single_group:
        pg: ParameterGroup = {
            "params": tuple(single_group),
            "meas_name": measurement_name,
            "measured_params": [],
        }
        grouped_parameters["group_0"] = pg
    if multi_group:
        for index, par in enumerate(multi_group):
            pg = {
                "params": tuple(par),
                "meas_name": measurement_name,
                "measured_params": [],
            }
            grouped_parameters[f"group_{index}"] = pg
    return tuple(measured_all), grouped_parameters, tuple(measured_parameters)


def _handle_plotting(
    data: DataSetProtocol,
    do_plot: bool = True,
    interrupted: MeasInterruptT = None,
) -> AxesTupleListWithDataSet:
    """
    Save the plots created by datasaver as pdf and png

    Args:
        datasaver: a measurement datasaver that contains a dataset to be saved
            as plot.
            :param do_plot:

    """
    if do_plot:
        res = plot_and_save_image(data)
    else:
        res = data, [None], [None]

    if interrupted:
        raise interrupted

    return res
