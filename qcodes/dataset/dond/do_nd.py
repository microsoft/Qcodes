from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple, Union, cast

import numpy as np
from tqdm.auto import tqdm
from typing_extensions import TypedDict

from qcodes import config
from qcodes.dataset.descriptions.detect_shapes import detect_shape_of_measurement
from qcodes.dataset.descriptions.versioning.rundescribertypes import Shapes
from qcodes.dataset.dond.do_nd_utils import (
    ActionsT,
    AxesTupleListWithDataSet,
    BreakConditionInterrupt,
    BreakConditionT,
    MultiAxesTupleListWithDataSet,
    ParamMeasT,
    _catch_interrupts,
    _handle_plotting,
    _register_actions,
    _register_parameters,
    _set_write_period,
)
from qcodes.dataset.experiment_container import Experiment
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.threading import (
    SequentialParamsCaller,
    ThreadPoolParamsCaller,
    process_params_meas,
)
from qcodes.parameters import ParameterBase

from .sweeps import AbstractSweep

LOG = logging.getLogger(__name__)

SweepVarType = Any


class ParameterGroup(TypedDict):
    params: tuple[ParamMeasT, ...]
    meas_name: str


class MultiSweep:
    def __init__(self, sweeps: Sequence[AbstractSweep]):
        # todo check that all sweeps are the same
        self._sweeps = tuple(sweeps)

    @property
    def sweeps(self) -> tuple[AbstractSweep, ...]:
        return self._sweeps

    def get_setpoints(self) -> Iterable:
        return zip(*(sweep.get_setpoints() for sweep in self.sweeps))

    @property
    def num_points(self) -> int:
        return self.sweeps[0].num_points


@dataclass
class ParameterSetEvent:
    parameter: ParameterBase
    new_value: float
    should_set: bool
    delay: float
    actions: ActionsT


class _Sweeper:
    def __init__(
        self,
        sweeps: Sequence[AbstractSweep | MultiSweep],
        additional_setpoints: Sequence[ParameterBase],
    ):
        self._additional_setpoints = additional_setpoints
        self._sweeps = sweeps
        self._setpoints = self._make_setpoints_tuples()
        self._setpoints_dict = self._make_setpoints_dict()
        self._shapes = self._make_shape(sweeps, additional_setpoints)

    @property
    def setpoint_dicts(self) -> dict[str, list[Any]]:
        return self._setpoints_dict

    def _make_setpoints_tuples(
        self,
    ) -> tuple[tuple[tuple[SweepVarType, ...] | SweepVarType]]:
        sweeps = tuple(sweep.get_setpoints() for sweep in self._sweeps)
        return cast(
            Tuple[Tuple[Union[Tuple[SweepVarType, ...], SweepVarType]]],
            tuple(itertools.product(*sweeps)),
        )

    def _make_single_point_setpoints_dict(self, index: int) -> dict[str, SweepVarType]:

        setpoint_dict = {}
        values = self._make_setpoints_tuples()[index]
        for sweep, subvalues in zip(self._sweeps, values):
            if isinstance(sweep, MultiSweep):
                for individual_sweep, value in zip(sweep.sweeps, subvalues):
                    setpoint_dict[individual_sweep.param.full_name] = value
            else:
                setpoint_dict[sweep.param.full_name] = subvalues
        return setpoint_dict

    def _make_setpoints_dict(self) -> dict[str, list[Any]]:

        setpoint_dict: dict[str, list[SweepVarType]] = {}

        for sweep in self._sweeps:
            if isinstance(sweep, MultiSweep):
                for individual_sweep in sweep.sweeps:
                    setpoint_dict[individual_sweep.param.full_name] = []
            else:
                setpoint_dict[sweep.param.full_name] = []

        for setpoint_tuples in self._setpoints:
            for sweep, values in zip(self._sweeps, setpoint_tuples):
                if isinstance(sweep, MultiSweep):
                    for individual_sweep, individual_value in zip(sweep.sweeps, values):
                        setpoint_dict[individual_sweep.param.full_name].append(
                            individual_value
                        )
                else:
                    setpoint_dict[sweep.param.full_name].append(values)
        return setpoint_dict

    @property
    def all_sweeps(self) -> tuple[AbstractSweep, ...]:
        sweeps: list[AbstractSweep] = []
        for sweep in self._sweeps:
            if isinstance(sweep, MultiSweep):
                sweeps.extend(sweep.sweeps)
            else:
                sweeps.append(sweep)
        return tuple(sweeps)

    @property
    def all_setpoint_params(self) -> tuple[ParameterBase, ...]:
        return tuple(sweep.param for sweep in self.all_sweeps)

    @property
    def param_tuples(self) -> tuple[tuple[ParameterBase, ...], ...]:
        """
        These are all the combinations of setpoints we consider
        valid for setpoints in a dataset. As of now that means
        take one element from each dimension. If that is a multisweep
        pick one of the components.
        """
        param_list: list[tuple[ParameterBase, ...]] = []
        for sweep in self._sweeps:
            if isinstance(sweep, MultiSweep):
                param_list.append(tuple(sub_sweep.param for sub_sweep in sweep.sweeps))
            else:
                param_list.append((sweep.param,))

        param_list.extend([(setpoint,) for setpoint in self._additional_setpoints])
        # looks lite itertools.product is not yet generic in input type
        # so output ends up being tuple[tuple[Any]] even with a specified input type
        param_tuples = cast(
            Tuple[Tuple[ParameterBase, ...]], tuple(itertools.product(*param_list))
        )
        return param_tuples

    @property
    def sweep_groupes(self) -> tuple[tuple[ParameterBase, ...], ...]:
        return self.param_tuples

    @staticmethod
    def _make_shape(
        sweeps: Sequence[AbstractSweep | MultiSweep],
        addtional_setpoints: Sequence[ParameterBase],
    ) -> tuple[int, ...]:
        loop_shape = tuple(sweep.num_points for sweep in sweeps) + tuple(
            1 for _ in addtional_setpoints
        )
        return loop_shape

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shapes

    def __getitem__(self, index: int) -> tuple[ParameterSetEvent, ...]:

        setpoints = self._make_single_point_setpoints_dict(index)

        if index == 0:
            previous_setpoints: dict[str, SweepVarType | None] = {}
            for key in setpoints.keys():
                previous_setpoints[key] = None
        else:
            previous_setpoints = self._make_single_point_setpoints_dict(index - 1)

        sweeps = self.all_sweeps

        parameter_set_events = []

        for sweep, new_value, old_value in zip(
            sweeps, setpoints.values(), previous_setpoints.values()
        ):
            if old_value is None or old_value != new_value:
                should_set = True
            else:
                should_set = False
            event = ParameterSetEvent(
                new_value=new_value,
                parameter=sweep.param,
                should_set=should_set,
                delay=sweep.delay,
                actions=sweep.post_actions,
            )
            parameter_set_events.append(event)
        return tuple(parameter_set_events)


class _Measurements:
    def __init__(
        self,
        measurement_name: str,
        params_meas: Sequence[ParamMeasT | Sequence[ParamMeasT]],
    ):
        (
            self._measured_all,
            self._grouped_parameters,
            self._measured_parameters,
        ) = self._extract_parameters_by_type_and_group(params_meas)

    @property
    def measured_all(self) -> tuple[ParamMeasT, ...]:
        return self._measured_all

    @property
    def grouped_parameters(self) -> list[list[ParamMeasT]]:
        return self._grouped_parameters

    @property
    def measured_parameters(self) -> tuple[ParameterBase, ...]:
        return self._measured_parameters

    @staticmethod
    def _extract_parameters_by_type_and_group(
        params_meas: Sequence[ParamMeasT | Sequence[ParamMeasT]],
    ) -> tuple[
        tuple[ParamMeasT, ...], list[list[ParamMeasT]], tuple[ParameterBase, ...]
    ]:
        measured_parameters: list[ParameterBase] = []
        measured_all: list[ParamMeasT] = []
        single_group: list[ParamMeasT] = []
        multi_group: list[list[ParamMeasT]] = []
        grouped_parameters: list[list[ParamMeasT]] = []
        for param in params_meas:
            if not isinstance(param, Sequence):
                single_group.append(param)
                measured_all.append(param)
                if isinstance(param, ParameterBase):
                    measured_parameters.append(param)
            elif not isinstance(param, str):
                multi_group.append(list(param))
                for nested_param in param:
                    measured_all.append(nested_param)
                    if isinstance(nested_param, ParameterBase):
                        measured_parameters.append(nested_param)
        if single_group:
            grouped_parameters = [single_group]
        if multi_group:
            grouped_parameters = multi_group
        return tuple(measured_all), grouped_parameters, tuple(measured_parameters)


# idealy we would want this to be frozen but then postinit cannot calulate all the parameters
# https://stackoverflow.com/questions/53756788/
@dataclass(frozen=False)
class _SweapMeasGroup:
    sweep_parameters: tuple[ParameterBase, ...]
    measure_parameters: tuple[
        ParamMeasT, ...
    ]  # todo should this be all or only Parameters?
    experiment: Experiment | None
    measurement_cxt: Measurement

    def __post_init__(self) -> None:
        meas_parameters = tuple(
            a for a in self.measure_parameters if isinstance(a, ParameterBase)
        )
        self._parameters = self.sweep_parameters + meas_parameters

    @property
    def parameters(self) -> tuple[ParameterBase, ...]:
        return self._parameters


class _SweeperMeasure:
    def __init__(
        self,
        sweeper: _Sweeper,
        measurements: _Measurements,
        enter_actions: ActionsT,
        exit_actions: ActionsT,
        experiments: Experiment | Sequence[Experiment] | None,
        write_period: float | None,
        log_info: str | None,
        dataset_mapping: Sequence[
            tuple[tuple[ParameterBase, ...], Sequence[ParamMeasT]]
        ]
        | None = None,
    ):

        self._sweeper = sweeper
        self._measurements = measurements
        self._enter_actions = enter_actions
        self._exit_actions = exit_actions
        self._experiments = self._get_experiments(experiments)
        self._write_period = write_period
        self._log_info = log_info
        self._dataset_mapping = dataset_mapping
        self._shapes = self._get_shape()
        self._groups = self._create_groups()

        if log_info is not None:
            self._extra_log_info = log_info
        else:
            self._extra_log_info = "Using 'qcodes.dataset.dond'"

    def _get_shape(self) -> Shapes | None:
        try:
            shapes: Shapes = detect_shape_of_measurement(
                self._measurements.measured_parameters, self._sweeper.shape
            )
            LOG.debug("Detected shapes to be %s", shapes)
        except TypeError:
            LOG.exception(
                f"Could not detect shape of {self._measurements.measured_parameters} "
                f"falling back to unknown shape."
            )
            shapes = None
        return shapes

    def _get_experiments(
        self, experiments: Experiment | Sequence[Experiment] | None
    ) -> Sequence[Experiment | None]:
        if not isinstance(experiments, Sequence):
            experiments_internal: Sequence[Experiment | None] = [
                experiments for _ in self._measurements.grouped_parameters
            ]
        else:
            experiments_internal = experiments

        if len(experiments_internal) != len(self._measurements.grouped_parameters):
            raise ValueError(
                f"Inconsistent number of "
                f"parameter groups and experiments "
                f"got {len(self._measurements.grouped_parameters)} and {len(experiments_internal)}"
            )
        return experiments_internal

    def _create_groups(self) -> tuple[_SweapMeasGroup, ...]:

        if self._dataset_mapping is None:
            setpoint_groups = self._sweeper.sweep_groupes
            meaure_groups = self._measurements.grouped_parameters

            if len(setpoint_groups) == 1:
                setpoint_groups = (setpoint_groups[0],) * len(meaure_groups)

            if len(setpoint_groups) != len(meaure_groups):
                raise ValueError(
                    f"Inconsistent number of "
                    f"parameter groups and setpoint groups "
                    f"got {len(meaure_groups)} and {len(setpoint_groups)}"
                )
            groups = []
            sp_group: Sequence[ParameterBase]
            m_group: Sequence[ParamMeasT]
            for sp_group, m_group, experiment in zip(
                setpoint_groups, meaure_groups, self._experiments
            ):
                meas_ctx = self._create_measurement_cx_manager(
                    experiment, tuple(sp_group), tuple(m_group)
                )
                s_m_group = _SweapMeasGroup(
                    tuple(sp_group), tuple(m_group), experiment, meas_ctx
                )
                groups.append(s_m_group)
        else:
            potential_setpoint_groups = self._sweeper.sweep_groupes
            requested_meaure_groups = self._measurements.grouped_parameters
            groups = []
            for (sp_group, m_group), experiment in zip(
                self._dataset_mapping, self._experiments
            ):
                LOG.info(f"creating context manager for {sp_group} {m_group}")
                meas_ctx = self._create_measurement_cx_manager(
                    experiment, tuple(sp_group), tuple(m_group)
                )
                s_m_group = _SweapMeasGroup(
                    tuple(sp_group), tuple(m_group), experiment, meas_ctx
                )
                groups.append(s_m_group)

            # verify that each sweepgroup in the dict is part of the sweep groups
            # verify that each measurement group is a value in the dict
            # verify that each value in the dict is a measurement group
            # take each of the sweep groups and map it to measure groups
        return tuple(groups)

    @property
    def shapes(self) -> Shapes | None:
        return self._shapes

    def _create_measurement_cx_manager(
        self,
        experiment: Experiment | None,
        sweep_parameters: Sequence[ParameterBase],
        measure_parameters: Sequence[ParamMeasT],
    ) -> Measurement:
        meas_name = "TODO"
        meas = Measurement(name=meas_name, exp=experiment)
        _register_parameters(meas, sweep_parameters)
        _register_parameters(
            meas,
            measure_parameters,
            setpoints=sweep_parameters,
            shapes=self.shapes,
        )
        meas._extra_log_info = self._log_info or ""
        _set_write_period(meas, self._write_period)
        _register_actions(meas, self._enter_actions, self._exit_actions)
        return meas

    @property
    def groups(self) -> tuple[_SweapMeasGroup, ...]:
        return self._groups


def dond(
    *params: AbstractSweep | ParamMeasT | Sequence[ParamMeasT],
    write_period: float | None = None,
    measurement_name: str = "",
    exp: Experiment | Sequence[Experiment] | None = None,
    enter_actions: ActionsT = (),
    exit_actions: ActionsT = (),
    do_plot: bool | None = None,
    show_progress: bool | None = None,
    use_threads: bool | None = None,
    additional_setpoints: Sequence[ParameterBase] = tuple(),
    log_info: str | None = None,
    break_condition: BreakConditionT | None = None,
    dataset_mapping: Sequence[tuple[tuple[ParameterBase, ...], Sequence[ParamMeasT]]]
    | None = None,
) -> AxesTupleListWithDataSet | MultiAxesTupleListWithDataSet:
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
        measurements.grouped_parameters,
    )
    sweeper_measurer = _SweeperMeasure(
        sweeper,
        measurements,
        enter_actions,
        exit_actions,
        exp,
        write_period,
        log_info,
        dataset_mapping,
    )

    datasets = []
    plots_axes = []
    plots_colorbar = []
    if use_threads is None:
        use_threads = config.dataset.use_threads

    params_meas_caller = (
        ThreadPoolParamsCaller(*sweeper_measurer._measurements.measured_all)
        if use_threads
        else SequentialParamsCaller(*sweeper_measurer._measurements.measured_all)
    )

    try:
        with _catch_interrupts() as interrupted, ExitStack() as stack, params_meas_caller as call_params_meas:
            datasavers = [
                stack.enter_context(group.measurement_cxt.run())
                for group in sweeper_measurer._groups
            ]
            additional_setpoints_data = process_params_meas(additional_setpoints)
            for set_events in tqdm(sweeper, disable=not show_progress):
                results = {}
                for set_event in set_events:
                    if set_event.should_set:
                        set_event.parameter(set_event.new_value)
                        for act in set_event.actions:
                            act()
                        time.sleep(set_event.delay)

                    results[set_event.parameter] = set_event.new_value

                meas_value_pair = call_params_meas()
                for name, res in meas_value_pair:
                    results[name] = res

                for datasaver, group in zip(datasavers, sweeper_measurer._groups):
                    filtered_results_list = [
                        (param, value)
                        for param, value in results.items()
                        if param in group.parameters
                    ]
                    datasaver.add_result(
                        *filtered_results_list,
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
    *params: AbstractSweep | ParamMeasT | Sequence[ParamMeasT],
) -> tuple[list[AbstractSweep | MultiSweep], list[ParamMeasT | Sequence[ParamMeasT]]]:
    """
    Parse supplied arguments into sweep objects and measurement parameters
    and their callables.
    """
    sweep_instances: list[AbstractSweep | MultiSweep] = []
    params_meas: list[ParamMeasT | Sequence[ParamMeasT]] = []
    for par in params:
        if isinstance(par, AbstractSweep):
            sweep_instances.append(par)
        elif isinstance(par, MultiSweep):
            sweep_instances.append(par)
        else:
            params_meas.append(par)
    return sweep_instances, params_meas


def _conditional_parameter_set(
    parameter: ParameterBase,
    value: float | complex,
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
) -> tuple[list[ActionsT], list[float]]:
    """
    Select ActionT (Sequence[Callable]) and delays(Sequence[float]) from
    a Sequence of ActionsT and delays, respectively, if the corresponding
    setpoint has changed. Otherwise, select an empty Sequence for actions
    and zero for delays.
    """
    actions_list: list[ActionsT] = [()] * len(setpoints)
    setpoints_delay: list[float] = [0] * len(setpoints)
    for ind, (new_setpoint, old_setpoint) in enumerate(
        zip(setpoints, previous_setpoints)
    ):
        if new_setpoint != old_setpoint:
            actions_list[ind] = actions[ind]
            setpoints_delay[ind] = delays[ind]
    return actions_list, setpoints_delay


def _create_measurements(
    all_setpoint_params: Sequence[ParameterBase],
    enter_actions: ActionsT,
    exit_actions: ActionsT,
    experiments: Experiment | Sequence[Experiment] | None,
    grouped_parameters: Mapping[str, ParameterGroup],
    shapes: Shapes,
    write_period: float | None,
    log_info: str | None,
) -> tuple[Measurement, ...]:
    meas_list: list[Measurement] = []
    if log_info is not None:
        _extra_log_info = log_info
    else:
        _extra_log_info = "Using 'qcodes.dataset.dond'"

    if not isinstance(experiments, Sequence):
        experiments_internal: Sequence[Experiment | None] = [
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
    params_meas: Sequence[ParamMeasT | Sequence[ParamMeasT]],
) -> tuple[
    tuple[ParamMeasT, ...], dict[str, ParameterGroup], tuple[ParameterBase, ...]
]:
    measured_parameters: list[ParameterBase] = []
    measured_all: list[ParamMeasT] = []
    single_group: list[ParamMeasT] = []
    multi_group: list[Sequence[ParamMeasT]] = []
    grouped_parameters: dict[str, ParameterGroup] = {}
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
        }
        grouped_parameters["group_0"] = pg
    if multi_group:
        for index, par in enumerate(multi_group):
            pg = {
                "params": tuple(par),
                "meas_name": measurement_name,
            }
            grouped_parameters[f"group_{index}"] = pg
    return tuple(measured_all), grouped_parameters, tuple(measured_parameters)
