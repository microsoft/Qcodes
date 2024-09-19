from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import numpy as np
from opentelemetry import trace
from tqdm.auto import tqdm
from typing_extensions import TypedDict

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

from .sweeps import AbstractSweep, TogetherSweep

LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    from qcodes.dataset.descriptions.versioning.rundescribertypes import Shapes
    from qcodes.dataset.dond.do_nd_utils import (
        ActionsT,
        AxesTupleListWithDataSet,
        BreakConditionT,
        MultiAxesTupleListWithDataSet,
        ParamMeasT,
    )
    from qcodes.dataset.experiment_container import Experiment

SweepVarType = Any

TRACER = trace.get_tracer(__name__)


class ParameterGroup(TypedDict):
    params: tuple[ParamMeasT, ...]
    meas_name: str


@dataclass
class ParameterSetEvent:
    parameter: ParameterBase
    new_value: SweepVarType
    should_set: bool
    delay: float
    actions: ActionsT
    get_after_set: bool


class _Sweeper:
    def __init__(
        self,
        sweeps: Sequence[AbstractSweep | TogetherSweep],
        additional_setpoints: Sequence[ParameterBase],
    ):
        self._additional_setpoints = additional_setpoints
        self._sweeps = sweeps
        self._setpoints = self._make_setpoints_tuples()
        self._setpoints_dict = self._make_setpoints_dict()
        self._shape = self._make_shape(sweeps, additional_setpoints)
        self._iter_index = 0

    @property
    def setpoints_dict(self) -> dict[str, list[Any]]:
        return self._setpoints_dict

    def _make_setpoints_tuples(
        self,
    ) -> tuple[tuple[tuple[SweepVarType, ...] | SweepVarType, ...], ...]:
        sweeps = tuple(sweep.get_setpoints() for sweep in self._sweeps)
        return cast(
            tuple[tuple[tuple[SweepVarType, ...] | SweepVarType, ...], ...],
            tuple(itertools.product(*sweeps)),
        )

    def _make_single_point_setpoints_dict(self, index: int) -> dict[str, SweepVarType]:
        setpoint_dict = {}
        values = self._setpoints[index]
        for sweep, subvalues in zip(self._sweeps, values):
            if isinstance(sweep, TogetherSweep):
                for individual_sweep, value in zip(sweep.sweeps, subvalues):
                    setpoint_dict[individual_sweep.param.full_name] = value
            else:
                setpoint_dict[sweep.param.full_name] = subvalues
        return setpoint_dict

    def _make_setpoints_dict(self) -> dict[str, list[Any]]:
        setpoint_dict: dict[str, list[SweepVarType]] = {}

        for sweep in self._sweeps:
            if isinstance(sweep, TogetherSweep):
                for individual_sweep in sweep.sweeps:
                    setpoint_dict[individual_sweep.param.full_name] = []
            else:
                setpoint_dict[sweep.param.full_name] = []

        for setpoint_tuples in self._setpoints:
            for sweep, values in zip(self._sweeps, setpoint_tuples):
                if isinstance(sweep, TogetherSweep):
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
            if isinstance(sweep, TogetherSweep):
                sweeps.extend(sweep.sweeps)
            else:
                sweeps.append(sweep)
        return tuple(sweeps)

    @property
    def all_setpoint_params(self) -> tuple[ParameterBase, ...]:
        return tuple(sweep.param for sweep in self.all_sweeps) + tuple(
            self._additional_setpoints
        )

    @property
    def sweep_groupes(self) -> tuple[tuple[ParameterBase, ...], ...]:
        """
        These are all the combinations of setpoints we consider
        valid for setpoints in a dataset. A dataset must depend on
        at least one parameter from each dimension of the dond.
        For dimensions that uses TogetherSweep they may depend
        on one or more of the parameters of that sweep.
        """
        param_tuple_list: list[tuple[ParameterBase, ...]] = []
        for sweep in self._sweeps:
            if isinstance(sweep, TogetherSweep):
                param_tuple_list.append(
                    tuple(sub_sweep.param for sub_sweep in sweep.sweeps)
                )
            else:
                param_tuple_list.append((sweep.param,))

        param_tuple_list.extend(
            [(setpoint,) for setpoint in self._additional_setpoints]
        )

        # in param_tuple_list there is a tuple of possible setpoints for each
        # dim in the dond. For regular sweeps this is a 1 tuple but for
        # a TogetherSweep it is of length of the number of parameters.

        # now we expand to a list of setpoints in a TogetherSweep
        # to a list of all possible combinations of these.

        expanded_param_tuples = tuple(
            tuple(
                itertools.chain.from_iterable(
                    itertools.combinations(param_tuple, j + 1)
                    for j in range(len(param_tuple))
                )
            )
            for param_tuple in param_tuple_list
        )

        # next we generate all valid combinations of picking one parameter from each
        # dimension in the setpoints.
        setpoint_combinations = itertools.product(*expanded_param_tuples)

        setpoint_combinations_expanded = tuple(
            tuple(itertools.chain.from_iterable(setpoint_combination))
            for setpoint_combination in setpoint_combinations
        )

        return setpoint_combinations_expanded

    @staticmethod
    def _make_shape(
        sweeps: Sequence[AbstractSweep | TogetherSweep],
        addtional_setpoints: Sequence[ParameterBase],
    ) -> tuple[int, ...]:
        loop_shape = tuple(sweep.num_points for sweep in sweeps) + tuple(
            1 for _ in addtional_setpoints
        )
        return loop_shape

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def __getitem__(self, index: int) -> tuple[ParameterSetEvent, ...]:
        setpoints = self._make_single_point_setpoints_dict(index)

        if index == 0:
            previous_setpoints: dict[str, SweepVarType | None] = {}
            for key in setpoints.keys():
                previous_setpoints[key] = None
        else:
            previous_setpoints = self._make_single_point_setpoints_dict(index - 1)

        parameter_set_events = []

        for sweep in self.all_sweeps:
            new_value = setpoints[sweep.param.full_name]
            old_value = previous_setpoints[sweep.param.full_name]
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
                get_after_set=sweep.get_after_set,
            )
            parameter_set_events.append(event)
        return tuple(parameter_set_events)

    def __len__(self) -> int:
        return int(np.prod(self.shape))

    def __iter__(self) -> _Sweeper:
        return self

    def __next__(self) -> tuple[ParameterSetEvent, ...]:
        if self._iter_index < len(self):
            return_val = self[self._iter_index]
            self._iter_index += 1
            return return_val
        else:
            raise StopIteration


class _Measurements:
    def __init__(
        self,
        sweeper: _Sweeper,
        measurement_name: str | Sequence[str],
        params_meas: Sequence[ParamMeasT | Sequence[ParamMeasT]],
        enter_actions: ActionsT,
        exit_actions: ActionsT,
        experiments: Experiment | Sequence[Experiment] | None,
        write_period: float | None,
        log_info: str | None,
        dataset_dependencies: Mapping[str, Sequence[ParamMeasT]] | None = None,
    ):
        self._sweeper = sweeper
        self._enter_actions = enter_actions
        self._exit_actions = exit_actions
        self._write_period = write_period
        self._log_info = log_info
        if log_info is not None:
            self._extra_log_info = log_info
        else:
            self._extra_log_info = "Using 'qcodes.dataset.dond'"

        (
            self._measured_all,
            grouped_parameters,
            self._measured_parameters,
        ) = self._extract_parameters_by_type_and_group(params_meas)

        self._shapes = self._get_shapes()

        if dataset_dependencies and len(grouped_parameters) > 1:
            raise ValueError(
                "Measured parameters have been grouped both in input "
                "and given in dataset dependencies. This is not supported, "
                "group measurement parameters either in input or in "
                "dataset dependencies."
            )

        if dataset_dependencies is None:
            self._groups = self._create_groups_from_grouped_parameters(
                grouped_parameters, experiments, measurement_name
            )
        elif dataset_dependencies:
            _validate_dataset_dependencies_and_names(
                dataset_dependencies, measurement_name
            )
            dataset_dependencies_split = self._split_dateset_dependencies(
                dataset_dependencies
            )
            self._groups = self._create_groups_from_dataset_dependencies(
                dataset_dependencies_split,
                self._measured_parameters,
                experiments,
                measurement_name,
            )

    @property
    def measured_all(self) -> tuple[ParamMeasT, ...]:
        return self._measured_all

    @property
    def measured_parameters(self) -> tuple[ParameterBase, ...]:
        return self._measured_parameters

    @property
    def shapes(self) -> Shapes | None:
        return self._shapes

    @property
    def groups(self) -> tuple[_SweepMeasGroup, ...]:
        return self._groups

    @staticmethod
    def _create_measurement_names(
        measurement_name: str | Sequence[str], n_names_required: int
    ) -> tuple[str, ...]:
        if isinstance(measurement_name, str):
            return (measurement_name,) * n_names_required
        else:
            if len(measurement_name) != n_names_required:
                raise ValueError(
                    f"Got {len(measurement_name)} measurement names "
                    f"but should create {n_names_required} dataset(s)."
                )
            return tuple(measurement_name)

    @staticmethod
    def _extract_parameters_by_type_and_group(
        params_meas: Sequence[ParamMeasT | Sequence[ParamMeasT]],
    ) -> tuple[
        tuple[ParamMeasT, ...],
        tuple[tuple[ParamMeasT, ...], ...],
        tuple[ParameterBase, ...],
    ]:
        measured_parameters: list[ParameterBase] = []
        measured_all: list[ParamMeasT] = []
        single_group: list[ParamMeasT] = []
        multi_group: list[tuple[ParamMeasT, ...]] = []
        grouped_parameters: tuple[tuple[ParamMeasT, ...], ...]
        for param in params_meas:
            if not isinstance(param, Sequence):
                single_group.append(param)
                measured_all.append(param)
                if isinstance(param, ParameterBase):
                    measured_parameters.append(param)
            elif not isinstance(param, str):
                multi_group.append(tuple(param))
                for nested_param in param:
                    measured_all.append(nested_param)
                    if isinstance(nested_param, ParameterBase):
                        measured_parameters.append(nested_param)

        if single_group and multi_group:
            raise ValueError(
                f"Got both grouped and non grouped "
                f"parameters to measure in "
                f"{params_meas}. This is not supported."
            )

        if single_group:
            grouped_parameters = (tuple(single_group),)
        elif multi_group:
            grouped_parameters = tuple(multi_group)
        else:
            raise ValueError("No parameters to measure supplied")

        return tuple(measured_all), grouped_parameters, tuple(measured_parameters)

    def _get_shapes(self) -> Shapes | None:
        try:
            shapes: Shapes | None = detect_shape_of_measurement(
                self.measured_parameters, self._sweeper.shape
            )
            LOG.debug("Detected shapes to be %s", shapes)
        except TypeError:
            LOG.exception(
                f"Could not detect shape of {self.measured_parameters} "
                f"falling back to unknown shape."
            )
            shapes = None
        return shapes

    @staticmethod
    def _get_experiments(
        experiments: Experiment | Sequence[Experiment] | None,
        n_experiments_required: int,
    ) -> Sequence[Experiment | None]:
        if not isinstance(experiments, Sequence):
            experiments_internal: Sequence[Experiment | None] = [
                experiments
            ] * n_experiments_required
        else:
            experiments_internal = experiments

        if len(experiments_internal) != n_experiments_required:
            raise ValueError(
                f"Inconsistent number of "
                f"datasets and experiments, "
                f"got {n_experiments_required} and {len(experiments_internal)}."
            )
        return experiments_internal

    def _create_groups_from_grouped_parameters(
        self,
        grouped_parameters: tuple[tuple[ParamMeasT, ...], ...],
        experiments: Experiment | Sequence[Experiment] | None,
        meas_names: str | Sequence[str],
    ) -> tuple[_SweepMeasGroup, ...]:
        setpoints = self._sweeper.all_setpoint_params

        groups = []
        m_group: Sequence[ParamMeasT]

        experiments_internal = self._get_experiments(
            experiments, len(grouped_parameters)
        )
        measurement_names = self._create_measurement_names(
            meas_names, len(grouped_parameters)
        )

        for m_group, experiment, meas_name in zip(
            grouped_parameters,
            experiments_internal,
            measurement_names,
        ):
            meas_ctx = self._create_measurement_ctx_manager(
                experiment, meas_name, setpoints, tuple(m_group)
            )
            s_m_group = _SweepMeasGroup(setpoints, tuple(m_group), meas_ctx)
            groups.append(s_m_group)
        return tuple(groups)

    def _create_groups_from_dataset_dependencies(
        self,
        dataset_dependencies: dict[
            str, tuple[Sequence[ParameterBase], Sequence[ParamMeasT]]
        ],
        all_measured_parameters: tuple[ParameterBase, ...],
        experiments: Experiment | Sequence[Experiment] | None,
        meas_names: str | Sequence[str],
    ) -> tuple[_SweepMeasGroup, ...]:
        potential_setpoint_groups = self._sweeper.sweep_groupes

        experiments_internal = self._get_experiments(
            experiments, len(dataset_dependencies)
        )
        if meas_names == "":
            meas_names = tuple(dataset_dependencies.keys())

        measurement_names = self._create_measurement_names(
            meas_names, len(dataset_dependencies)
        )

        all_dataset_dependencies_meas_parameters = tuple(
            itertools.chain.from_iterable(
                output[1] for output in dataset_dependencies.values()
            )
        )

        for meas_param in all_measured_parameters:
            if meas_param not in all_dataset_dependencies_meas_parameters:
                raise ValueError(
                    f"Parameter {meas_param} is measured but not added "
                    f"to any dataset in dataset_dependencies."
                )

        groups = []
        for experiment, meas_name in zip(
            experiments_internal,
            measurement_names,
        ):
            (sp_group, m_group) = dataset_dependencies[meas_name]
            if tuple(sp_group) not in potential_setpoint_groups:
                raise ValueError(
                    f"dataset_dependencies contains {sp_group} "
                    f"which is not among the expected groups of setpoints "
                    f"{potential_setpoint_groups}"
                )

            LOG.info(
                f"creating context manager for setpoints"
                f" {sp_group} and measurement parameters {m_group}"
            )
            meas_ctx = self._create_measurement_ctx_manager(
                experiment, meas_name, tuple(sp_group), tuple(m_group)
            )
            s_m_group = _SweepMeasGroup(tuple(sp_group), tuple(m_group), meas_ctx)
            groups.append(s_m_group)
        return tuple(groups)

    def _create_measurement_ctx_manager(
        self,
        experiment: Experiment | None,
        measurement_name: str,
        sweep_parameters: Sequence[ParameterBase],
        measure_parameters: Sequence[ParamMeasT],
    ) -> Measurement:
        meas = Measurement(name=measurement_name, exp=experiment)
        _register_parameters(meas, sweep_parameters)
        _register_parameters(
            meas,
            measure_parameters,
            setpoints=sweep_parameters,
            shapes=self.shapes,
        )
        meas._extra_log_info = self._extra_log_info
        _set_write_period(meas, self._write_period)
        _register_actions(meas, self._enter_actions, self._exit_actions)
        return meas

    def _split_dateset_dependencies(
        self,
        dataset_dependencies: Mapping[str, Sequence[ParamMeasT]],
    ) -> dict[str, tuple[Sequence[ParameterBase], Sequence[ParamMeasT]]]:
        # split measured parameters from setpoint parameters using param_meas
        dataset_dependencies_split: dict[
            str, tuple[Sequence[ParameterBase], Sequence[ParamMeasT]]
        ] = {}
        for name, dataset_parameters in dataset_dependencies.items():
            meas_parameters = tuple(
                param for param in dataset_parameters if param in self.measured_all
            )
            setpoint_parameters = cast(
                Sequence[ParameterBase],
                tuple(
                    param
                    for param in dataset_parameters
                    if param not in self.measured_all
                ),
            )
            dataset_dependencies_split[name] = (
                setpoint_parameters,
                meas_parameters,
            )
        return dataset_dependencies_split


# idealy we would want this to be frozen but then postinit
# cannot calculate all the parameters
# https://stackoverflow.com/questions/53756788/
@dataclass(frozen=False)
class _SweepMeasGroup:
    sweep_parameters: tuple[ParameterBase, ...]
    measure_parameters: tuple[ParamMeasT, ...]
    measurement_cxt: Measurement

    def __post_init__(self) -> None:
        meas_parameters = tuple(
            a for a in self.measure_parameters if isinstance(a, ParameterBase)
        )
        self._parameters = self.sweep_parameters + meas_parameters

    @property
    def parameters(self) -> tuple[ParameterBase, ...]:
        return self._parameters


@overload
def dond(
    *params: AbstractSweep | TogetherSweep | ParamMeasT | Sequence[ParamMeasT],
    write_period: float | None = None,
    measurement_name: str | Sequence[str] = "",
    exp: Experiment | Sequence[Experiment] | None = None,
    enter_actions: ActionsT = (),
    exit_actions: ActionsT = (),
    do_plot: bool | None = None,
    show_progress: bool | None = None,
    use_threads: bool | None = None,
    additional_setpoints: Sequence[ParameterBase] = tuple(),
    log_info: str | None = None,
    break_condition: BreakConditionT | None = None,
    dataset_dependencies: Mapping[str, Sequence[ParamMeasT]] | None = None,
    in_memory_cache: bool | None = None,
    squeeze: Literal[False],
) -> MultiAxesTupleListWithDataSet: ...


@overload
def dond(
    *params: AbstractSweep | TogetherSweep | ParamMeasT | Sequence[ParamMeasT],
    write_period: float | None = None,
    measurement_name: str | Sequence[str] = "",
    exp: Experiment | Sequence[Experiment] | None = None,
    enter_actions: ActionsT = (),
    exit_actions: ActionsT = (),
    do_plot: bool | None = None,
    show_progress: bool | None = None,
    use_threads: bool | None = None,
    additional_setpoints: Sequence[ParameterBase] = tuple(),
    log_info: str | None = None,
    break_condition: BreakConditionT | None = None,
    dataset_dependencies: Mapping[str, Sequence[ParamMeasT]] | None = None,
    in_memory_cache: bool | None = None,
    squeeze: Literal[True],
) -> AxesTupleListWithDataSet | MultiAxesTupleListWithDataSet: ...


@overload
def dond(
    *params: AbstractSweep | TogetherSweep | ParamMeasT | Sequence[ParamMeasT],
    write_period: float | None = None,
    measurement_name: str | Sequence[str] = "",
    exp: Experiment | Sequence[Experiment] | None = None,
    enter_actions: ActionsT = (),
    exit_actions: ActionsT = (),
    do_plot: bool | None = None,
    show_progress: bool | None = None,
    use_threads: bool | None = None,
    additional_setpoints: Sequence[ParameterBase] = tuple(),
    log_info: str | None = None,
    break_condition: BreakConditionT | None = None,
    dataset_dependencies: Mapping[str, Sequence[ParamMeasT]] | None = None,
    in_memory_cache: bool | None = None,
    squeeze: bool = True,
) -> AxesTupleListWithDataSet | MultiAxesTupleListWithDataSet: ...


@TRACER.start_as_current_span("qcodes.dataset.dond")
def dond(
    *params: AbstractSweep | TogetherSweep | ParamMeasT | Sequence[ParamMeasT],
    write_period: float | None = None,
    measurement_name: str | Sequence[str] = "",
    exp: Experiment | Sequence[Experiment] | None = None,
    enter_actions: ActionsT = (),
    exit_actions: ActionsT = (),
    do_plot: bool | None = None,
    show_progress: bool | None = None,
    use_threads: bool | None = None,
    additional_setpoints: Sequence[ParameterBase] = tuple(),
    log_info: str | None = None,
    break_condition: BreakConditionT | None = None,
    dataset_dependencies: Mapping[str, Sequence[ParamMeasT]] | None = None,
    in_memory_cache: bool | None = None,
    squeeze: bool = True,
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

            If you want to sweep multiple parameters together.

            .. code-block::

                TogetherSweep(LinSweep(param_set_1, start_1, stop_1, num_points, delay_1),
                              LinSweep(param_set_2, start_2, stop_2, num_points, delay_2))
                param_meas_1, param_meas_2, ..., param_meas_m


        write_period: The time after which the data is actually written to the
            database.
        measurement_name: Name(s) of the measurement. This will be passed down to
            the dataset produced by the measurement. If not given, a default
            value of 'results' is used for the dataset. If more than one is
            given, each dataset will have an individual name.
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
        dataset_dependencies: Optionally describe that measured datasets only depend
            on a subset of the setpoint parameters. Given as a mapping from
            measurement names to Sequence of Parameters. Note that a dataset must
            depend on at least one parameter from each dimension but can depend
            on one or more parameters from a dimension sweeped with a TogetherSweep.
        in_memory_cache:
            Should a cache of the data be kept available in memory for faster
            plotting and exporting. Useful to disable if the data is very large
            in order to save on memory consumption.
            If ``None``, the value for this will be read from ``qcodesrc.json`` config file.
        squeeze: If True, will return a tuple of QCoDeS DataSet, Matplotlib axis,
            Matplotlib colorbar if only one group of measurements was performed
            and a tuple of tuples of these if more than one group of measurements
            was performed. If False, will always return a tuple where the first
            member is a tuple of QCoDeS DataSet(s) and the second member is a tuple
            of Matplotlib axis(es) and the third member is a tuple of Matplotlib
            colorbar(s).

    Returns:
        A tuple of QCoDeS DataSet, Matplotlib axis, Matplotlib colorbar. If
        more than one group of measurement parameters is supplied, the output
        will be a tuple of tuple(QCoDeS DataSet), tuple(Matplotlib axis),
        tuple(Matplotlib colorbar), in which each element of each sub-tuple
        belongs to one group, and the order of elements is the order of
        the supplied groups.
    """
    if do_plot is None:
        do_plot = cast(bool, config.dataset.dond_plot)
    if show_progress is None:
        show_progress = config.dataset.dond_show_progress

    sweep_instances, params_meas = _parse_dond_arguments(*params)

    sweeper = _Sweeper(sweep_instances, additional_setpoints)

    measurements = _Measurements(
        sweeper,
        measurement_name,
        params_meas,
        enter_actions,
        exit_actions,
        exp,
        write_period,
        log_info,
        dataset_dependencies,
    )

    LOG.info(
        "Starting a doNd with scan with\n setpoints: %s,\n measuring: %s",
        sweeper.all_setpoint_params,
        measurements.measured_all,
    )
    LOG.debug(
        "dond has been grouped into the following datasets:\n%s",
        measurements.groups,
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

    datasavers = []
    interrupted: Callable[  # noqa E731
        [], KeyboardInterrupt | BreakConditionInterrupt | None
    ] = lambda: None
    try:
        with (
            catch_interrupts() as interrupted,
            ExitStack() as stack,
            params_meas_caller as call_params_meas,
        ):
            datasavers = [
                stack.enter_context(
                    group.measurement_cxt.run(in_memory_cache=in_memory_cache)
                )
                for group in measurements.groups
            ]
            additional_setpoints_data = process_params_meas(additional_setpoints)
            for set_events in tqdm(sweeper, disable=not show_progress):
                LOG.debug("Processing set events: %s", set_events)
                results: dict[ParameterBase, Any] = {}
                for set_event in set_events:
                    if set_event.should_set:
                        set_event.parameter(set_event.new_value)
                        for act in set_event.actions:
                            act()
                        time.sleep(set_event.delay)

                    if set_event.get_after_set:
                        results[set_event.parameter] = set_event.parameter()
                    else:
                        results[set_event.parameter] = set_event.new_value

                meas_value_pair = call_params_meas()
                for meas_param, value in meas_value_pair:
                    results[meas_param] = value

                for datasaver, group in zip(datasavers, measurements.groups):
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

    if len(measurements.groups) == 1 and squeeze is True:
        return datasets[0], plots_axes[0], plots_colorbar[0]
    else:
        return tuple(datasets), tuple(plots_axes), tuple(plots_colorbar)


def _validate_dataset_dependencies_and_names(
    dataset_dependencies: Mapping[str, Sequence[ParamMeasT]] | None,
    measurement_name: str | Sequence[str],
) -> None:
    if dataset_dependencies is not None and measurement_name != "":
        if isinstance(measurement_name, str):
            raise ValueError(
                "Creating multiple datasets but only one measurement name given."
            )
        if set(dataset_dependencies.keys()) != set(measurement_name):
            raise ValueError(
                f"Inconsistent measurement names: measurement_name "
                f"contains {measurement_name} "
                f"but dataset_dependencies contains "
                f"{tuple(dataset_dependencies.keys())}."
            )


def _parse_dond_arguments(
    *params: AbstractSweep | TogetherSweep | ParamMeasT | Sequence[ParamMeasT],
) -> tuple[
    list[AbstractSweep | TogetherSweep], list[ParamMeasT | Sequence[ParamMeasT]]
]:
    """
    Parse supplied arguments into sweep objects and measurement parameters
    and their callables.
    """
    sweep_instances: list[AbstractSweep | TogetherSweep] = []
    params_meas: list[ParamMeasT | Sequence[ParamMeasT]] = []
    for par in params:
        if isinstance(par, AbstractSweep):
            sweep_instances.append(par)
        elif isinstance(par, TogetherSweep):
            sweep_instances.append(par)
        else:
            params_meas.append(par)
    return sweep_instances, params_meas
