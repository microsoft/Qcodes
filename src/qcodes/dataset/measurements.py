"""
The measurement module provides a context manager for registering parameters
to measure and storing results. The user is expected to mainly interact with it
using the :class:`.Measurement` class.
"""

from __future__ import annotations

import collections
import io
import logging
import traceback as tb_module
import warnings
from collections.abc import Callable, Mapping, MutableMapping, MutableSequence, Sequence
from contextlib import ExitStack
from copy import deepcopy
from inspect import signature
from itertools import chain
from numbers import Number
from time import perf_counter, perf_counter_ns
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, cast

import numpy as np
import numpy.typing as npt
from opentelemetry import trace

import qcodes as qc
import qcodes.validators as vals
from qcodes.dataset.data_set import DataSet, load_by_guid
from qcodes.dataset.data_set_in_memory import DataSetInMem
from qcodes.dataset.data_set_protocol import (
    DataSetProtocol,
    DataSetType,
    ResType,
    SetpointsType,
    ValuesType,
)
from qcodes.dataset.descriptions.dependencies import (
    FrozenInterDependencies_,
    IncompleteSubsetError,
    InterDependencies_,
    ParamSpecTree,
)
from qcodes.dataset.descriptions.param_spec import ParamSpec
from qcodes.dataset.export_config import get_data_export_automatic
from qcodes.parameters import (
    ArrayParameter,
    GroupedParameter,
    ManualParameter,
    MultiParameter,
    Parameter,
    ParameterBase,
    ParameterWithSetpoints,
    ParamSpecBase,
)
from qcodes.station import Station
from qcodes.utils import DelayedKeyboardInterrupt

if TYPE_CHECKING:
    from types import TracebackType
    from typing import Self

    from qcodes.dataset.descriptions.versioning.rundescribertypes import Shapes
    from qcodes.dataset.experiment_container import Experiment
    from qcodes.dataset.sqlite.connection import AtomicConnection
    from qcodes.dataset.sqlite.query_helpers import VALUE

log = logging.getLogger(__name__)
TRACER = trace.get_tracer(__name__)


ActionType = tuple[Callable[..., Any], Sequence[Any]]
SubscriberType = tuple[
    Callable[..., Any], MutableSequence[Any] | MutableMapping[Any, Any]
]

ParameterResultType: TypeAlias = tuple[ParameterBase, ValuesType]
DatasetResultDict: TypeAlias = dict[ParamSpecBase, npt.NDArray]


class ParameterTypeError(Exception):
    pass


class DataSaver:
    """
    The class used by the :class:`Runner` context manager to handle the
    datasaving to the database.
    """

    default_callback: dict[Any, Any] | None = None

    def __init__(
        self,
        dataset: DataSetProtocol,
        write_period: float,
        interdeps: InterDependencies_,
        registered_parameters: Sequence[ParameterBase],
        span: trace.Span | None = None,
    ) -> None:
        self._span = span
        self._dataset = dataset
        self._add_result_time_ns = 0
        if (
            DataSaver.default_callback is not None
            and "run_tables_subscription_callback" in DataSaver.default_callback
        ):
            callback = DataSaver.default_callback["run_tables_subscription_callback"]
            min_wait = DataSaver.default_callback["run_tables_subscription_min_wait"]
            min_count = DataSaver.default_callback["run_tables_subscription_min_count"]
            snapshot = dataset.metadata["snapshot"]
            if isinstance(self._dataset, DataSet):
                self._dataset.subscribe(
                    callback,
                    min_wait=min_wait,
                    min_count=min_count,
                    state={},
                    callback_kwargs={
                        "run_id": self._dataset.run_id,
                        "snapshot": snapshot,
                    },
                )
        if isinstance(self._dataset, DataSet):
            default_subscribers = qc.config.subscription.default_subscribers
            for subscriber in default_subscribers:
                self._dataset.subscribe_from_config(subscriber)

        self._interdeps = interdeps
        self.write_period = float(write_period)
        # self._results will be filled by add_result
        self._results: list[dict[str, VALUE]] = []
        self._last_save_time = perf_counter()
        self._known_dependencies: dict[str, list[str]] = {}
        self.parent_datasets: list[DataSetProtocol] = []
        self._registered_parameters = registered_parameters

        for link in self._dataset.parent_dataset_links:
            self.parent_datasets.append(load_by_guid(link.tail))

    def _validate_result_tuples_no_duplicates(self, *result_tuples: ResType) -> None:
        """Validate that the result tuples do not contain duplicates"""

        parameter_names = tuple(
            result_tuple[0].register_name
            if isinstance(result_tuple[0], ParameterBase)
            else result_tuple[0]
            for result_tuple in result_tuples
        )
        if len(set(parameter_names)) != len(parameter_names):
            non_unique = [
                item
                for item, count in collections.Counter(parameter_names).items()
                if count > 1
            ]
            raise ValueError(
                f"Not all parameter names are unique. "
                f"Got multiple values for {non_unique}"
            )

    def _coerce_result_tuple_to_parameter_result_type(
        self, result_tuple: ResType
    ) -> ParameterResultType:
        param_or_str = result_tuple[0]
        if isinstance(param_or_str, ParameterBase):
            return (param_or_str, result_tuple[1])
        else:  # param_or_str is a str
            candidate_params = [
                param
                for param in self._registered_parameters
                if param.register_name == result_tuple[0]
            ]
            if len(candidate_params) > 1:
                raise ValueError(
                    f"More than one parameter matched the name {param_or_str}"
                    f"{candidate_params}"
                )
            elif len(candidate_params) < 1:
                raise ValueError("No matching parameters")
            return (candidate_params[0], result_tuple[1])

    def add_result(self, *result_tuples: ResType) -> None:
        """
        Add a result to the measurement results. Represents a measurement
        point in the space of measurement parameters, e.g. in an experiment
        varying two voltages and measuring two currents, a measurement point
        is four dimensional (v1, v2, c1, c2). The corresponding call
        to this function would be

            >>> datasaver.add_result((v1, 0.1), (v2, 0.2), (c1, 5), (c2, -2.1))

        For better performance, this function does not immediately write to
        the database, but keeps the results in memory. Writing happens every
        ``write_period`` seconds and during the ``__exit__`` method
        of this class.

        Args:
            result_tuples: One or more result tuples with the first element
                being the parameter name and the second element is the
                corresponding value(s) at this measurement point. The function
                takes as many tuples as there are results.

        Raises:
            ValueError: If a parameter name is not registered in the parent
                Measurement object.
            ValueError: If the shapes of parameters do not match, i.e. if a
                parameter gets values of a different shape than its setpoints
                (the exception being that setpoints can always be scalar)
            ValueError: If multiple results are given for the same parameter.
            ParameterTypeError: If a parameter is given a value not matching
                its type.

        """
        start_time = perf_counter_ns()

        parameter_results: list[ParameterResultType] = [
            self._coerce_result_tuple_to_parameter_result_type(result_tuple)
            for result_tuple in result_tuples
        ]

        non_unique = [
            item.register_name
            for item, count in collections.Counter(
                [parameter_result[0] for parameter_result in parameter_results]
            ).items()
            if count > 1
        ]
        if len(non_unique) > 0:
            raise ValueError(
                f"Not all parameter names are unique. "
                f"Got multiple values for {non_unique}"
            )

        legacy_results_dict: DatasetResultDict = {}
        self_unpacked_parameter_results: list[ParameterResultType] = []

        for parameter_result in parameter_results:
            if isinstance(parameter_result[0], ArrayParameter):
                legacy_results_dict.update(
                    self._unpack_arrayparameter(parameter_result)
                )
            elif isinstance(parameter_result[0], MultiParameter):
                legacy_results_dict.update(
                    self._unpack_multiparameter(parameter_result)
                )
            else:
                self_unpacked_parameter_results.extend(
                    parameter_result[0].unpack_self(parameter_result[1])
                )

        all_results_dict: dict[ParamSpecBase, list[npt.NDArray]] = (
            collections.defaultdict(list)
        )
        for parameter_result in self_unpacked_parameter_results:
            try:
                result_paramspec = self._interdeps._id_to_paramspec[
                    parameter_result[0].register_name
                ]
            except KeyError:
                raise ValueError(
                    "Can not add result for parameter "
                    f"{parameter_result[0].register_name}, "
                    "no such parameter registered "
                    "with this measurement."
                )
            all_results_dict[result_paramspec].append(np.array(parameter_result[1]))

        # Add any unpacked results from legacy Parameter types
        for key, value in legacy_results_dict.items():
            all_results_dict[key].append(value)

        datasaver_results_dict: DatasetResultDict = _deduplicate_results(
            all_results_dict
        )

        self._validate_result_deps(datasaver_results_dict)
        self._validate_result_shapes(datasaver_results_dict)
        self._validate_result_types(datasaver_results_dict)

        self.dataset._enqueue_results(datasaver_results_dict)

        if perf_counter() - self._last_save_time > self.write_period:
            self.flush_data_to_database()
            self._last_save_time = perf_counter()
        self._add_result_time_ns += perf_counter_ns() - start_time

    def _unpack_arrayparameter(
        self, partial_result: ResType
    ) -> dict[ParamSpecBase, npt.NDArray]:
        """
        Unpack a partial result containing an :class:`Arrayparameter` into a
        standard results dict form and return that dict
        """
        array_param, values_array = partial_result
        array_param = cast("ArrayParameter", array_param)

        if array_param.setpoints is None:
            raise RuntimeError(
                f"{array_param.full_name} is an "
                f"{type(array_param)} "
                f"without setpoints. Cannot handle this."
            )
        try:
            main_parameter = self._interdeps._id_to_paramspec[str(array_param)]
        except KeyError:
            raise ValueError(
                "Can not add result for parameter "
                f"{array_param}, no such parameter registered "
                "with this measurement."
            )

        res_dict = {main_parameter: np.array(values_array)}

        sp_names = array_param.setpoint_full_names
        fallback_sp_name = f"{array_param.full_name}_setpoint"

        res_dict.update(
            self._unpack_setpoints_from_parameter(
                array_param, array_param.setpoints, sp_names, fallback_sp_name
            )
        )

        return res_dict

    def _unpack_multiparameter(
        self, partial_result: ResType
    ) -> dict[ParamSpecBase, npt.NDArray]:
        """
        Unpack the `subarrays` and `setpoints` from a :class:`MultiParameter`
        and into a standard results dict form and return that dict

        """

        parameter, data = partial_result
        parameter = cast("MultiParameter", parameter)

        result_dict = {}

        if parameter.setpoints is None:
            raise RuntimeError(
                f"{parameter.full_name} is an "
                f"{type(parameter)} "
                f"without setpoints. Cannot handle this."
            )
        for i in range(len(parameter.shapes)):
            # if this loop runs, then 'data' is a Sequence
            data = cast("Sequence[str | int | float | Any]", data)

            shape = parameter.shapes[i]

            try:
                paramspec = self._interdeps._id_to_paramspec[parameter.full_names[i]]
            except KeyError:
                raise ValueError(
                    "Can not add result for parameter "
                    f"{parameter.names[i]}, "
                    "no such parameter registered "
                    "with this measurement."
                )

            result_dict.update({paramspec: np.array(data[i])})
            if shape != ():
                # array parameter like part of the multiparameter
                # need to find setpoints too
                fallback_sp_name = f"{parameter.full_names[i]}_setpoint"

                sp_names: Sequence[str] | None
                if (
                    parameter.setpoint_full_names is not None
                    and parameter.setpoint_full_names[i] is not None
                ):
                    sp_names = parameter.setpoint_full_names[i]
                else:
                    sp_names = None

                result_dict.update(
                    self._unpack_setpoints_from_parameter(
                        parameter, parameter.setpoints[i], sp_names, fallback_sp_name
                    )
                )

        return result_dict

    def _unpack_setpoints_from_parameter(
        self,
        parameter: ParameterBase,
        setpoints: Sequence[Any],
        sp_names: Sequence[str] | None,
        fallback_sp_name: str,
    ) -> dict[ParamSpecBase, npt.NDArray]:
        """
        Unpack the `setpoints` and their values from a
        :class:`ArrayParameter` or :class:`MultiParameter`
        into a standard results dict form and return that dict
        """
        setpoint_axes = []
        setpoint_parameters: list[ParamSpecBase] = []

        for i, sps in enumerate(setpoints):
            if sp_names is not None:
                spname = sp_names[i]
            else:
                spname = f"{fallback_sp_name}_{i}"

            try:
                setpoint_parameter = self._interdeps[spname]
            except KeyError:
                raise RuntimeError(
                    "No setpoints registered for "
                    f"{type(parameter)} {parameter.full_name}!"
                )
            sps = np.array(sps)
            while sps.ndim > 1:
                # The outermost setpoint axis or an nD param is nD
                # but the innermost is 1D. In all cases we just need
                # the axis along one dim, the innermost one.
                sps = sps[0]

            setpoint_parameters.append(setpoint_parameter)
            setpoint_axes.append(sps)

        output_grids = np.meshgrid(*setpoint_axes, indexing="ij")
        result_dict = {}
        for grid, param in zip(output_grids, setpoint_parameters):
            result_dict.update({param: grid})

        return result_dict

    def _validate_result_deps(
        self, results_dict: Mapping[ParamSpecBase, ValuesType]
    ) -> None:
        """
        Validate that the dependencies of the ``results_dict`` are met,
        meaning that (some) values for all required setpoints and inferences
        are present
        """
        try:
            self._interdeps.validate_subset(list(results_dict.keys()))
        except IncompleteSubsetError as err:
            raise ValueError(
                "Can not add result, some required parameters are missing."
            ) from err

    def _validate_result_shapes(
        self, results_dict: Mapping[ParamSpecBase, ValuesType]
    ) -> None:
        """
        Validate that all sizes of the ``results_dict`` are consistent.
        This means that array-values of parameters and their setpoints are
        of the same size, whereas parameters with no setpoint relation to
        each other can have different sizes.
        """
        toplevel_params = set(self._interdeps.dependencies).intersection(
            set(results_dict)
        )
        for toplevel_param in toplevel_params:
            required_shape = np.shape(np.array(results_dict[toplevel_param]))
            for setpoint in self._interdeps.dependencies[toplevel_param]:
                # a setpoint is allowed to be a scalar; shape is then ()
                setpoint_shape = np.shape(np.array(results_dict[setpoint]))
                if setpoint_shape not in [(), required_shape]:
                    raise ValueError(
                        f"Incompatible shapes. Parameter "
                        f"{toplevel_param.name} has shape "
                        f"{required_shape}, but its setpoint "
                        f"{setpoint.name} has shape "
                        f"{setpoint_shape}."
                    )

    @staticmethod
    def _validate_result_types(
        results_dict: Mapping[ParamSpecBase, npt.NDArray],
    ) -> None:
        """
        Validate the type of the results
        """

        allowed_kinds = {
            "numeric": "iuf",
            "text": "SU",
            "array": "iufcSUmM",
            "complex": "c",
        }

        for ps, values in results_dict.items():
            if values.dtype.kind not in allowed_kinds[ps.type]:
                raise ValueError(
                    f"Parameter {ps.name} is of type "
                    f'"{ps.type}", but got a result of '
                    f"type {values.dtype} ({values})."
                )

    def flush_data_to_database(self, block: bool = False) -> None:
        """
        Write the in-memory results to the database.

        Args:
            block: If writing using a background thread block until the
                background thread has written all data to disc. The
                argument has no effect if not using a background thread.

        """
        self.dataset._flush_data_to_database(block=block)

    def export_data(self) -> None:
        """Export data at end of measurement as per export_type
        specification in "dataset" section of qcodes config
        """
        self.dataset.export(automatic_export=True)

    @property
    def run_id(self) -> int:
        return self._dataset.run_id

    @property
    def points_written(self) -> int:
        return self._dataset.number_of_results

    @property
    def dataset(self) -> DataSetProtocol:
        return self._dataset


class Runner:
    """
    Context manager for the measurement.

    Lives inside a :class:`Measurement` and should never be instantiated
    outside a Measurement.

    This context manager handles all the dirty business of writing data
    to the database. Additionally, it may perform experiment bootstrapping
    and clean-up after a measurement.
    """

    def __init__(
        self,
        enteractions: Sequence[ActionType],
        exitactions: Sequence[ActionType],
        experiment: Experiment | None = None,
        station: Station | None = None,
        write_period: float | None = None,
        interdeps: InterDependencies_ = InterDependencies_(),
        name: str = "",
        subscribers: Sequence[SubscriberType] | None = None,
        parent_datasets: Sequence[Mapping[Any, Any]] = (),
        extra_log_info: str = "",
        write_in_background: bool = False,
        shapes: Shapes | None = None,
        in_memory_cache: bool | None = None,
        dataset_class: DataSetType = DataSetType.DataSet,
        parent_span: trace.Span | None = None,
        registered_parameters: Sequence[ParameterBase] = (),
    ) -> None:
        if in_memory_cache is None:
            in_memory_cache = qc.config.dataset.in_memory_cache
            in_memory_cache = cast("bool", in_memory_cache)

        self._dataset_class = dataset_class
        self.write_period = self._calculate_write_period(
            write_in_background, write_period
        )

        self.enteractions = enteractions
        self.exitactions = exitactions
        self.subscribers: Sequence[SubscriberType]
        if subscribers is None:
            self.subscribers = []
        else:
            self.subscribers = subscribers
        self.experiment = experiment
        self.station = station
        self._interdependencies = interdeps
        self._shapes: Shapes | None = shapes
        self.name = name if name else "results"
        self._parent_datasets = parent_datasets
        self._extra_log_info = extra_log_info
        self._write_in_background = write_in_background
        self._in_memory_cache = in_memory_cache
        self._parent_span = parent_span
        self.ds: DataSetProtocol
        self._registered_parameters = registered_parameters

    @staticmethod
    def _calculate_write_period(
        write_in_background: bool, write_period: float | None
    ) -> float:
        write_period_changed_from_default = (
            write_period is not None
            and write_period != qc.config.defaults.dataset.write_period
        )
        if write_in_background and write_period_changed_from_default:
            warnings.warn(
                f"The specified write period of {write_period} s "
                "will be ignored, since write_in_background==True"
            )
        if write_in_background:
            return 0.0
        if write_period is None:
            write_period = cast("float", qc.config.dataset.write_period)
        return float(write_period)

    def __enter__(self) -> DataSaver:
        # multiple runners can be active at the same time.
        # If we just activate them in order the first one
        # would be the parent of the next one but that is wrong
        # since they are siblings that should coexist with the
        # same parent.
        if self._parent_span is not None:
            context = trace.set_span_in_context(self._parent_span)
        else:
            context = None
        # We want to enter the opentelemetry span here
        # and end it in the `__exit__` of this context manger
        # so here we capture it in a exitstack that we keep around.
        self._span = TRACER.start_span(
            "qcodes.dataset.Measurement.run", context=context
        )
        with ExitStack() as stack:
            stack.enter_context(trace.use_span(self._span, end_on_exit=True))

            self._exit_stack = stack.pop_all()

        # TODO: should user actions really precede the dataset?
        # first do whatever bootstrapping the user specified
        for func, args in self.enteractions:
            func(*args)

        # next set up the "datasaver"
        if self.experiment is not None:
            exp_id: int | None = self.experiment.exp_id
            path_to_db: str | None = self.experiment.path_to_db
            conn: AtomicConnection | None = self.experiment.conn
        else:
            exp_id = None
            path_to_db = None
            conn = None

        if self._dataset_class is DataSetType.DataSet:
            self.ds = DataSet(
                name=self.name,
                exp_id=exp_id,
                conn=conn,
                in_memory_cache=self._in_memory_cache,
            )
        elif self._dataset_class is DataSetType.DataSetInMem:
            if self._in_memory_cache is False:
                raise RuntimeError(
                    "Cannot disable the in memory cache for a "
                    "dataset that is only in memory."
                )
            self.ds = DataSetInMem._create_new_run(
                name=self.name,
                exp_id=exp_id,
                path_to_db=path_to_db,
            )
        else:
            raise RuntimeError("Does not support any other dataset classes")

        # .. and give the dataset a snapshot as metadata
        if self.station is None:
            station = Station.default
        else:
            station = self.station

        if station is not None:
            snapshot = {"station": station.snapshot()}
        else:
            snapshot = {}
        if self._registered_parameters is not None:
            parameter_snapshot = {
                param.short_name: param.snapshot()
                for param in self._registered_parameters
            }
            parameter_snapshot.update(
                {
                    param.register_name: param.snapshot()
                    for param in self._registered_parameters
                }
            )
            snapshot["parameters"] = parameter_snapshot

        self.ds.prepare(
            snapshot=snapshot,
            interdeps=self._interdependencies,
            write_in_background=self._write_in_background,
            shapes=self._shapes,
            parent_datasets=self._parent_datasets,
        )

        # register all subscribers
        if isinstance(self.ds, DataSet):
            for callble, state in self.subscribers:
                # We register with minimal waiting time.
                # That should make all subscribers be called when data is flushed
                # to the database
                log.debug(f"Subscribing callable {callble} with state {state}")
                self.ds.subscribe(callble, min_wait=0, min_count=1, state=state)
        self._span.set_attributes(
            {
                "qcodes_guid": self.ds.guid,
                "run_id": self.ds.run_id,
                "exp_name": self.ds.exp_name,
                "SN": self.ds.sample_name,
                "ds_name": self.ds.name,
                "write_in_background": self._write_in_background,
                "extra_log_info": self._extra_log_info,
                "dataset_class": self._dataset_class.name,
            }
        )
        print(
            f"Starting experimental run with id: {self.ds.captured_run_id}."
            f" {self._extra_log_info}"
        )
        log.info(
            f"Starting measurement with guid: {self.ds.guid}, "
            f'sample_name: "{self.ds.sample_name}", '
            f'exp_name: "{self.ds.exp_name}", '
            f'ds_name: "{self.ds.name}". '
            f"{self._extra_log_info}"
        )
        log.info(f"Using background writing: {self._write_in_background}")

        self.datasaver = DataSaver(
            dataset=self.ds,
            write_period=self.write_period,
            interdeps=self._interdependencies,
            registered_parameters=self._registered_parameters,
            span=self._span,
        )

        return self.datasaver

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        with DelayedKeyboardInterrupt(
            context={"reason": "qcodes measurement exit", "qcodes_guid": self.ds.guid}
        ):
            add_result_time = self.datasaver._add_result_time_ns
            self._span.set_attribute("qcodes_add_result_time_ms", add_result_time / 1e6)
            self.datasaver.flush_data_to_database(block=True)

            # perform the "teardown" events
            for func, args in self.exitactions:
                func(*args)

            if exception_type:
                # if an exception happened during the measurement,
                # log the exception
                stream = io.StringIO()
                tb_module.print_exception(
                    exception_type, exception_value, traceback, file=stream
                )
                exception_string = stream.getvalue()
                log.warning(
                    "An exception occurred in measurement with guid: %s;"
                    "\nTraceback:\n%s",
                    self.ds.guid,
                    exception_string,
                )
                self._span.set_status(trace.Status(trace.StatusCode.ERROR))
                if isinstance(exception_value, Exception):
                    self._span.record_exception(exception_value)
                self.ds.add_metadata("measurement_exception", exception_string)

            # for now we set the interdependencies back to the
            # not frozen state, so that further modifications are possible
            # this is not recommended but we want to minimize the changes for now

            if isinstance(self.ds.description.interdeps, FrozenInterDependencies_):
                intedeps = self.ds.description.interdeps.to_interdependencies()
            else:
                intedeps = self.ds.description.interdeps

            if isinstance(self.ds, DataSet):
                self.ds.set_interdependencies(
                    shapes=self.ds.description.shapes,
                    interdeps=intedeps,
                    override=True,
                )
            elif isinstance(self.ds, DataSetInMem):
                self.ds._set_interdependencies(
                    shapes=self.ds.description.shapes,
                    interdeps=intedeps,
                    override=True,
                )

            # and finally mark the dataset as closed, thus
            # finishing the measurement
            # Note that the completion of a dataset entails waiting for the
            # write thread to terminate (iff the write thread has been started)
            self.ds.mark_completed()
            if get_data_export_automatic():
                self.datasaver.export_data()
            log.info(
                f"Finished measurement with guid: {self.ds.guid}. "
                f"{self._extra_log_info}"
            )
            if isinstance(self.ds, DataSet):
                self.ds.unsubscribe_all()
            self._exit_stack.close()


T = TypeVar("T", bound="Measurement")


class Measurement:
    """
    Measurement procedure container. Note that multiple measurement
    instances cannot be nested.

    Args:
        exp: Specify the experiment to use. If not given
            the default one is used. The default experiment
            is the latest one created.
        station: The QCoDeS station to snapshot. If not given, the
            default one is used.
        name: Name of the measurement. This will be passed down to the dataset
            produced by the measurement. If not given, a default value of
            'results' is used for the dataset.

    """

    def __init__(
        self,
        exp: Experiment | None = None,
        station: Station | None = None,
        name: str = "",
    ) -> None:
        self.exitactions: list[ActionType] = []
        self.enteractions: list[ActionType] = []
        self.subscribers: list[SubscriberType] = []

        self.experiment = exp
        self.station = station
        self.name = name
        self.write_period = qc.config.dataset.write_period
        self._interdeps = InterDependencies_()
        self._shapes: Shapes | None = None
        self._parent_datasets: list[dict[str, str]] = []
        self._extra_log_info: str = ""
        self._registered_parameters: set[ParameterBase] = set()

    @property
    def parameters(self) -> dict[str, ParamSpecBase]:
        return deepcopy(self._interdeps._id_to_paramspec)

    @property
    def write_period(self) -> float:
        return self._write_period

    @write_period.setter
    def write_period(self, wp: float) -> None:
        if not isinstance(wp, Number):
            raise ValueError("The write period must be a number (of seconds).")
        wp_float = float(wp)
        if wp_float < 1e-3:
            raise ValueError("The write period must be at least 1 ms.")
        self._write_period = wp_float

    def _paramspecbase_from_strings(
        self,
        setpoints: Sequence[str] | None = None,
        basis: Sequence[str] | None = None,
    ) -> tuple[tuple[ParamSpecBase, ...], tuple[ParamSpecBase, ...]]:
        """
        Helper function to look up and get ParamSpecBases and to give a nice
        error message if the user tries to register a parameter with reference
        (setpoints, basis) to a parameter not registered with this measurement

        Called by _register_parameter and _self_register_parameter only.

        Args:
            setpoints: name(s) of the setpoint parameter(s)
            basis: name(s) of the parameter(s) that this parameter is
                inferred from

        """

        idps = self._interdeps

        # now handle setpoints
        depends_on = []
        if setpoints:
            for sp in setpoints:
                try:
                    sp_psb = idps._id_to_paramspec[sp]
                    depends_on.append(sp_psb)
                except KeyError:
                    raise ValueError(
                        f"Unknown setpoint: {sp}. Please register that parameter first."
                    )

        # now handle inferred parameters
        inf_from = []
        if basis:
            for inff in basis:
                try:
                    inff_psb = idps._id_to_paramspec[inff]
                    inf_from.append(inff_psb)
                except KeyError:
                    raise ValueError(
                        f"Unknown basis parameter: {inff}."
                        " Please register that parameter first."
                    )

        return tuple(depends_on), tuple(inf_from)

    def register_parent(
        self: Self, parent: DataSetProtocol, link_type: str, description: str = ""
    ) -> Self:
        """
        Register a parent for the outcome of this measurement

        Args:
            parent: The parent dataset
            link_type: A name for the type of parent-child link
            description: A free-text description of the relationship

        """
        # we save the information in a way that is very compatible with the
        # Link object we will eventually make out of this information. We
        # cannot create a Link object just yet, because the DataSet of this
        # Measurement has not been given a GUID yet
        parent_dict = {
            "tail": parent.guid,
            "edge_type": link_type,
            "description": description,
        }
        self._parent_datasets.append(parent_dict)

        return self

    def _paramspecs_and_parameters_from_setpoints(
        self, setpoints: SetpointsType | None
    ) -> tuple[list[ParamSpecBase], list[ParameterBase]]:
        paramspecs = []
        parameters = []
        if setpoints is not None:
            for setpoint in setpoints:
                if isinstance(setpoint, ParameterBase):
                    paramspecs.append(setpoint.param_spec)
                    parameters.append(setpoint)
                elif (
                    isinstance(setpoint, str)
                    and (
                        setpoint_paramspec := self._interdeps._id_to_paramspec.get(
                            setpoint, None
                        )
                    )
                    is not None
                ):
                    paramspecs.append(setpoint_paramspec)
                else:
                    raise ValueError(
                        f"Unknown interdependency: {setpoint}. Please register that parameter first."
                    )
        return paramspecs, parameters

    def _self_register_parameter(
        self: Self,
        parameter: ParameterBase,
        setpoints: SetpointsType | None = None,
        basis: SetpointsType | None = None,
    ) -> Self:
        # It is important to preserve the order of the setpoints (and basis) arguments
        # when building the dependency trees, as this order is implicitly used to assign
        # the axis-order for multidimensional data variables where shape alone is
        # insufficient (eg, if the shape is square)

        # Convert setpoints and basis arguments to ParamSpecBases
        dependency_paramspecs, dependency_parameters = (
            self._paramspecs_and_parameters_from_setpoints(setpoints)
        )
        inference_paramspecs, inference_parameters = (
            self._paramspecs_and_parameters_from_setpoints(basis)
        )

        # Append internal dependencies/inferences
        dependency_paramspecs.extend(
            [param.param_spec for param in parameter.depends_on]
        )
        inference_paramspecs.extend(
            [param.param_spec for param in parameter.is_controlled_by]
        )

        # Make ParamSpecTrees and extend interdeps
        dependencies_tree: ParamSpecTree | None = None
        if len(dependency_paramspecs) > 0:
            dependencies_tree = {parameter.param_spec: tuple(dependency_paramspecs)}

        inferences_tree: ParamSpecTree | None = None
        if len(inference_paramspecs) > 0:
            inferences_tree = {parameter.param_spec: tuple(inference_paramspecs)}

        standalones: tuple[ParamSpecBase, ...] = ()
        if dependencies_tree is None and inferences_tree is None:
            standalones = (parameter.param_spec,)

        self._interdeps = self._interdeps.extend(
            dependencies=dependencies_tree,
            inferences=inferences_tree,
            standalones=standalones,
        )
        self._registered_parameters.add(parameter)
        log.info(f"Registered {parameter.register_name} in the Measurement.")

        # Recursively register all other interdependent parameters related to this parameter
        interdependent_parameters = list(
            chain.from_iterable(
                [
                    dependency_parameters,
                    inference_parameters,
                    parameter.depends_on,
                    parameter.is_controlled_by,
                ]
            )
        )
        for interdependent_parameter in interdependent_parameters:
            if interdependent_parameter not in self._registered_parameters:
                self._self_register_parameter(interdependent_parameter)

        # We handle the `has_control_of` relationship differently so that the controlled parameter
        # does not need to implement the reverse-direction `is_controlled_by` to get the
        # inference relationship
        for controlled_parameter in parameter.has_control_of:
            self._self_register_parameter(controlled_parameter, basis=(parameter,))

        return self

    def register_parameter(
        self: Self,
        parameter: ParameterBase,
        setpoints: SetpointsType | None = None,
        basis: SetpointsType | None = None,
        paramtype: str | None = None,
    ) -> Self:
        """
        Add QCoDeS Parameter to the dataset produced by running this
        measurement.

        Args:
            parameter: The parameter to add
            setpoints: The Parameter representing the setpoints for this
                parameter. If this parameter is a setpoint,
                it should be left blank
            basis: The parameters that this parameter is inferred from. If
                this parameter is not inferred from any other parameters,
                this should be left blank.
            paramtype: Type of the parameter, i.e. the SQL storage class,
                If None the paramtype will be inferred from the parameter type
                and the validator of the supplied parameter.

        """
        if setpoints is not None:
            self._check_setpoints_type(setpoints, "setpoints")

        if basis is not None:
            self._check_setpoints_type(basis, "basis")

        match parameter:
            case ArrayParameter():
                paramtype = self._infer_paramtype(parameter, paramtype)
                self._register_arrayparameter(parameter, setpoints, basis, paramtype)
            case MultiParameter():
                paramtype = self._infer_paramtype(parameter, paramtype)
                self._register_multiparameter(
                    parameter,
                    setpoints,
                    basis,
                    paramtype,
                )
            case GroupedParameter():
                paramtype = self._infer_paramtype(parameter, paramtype)
                self._register_parameter(
                    parameter.register_name,
                    parameter.label,
                    parameter.unit,
                    setpoints,
                    basis,
                    paramtype,
                )
            case ParameterBase() | ParameterWithSetpoints():
                if paramtype is not None:
                    parameter.paramtype = paramtype
                self._self_register_parameter(parameter, setpoints, basis)
            case _:
                raise ValueError(
                    f"Can not register object of type {type(parameter)}. Can only "
                    "register a QCoDeS Parameter."
                )
        self._registered_parameters.add(parameter)

        return self

    @staticmethod
    def _check_setpoints_type(arg: SetpointsType, name: str) -> None:
        if (
            not isinstance(arg, Sequence)
            or isinstance(arg, str)
            or any(not isinstance(a, (str, ParameterBase)) for a in arg)
        ):
            raise TypeError(
                f"{name} should be a sequence of str or ParameterBase, not {type(arg)}"
            )

    @staticmethod
    def _infer_paramtype(parameter: ParameterBase, paramtype: str | None) -> str:
        """
        Infer the best parameter type to store the parameter supplied.

        Args:
            parameter: The parameter to to infer the type for
            paramtype: The initial supplied parameter type or None

        Returns:
            The inferred parameter type. If a not None parameter type is
            supplied this will be preferred over any inferred type.
            Returns None if a parameter type could not be inferred

        """
        return_paramtype: str
        if paramtype is not None:  # override with argument
            return_paramtype = paramtype
        elif isinstance(parameter.vals, vals.Arrays):
            return_paramtype = "array"
        elif isinstance(parameter, ArrayParameter):
            return_paramtype = "array"
        elif isinstance(parameter.vals, vals.Strings):
            return_paramtype = "text"
        elif isinstance(parameter.vals, vals.ComplexNumbers):
            return_paramtype = "complex"
        else:  # Default to this if nothing else matches
            return_paramtype = "numeric"

        if return_paramtype not in ParamSpec.allowed_types:
            raise RuntimeError(
                "Trying to register a parameter with type "
                f"{return_paramtype}. However, only "
                f"{ParamSpec.allowed_types} are supported."
            )
        # TODO should we try to figure out if parts of a multiparameter are
        # arrays or something else?
        return return_paramtype

    def _register_parameter(
        self: Self,
        name: str,
        label: str | None,
        unit: str | None,
        setpoints: SetpointsType | None,
        basis: SetpointsType | None,
        paramtype: str,
        metadata: dict[str, Any] | None = None,
    ) -> Self:
        """
        Update the interdependencies object with a new group
        """

        parameter: ParamSpecBase | None

        try:
            parameter = self._interdeps[name]
        except KeyError:
            parameter = None

        paramspec = ParamSpecBase(
            name=name, paramtype=paramtype, label=label, unit=unit
        )

        # We want to allow the registration of the exact same parameter twice,
        # the reason being that e.g. two ArrayParameters could share the same
        # setpoint parameter, which would then be registered along with each
        # dependent (array)parameter

        if parameter is not None and parameter != paramspec:
            raise ValueError("Parameter already registered in this Measurement.")

        if setpoints is not None:
            sp_strings = [str_or_register_name(sp) for sp in setpoints]
        else:
            sp_strings = []

        if basis is not None:
            bs_strings = [str_or_register_name(bs) for bs in basis]
        else:
            bs_strings = []

        # get the ParamSpecBases
        depends_on, inf_from = self._paramspecbase_from_strings(sp_strings, bs_strings)

        if depends_on:
            self._interdeps = self._interdeps.extend(
                dependencies={paramspec: depends_on}
            )
        if inf_from:
            self._interdeps = self._interdeps.extend(inferences={paramspec: inf_from})
        if not (depends_on or inf_from):
            self._interdeps = self._interdeps.extend(standalones=(paramspec,))

        log.info(f"Registered {name} in the Measurement.")

        return self

    def _register_arrayparameter(
        self,
        parameter: ArrayParameter,
        setpoints: SetpointsType | None,
        basis: SetpointsType | None,
        paramtype: str,
    ) -> None:
        """
        Register an ArrayParameter and the setpoints belonging to that
        ArrayParameter
        """
        my_setpoints = list(setpoints) if setpoints else []
        for i in range(len(parameter.shape)):
            if (
                parameter.setpoint_full_names is not None
                and parameter.setpoint_full_names[i] is not None
            ):
                spname = parameter.setpoint_full_names[i]
            else:
                spname = f"{parameter.register_name}_setpoint_{i}"
            if parameter.setpoint_labels:
                splabel = parameter.setpoint_labels[i]
            else:
                splabel = ""
            if parameter.setpoint_units:
                spunit = parameter.setpoint_units[i]
            else:
                spunit = ""

            self._register_parameter(
                name=spname,
                paramtype=paramtype,
                label=splabel,
                unit=spunit,
                setpoints=None,
                basis=None,
            )

            my_setpoints += [spname]

        self._register_parameter(
            parameter.register_name,
            parameter.label,
            parameter.unit,
            my_setpoints,
            basis,
            paramtype,
        )

    def _register_parameter_with_setpoints(
        self,
        parameter: ParameterWithSetpoints,
        setpoints: SetpointsType | None,
        basis: SetpointsType | None,
        paramtype: str,
    ) -> None:
        """
        Register an ParameterWithSetpoints and the setpoints belonging to the
        Parameter
        """
        my_setpoints = list(setpoints) if setpoints else []
        for sp in parameter.setpoints:
            if not isinstance(sp, Parameter):
                raise RuntimeError(
                    "The setpoints of a ParameterWithSetpoints must be a Parameter"
                )
            spname = sp.register_name
            splabel = sp.label
            spunit = sp.unit

            self._register_parameter(
                name=spname,
                paramtype=paramtype,
                label=splabel,
                unit=spunit,
                setpoints=None,
                basis=None,
            )

            my_setpoints.append(spname)

        self._register_parameter(
            parameter.register_name,
            parameter.label,
            parameter.unit,
            my_setpoints,
            basis,
            paramtype,
        )

    def _register_multiparameter(
        self,
        multiparameter: MultiParameter,
        setpoints: SetpointsType | None,
        basis: SetpointsType | None,
        paramtype: str,
    ) -> None:
        """
        Find the individual multiparameter components and their setpoints
        and register those as individual parameters
        """
        setpoints_lists = []
        for i in range(len(multiparameter.shapes)):
            shape = multiparameter.shapes[i]
            name = multiparameter.full_names[i]
            if shape == ():
                my_setpoints = setpoints
            else:
                my_setpoints = list(setpoints) if setpoints else []
                for j in range(len(shape)):
                    if (
                        multiparameter.setpoint_full_names is not None
                        and multiparameter.setpoint_full_names[i] is not None
                    ):
                        spname = multiparameter.setpoint_full_names[i][j]
                    else:
                        spname = f"{name}_setpoint_{j}"
                    if (
                        multiparameter.setpoint_labels is not None
                        and multiparameter.setpoint_labels[i] is not None
                    ):
                        splabel = multiparameter.setpoint_labels[i][j]
                    else:
                        splabel = ""
                    if (
                        multiparameter.setpoint_units is not None
                        and multiparameter.setpoint_units[i] is not None
                    ):
                        spunit = multiparameter.setpoint_units[i][j]
                    else:
                        spunit = ""

                    self._register_parameter(
                        name=spname,
                        paramtype=paramtype,
                        label=splabel,
                        unit=spunit,
                        setpoints=None,
                        basis=None,
                    )

                    my_setpoints += [spname]

            setpoints_lists.append(my_setpoints)

        for i, expanded_setpoints in enumerate(setpoints_lists):
            self._register_parameter(
                multiparameter.full_names[i],
                multiparameter.labels[i],
                multiparameter.units[i],
                expanded_setpoints,
                basis,
                paramtype,
            )

    def register_custom_parameter(
        self: Self,
        name: str,
        label: str | None = None,
        unit: str | None = None,
        basis: SetpointsType | None = None,
        setpoints: SetpointsType | None = None,
        paramtype: str = "numeric",
    ) -> Self:
        """
        Register a custom parameter with this measurement

        Args:
            name: The name that this parameter will have in the dataset. Must
                be unique (will overwrite an existing parameter with the same
                name!)
            label: The label
            unit: The unit
            basis: A list of either QCoDeS Parameters or the names
                of parameters already registered in the measurement that
                this parameter is inferred from
            setpoints: A list of either QCoDeS Parameters or the names of
                of parameters already registered in the measurement that
                are the setpoints of this parameter
            paramtype: Type of the parameter, i.e. the SQL storage class

        """
        custom_parameter = ManualParameter(name=name, label=label, unit=unit)
        self._registered_parameters.add(custom_parameter)
        return self._register_parameter(name, label, unit, setpoints, basis, paramtype)

    def unregister_parameter(self, parameter: SetpointsType) -> None:
        """
        Remove a custom/QCoDeS parameter from the dataset produced by
        running this measurement
        """
        if isinstance(parameter, ParameterBase):
            param_name = str_or_register_name(parameter)
        elif isinstance(parameter, str):
            param_name = parameter
        else:
            raise ValueError(
                "Wrong input type. Must be a QCoDeS parameter or"
                " the name (a string) of a parameter."
            )

        try:
            paramspec: ParamSpecBase = self._interdeps[param_name]
        except KeyError:
            return

        self._interdeps = self._interdeps.remove(paramspec)

        # Must follow interdeps removal, because interdeps removal may error
        if isinstance(parameter, ParameterBase):
            try:
                self._registered_parameters.remove(parameter)
            except ValueError:
                return
        elif isinstance(parameter, str):
            with_parameters_removed = [
                param
                for param in self._registered_parameters
                if parameter not in (param.name, param.register_name)
            ]
            self._registered_parameters = set(with_parameters_removed)

        log.info(f"Removed {param_name} from Measurement.")

    def add_before_run(
        self: Self, func: Callable[..., Any], args: Sequence[Any]
    ) -> Self:
        """
        Add an action to be performed before the measurement.

        Args:
            func: Function to be performed
            args: The arguments to said function

        """
        # some tentative cheap checking
        nargs = len(signature(func).parameters)
        if len(args) != nargs:
            raise ValueError(
                "Mismatch between function call signature and the provided arguments."
            )

        self.enteractions.append((func, args))

        return self

    def add_after_run(
        self: Self, func: Callable[..., Any], args: Sequence[Any]
    ) -> Self:
        """
        Add an action to be performed after the measurement.

        Args:
            func: Function to be performed
            args: The arguments to said function

        """
        # some tentative cheap checking
        nargs = len(signature(func).parameters)
        if len(args) != nargs:
            raise ValueError(
                "Mismatch between function call signature and the provided arguments."
            )

        self.exitactions.append((func, args))

        return self

    def add_subscriber(
        self: Self,
        func: Callable[..., Any],
        state: MutableSequence[Any] | MutableMapping[Any, Any],
    ) -> Self:
        """
        Add a subscriber to the dataset of the measurement.

        Args:
            func: A function taking three positional arguments: a list of
                tuples of parameter values, an integer, a mutable variable
                (list or dict) to hold state/writes updates to.
            state: The variable to hold the state.

        """
        self.subscribers.append((func, state))

        return self

    def set_shapes(self, shapes: Shapes | None) -> None:
        """
        Set the shapes of the data to be recorded in this
        measurement.

        Args:
            shapes: Dictionary from names of dependent parameters to a tuple
                of integers describing the shape of the measurement.

        """
        self._shapes = shapes

    def run(
        self,
        write_in_background: bool | None = None,
        in_memory_cache: bool | None = True,
        dataset_class: DataSetType = DataSetType.DataSet,
        parent_span: trace.Span | None = None,
    ) -> Runner:
        """
        Returns the context manager for the experimental run

        Args:
            write_in_background: if True, results that will be added
                within the context manager with ``DataSaver.add_result``
                will be stored in background, without blocking the
                main thread that is executing the context manager.
                By default the setting for write in background will be
                read from the ``qcodesrc.json`` config file.
            in_memory_cache: Should measured data be keep in memory
                and available as part of the `dataset.cache` object.
            dataset_class: Enum representing the Class used to store data
                with.
            parent_span: An optional opentelemetry span that this should be registered a
                a child of if using opentelemetry.

        """
        if write_in_background is None:
            write_in_background = cast("bool", qc.config.dataset.write_in_background)
        return Runner(
            self.enteractions,
            self.exitactions,
            self.experiment,
            station=self.station,
            write_period=self._write_period,
            interdeps=FrozenInterDependencies_(self._interdeps),
            name=self.name,
            subscribers=self.subscribers,
            parent_datasets=self._parent_datasets,
            extra_log_info=self._extra_log_info,
            write_in_background=write_in_background,
            shapes=self._shapes,
            in_memory_cache=in_memory_cache,
            dataset_class=dataset_class,
            parent_span=parent_span,
            registered_parameters=tuple(self._registered_parameters),
        )


def str_or_register_name(sp: str | ParameterBase) -> str:
    """Returns either the str passed or the register_name of the Parameter"""
    if isinstance(sp, str):
        return sp
    else:
        return sp.register_name


# TODO: These deduplication methods need testing against arrays with all ValuesType types
def _deduplicate_results(
    results_dict: dict[ParamSpecBase, list[npt.NDArray]],
) -> DatasetResultDict:
    deduplicated_results: dict[ParamSpecBase, npt.NDArray] = {}
    for param_spec, list_of_ndarrays_of_values in results_dict.items():
        if len(list_of_ndarrays_of_values) == 1 or _values_are_equal(
            list_of_ndarrays_of_values[0], *list_of_ndarrays_of_values[1:]
        ):
            deduplicated_results[param_spec] = list_of_ndarrays_of_values[0]
        else:
            raise ValueError(f"Multiple distinct values found for {param_spec.name}")
    return deduplicated_results


def _values_are_equal(ref_array: npt.NDArray, *values_arrays: npt.NDArray) -> bool:
    if np.issubdtype(ref_array.dtype, np.number):
        return _numeric_values_are_equal(ref_array, *values_arrays)
    return _non_numeric_values_are_equal(ref_array, *values_arrays)


def _non_numeric_values_are_equal(
    ref_array: npt.NDArray, *values_arrays: npt.NDArray
) -> bool:
    # For non-numeric values, we can use direct equality
    for value_array in values_arrays:
        if (ref_array.shape != value_array.shape) or not np.array_equal(
            value_array, ref_array
        ):
            return False
    return True


def _numeric_values_are_equal(
    ref_array: npt.NDArray, *values_arrays: npt.NDArray
) -> bool:
    # The equal_nan arg in np.allclose considers complex values with np.nan in
    # either real or imaginary part to be equal. That is, np.nan + 1.0j is equal to 1.0 + np.nan*1.0j.
    # Since we want a more granular equality, we split arrays with complex values
    # into real and imaginary parts to evaluate equality
    if np.issubdtype(ref_array.dtype, np.complexfloating):
        return _numeric_values_are_equal(
            np.real(ref_array), *[np.real(value_array) for value_array in values_arrays]
        ) and _numeric_values_are_equal(
            np.imag(ref_array), *[np.imag(value_array) for value_array in values_arrays]
        )

    for value_array in values_arrays:
        if (ref_array.shape != value_array.shape) or not np.allclose(
            value_array,
            ref_array,
            atol=0,
            rtol=1e-8,  # TODO: allow flexible rtol
            equal_nan=True,
        ):
            return False
    return True
