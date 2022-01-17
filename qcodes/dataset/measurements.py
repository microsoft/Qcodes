"""
The measurement module provides a context manager for registering parameters
to measure and storing results. The user is expected to mainly interact with it
using the :class:`.Measurement` class.
"""

import collections
import io
import logging
import traceback as tb_module
import warnings
from copy import deepcopy
from inspect import signature
from numbers import Number
from time import perf_counter
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np

import qcodes as qc
import qcodes.utils.validators as vals
from qcodes.dataset.data_set import VALUE, DataSet, load_by_guid
from qcodes.dataset.data_set_in_memory import DataSetInMem
from qcodes.dataset.data_set_protocol import (
    DataSetProtocol,
    DataSetType,
    res_type,
    setpoints_type,
    values_type,
)
from qcodes.dataset.descriptions.dependencies import (
    DependencyError,
    InferenceError,
    InterDependencies_,
)
from qcodes.dataset.descriptions.param_spec import ParamSpec, ParamSpecBase
from qcodes.dataset.descriptions.rundescriber import RunDescriber
from qcodes.dataset.descriptions.versioning.rundescribertypes import Shapes
from qcodes.dataset.experiment_container import Experiment
from qcodes.dataset.export_config import get_data_export_automatic
from qcodes.instrument.delegate.grouped_parameter import GroupedParameter
from qcodes.instrument.parameter import (
    ArrayParameter,
    MultiParameter,
    Parameter,
    ParameterWithSetpoints,
    _BaseParameter,
    expand_setpoints_helper,
)
from qcodes.station import Station
from qcodes.utils.delaykeyboardinterrupt import DelayedKeyboardInterrupt

if TYPE_CHECKING:
    from qcodes.dataset.sqlite.connection import ConnectionPlus

log = logging.getLogger(__name__)


ActionType = Tuple[Callable[..., Any], Sequence[Any]]
SubscriberType = Tuple[Callable[..., Any],
                       Union[MutableSequence[Any],
                             MutableMapping[Any, Any]]]


class ParameterTypeError(Exception):
    pass


class DataSaver:
    """
    The class used by the :class:`Runner` context manager to handle the
    datasaving to the database.
    """

    default_callback: Optional[Dict[Any, Any]] = None

    def __init__(
        self,
        dataset: DataSetProtocol,
        write_period: float,
        interdeps: InterDependencies_,
    ) -> None:
        self._dataset = dataset
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
        self._results: List[Dict[str, VALUE]] = []
        self._last_save_time = perf_counter()
        self._known_dependencies: Dict[str, List[str]] = {}
        self.parent_datasets: List[DataSetProtocol] = []

        for link in self._dataset.parent_dataset_links:
            self.parent_datasets.append(load_by_guid(link.tail))

    def add_result(self, *res_tuple: res_type) -> None:
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
            res_tuple: A tuple with the first element being the parameter name
                and the second element is the corresponding value(s) at this
                measurement point. The function takes as many tuples as there
                are results.

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

        # we iterate through the input twice. First we find any array and
        # multiparameters that need to be unbundled and collect the names
        # of all parameters. This also allows users to call
        # add_result with the arguments in any particular order, i.e. NOT
        # enforcing that setpoints come before dependent variables.
        results_dict: Dict[ParamSpecBase, np.ndarray] = {}

        parameter_names = tuple(partial_result[0].full_name
                                if isinstance(partial_result[0], _BaseParameter) else partial_result[0]
                                for partial_result in res_tuple)
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

        for partial_result in res_tuple:
            parameter = partial_result[0]
            data = partial_result[1]

            if (isinstance(parameter, _BaseParameter) and
                    isinstance(parameter.vals, vals.Arrays)):
                if not isinstance(data, np.ndarray):
                    raise TypeError(
                        f"Expected data for Parameter with Array validator "
                        f"to be a numpy array but got: {type(data)}")

                if (parameter.vals.shape is not None
                        and data.shape != parameter.vals.shape):
                    raise TypeError(
                        f"Expected data with shape {parameter.vals.shape}, "
                        f"but got {data.shape} for parameter: {parameter.full_name}"
                    )

            if isinstance(parameter, ArrayParameter):
                results_dict.update(
                    self._unpack_arrayparameter(partial_result))
            elif isinstance(parameter, MultiParameter):
                results_dict.update(
                    self._unpack_multiparameter(partial_result))
            elif isinstance(parameter, ParameterWithSetpoints):
                results_dict.update(
                    self._conditionally_expand_parameter_with_setpoints(
                        data, parameter, parameter_names, partial_result
                    )
                )
            else:
                results_dict.update(
                    self._unpack_partial_result(partial_result)
                )

        self._validate_result_deps(results_dict)
        self._validate_result_shapes(results_dict)
        self._validate_result_types(results_dict)

        self.dataset._enqueue_results(results_dict)

        if perf_counter() - self._last_save_time > self.write_period:
            self.flush_data_to_database()
            self._last_save_time = perf_counter()

    def _conditionally_expand_parameter_with_setpoints(
            self, data: values_type, parameter: ParameterWithSetpoints,
            parameter_names: Sequence[str], partial_result: res_type
    ) -> Dict[ParamSpecBase, np.ndarray]:
        local_results = {}
        setpoint_names = tuple(setpoint.full_name for setpoint in parameter.setpoints)
        expanded = tuple(setpoint_name in parameter_names for setpoint_name in setpoint_names)
        if all(expanded):
            local_results.update(
                self._unpack_partial_result(partial_result))
        elif any(expanded):
            raise ValueError(f"Some of the setpoints of {parameter.full_name} "
                             "were explicitly given but others were not. "
                             "Either supply all of them or none of them.")
        else:
            expanded_partial_result = expand_setpoints_helper(parameter, data)
            for res in expanded_partial_result:
                local_results.update(
                    self._unpack_partial_result(res)
                )
        return local_results

    def _unpack_partial_result(
            self,
            partial_result: res_type) -> Dict[ParamSpecBase, np.ndarray]:
        """
        Unpack a partial result (not containing :class:`ArrayParameters` or
        class:`MultiParameters`) into a standard results dict form and return
        that dict
        """
        param, values = partial_result
        try:
            parameter = self._interdeps._id_to_paramspec[str(param)]
        except KeyError:
            raise ValueError('Can not add result for parameter '
                             f'{param}, no such parameter registered '
                             'with this measurement.')
        return {parameter: np.array(values)}

    def _unpack_arrayparameter(
        self, partial_result: res_type) -> Dict[ParamSpecBase, np.ndarray]:
        """
        Unpack a partial result containing an :class:`Arrayparameter` into a
        standard results dict form and return that dict
        """
        array_param, values_array = partial_result
        array_param = cast(ArrayParameter, array_param)

        if array_param.setpoints is None:
            raise RuntimeError(f"{array_param.full_name} is an "
                               f"{type(array_param)} "
                               f"without setpoints. Cannot handle this.")
        try:
            main_parameter = self._interdeps._id_to_paramspec[str(array_param)]
        except KeyError:
            raise ValueError('Can not add result for parameter '
                             f'{array_param}, no such parameter registered '
                             'with this measurement.')

        res_dict = {main_parameter: np.array(values_array)}

        sp_names = array_param.setpoint_full_names
        fallback_sp_name = f"{array_param.full_name}_setpoint"

        res_dict.update(
            self._unpack_setpoints_from_parameter(
                array_param, array_param.setpoints,
                sp_names, fallback_sp_name))

        return res_dict

    def _unpack_multiparameter(
            self, partial_result: res_type) -> Dict[ParamSpecBase, np.ndarray]:
        """
        Unpack the `subarrays` and `setpoints` from a :class:`MultiParameter`
        and into a standard results dict form and return that dict

        """

        parameter, data = partial_result
        parameter = cast(MultiParameter, parameter)

        result_dict = {}

        if parameter.setpoints is None:
            raise RuntimeError(f"{parameter.full_name} is an "
                               f"{type(parameter)} "
                               f"without setpoints. Cannot handle this.")
        for i in range(len(parameter.shapes)):
            # if this loop runs, then 'data' is a Sequence
            data = cast(Sequence[Union[str, int, float, Any]], data)

            shape = parameter.shapes[i]

            try:
                paramspec = self._interdeps._id_to_paramspec[parameter.full_names[i]]
            except KeyError:
                raise ValueError('Can not add result for parameter '
                                 f'{parameter.names[i]}, '
                                 'no such parameter registered '
                                 'with this measurement.')

            result_dict.update({paramspec: np.array(data[i])})
            if shape != ():
                # array parameter like part of the multiparameter
                # need to find setpoints too
                fallback_sp_name = f'{parameter.full_names[i]}_setpoint'

                sp_names: Optional[Sequence[str]]
                if (parameter.setpoint_full_names is not None
                        and parameter.setpoint_full_names[i] is not None):
                    sp_names = parameter.setpoint_full_names[i]
                else:
                    sp_names = None

                result_dict.update(
                    self._unpack_setpoints_from_parameter(
                        parameter,
                        parameter.setpoints[i],
                        sp_names,
                        fallback_sp_name))

        return result_dict

    def _unpack_setpoints_from_parameter(
        self, parameter: _BaseParameter, setpoints: Sequence[Any],
        sp_names: Optional[Sequence[str]], fallback_sp_name: str
            ) -> Dict[ParamSpecBase, np.ndarray]:
        """
        Unpack the `setpoints` and their values from a
        :class:`ArrayParameter` or :class:`MultiParameter`
        into a standard results dict form and return that dict
        """
        setpoint_axes = []
        setpoint_parameters: List[ParamSpecBase] = []

        for i, sps in enumerate(setpoints):
            if sp_names is not None:
                spname = sp_names[i]
            else:
                spname = f'{fallback_sp_name}_{i}'

            try:
                setpoint_parameter = self._interdeps[spname]
            except KeyError:
                raise RuntimeError('No setpoints registered for '
                                   f'{type(parameter)} {parameter.full_name}!')
            sps = np.array(sps)
            while sps.ndim > 1:
                # The outermost setpoint axis or an nD param is nD
                # but the innermost is 1D. In all cases we just need
                # the axis along one dim, the innermost one.
                sps = sps[0]

            setpoint_parameters.append(setpoint_parameter)
            setpoint_axes.append(sps)

        output_grids = np.meshgrid(*setpoint_axes, indexing='ij')
        result_dict = {}
        for grid, param in zip(output_grids, setpoint_parameters):
            result_dict.update({param: grid})

        return result_dict

    def _validate_result_deps(
            self, results_dict: Mapping[ParamSpecBase, values_type]) -> None:
        """
        Validate that the dependencies of the ``results_dict`` are met,
        meaning that (some) values for all required setpoints and inferences
        are present
        """
        try:
            self._interdeps.validate_subset(list(results_dict.keys()))
        except (DependencyError, InferenceError) as err:
            raise ValueError('Can not add result, some required parameters '
                             'are missing.') from err

    def _validate_result_shapes(
            self, results_dict: Mapping[ParamSpecBase, values_type]) -> None:
        """
        Validate that all sizes of the ``results_dict`` are consistent.
        This means that array-values of parameters and their setpoints are
        of the same size, whereas parameters with no setpoint relation to
        each other can have different sizes.
        """
        toplevel_params = (set(self._interdeps.dependencies)
                           .intersection(set(results_dict)))
        for toplevel_param in toplevel_params:
            required_shape = np.shape(np.array(results_dict[toplevel_param]))
            for setpoint in self._interdeps.dependencies[toplevel_param]:
                # a setpoint is allowed to be a scalar; shape is then ()
                setpoint_shape = np.shape(np.array(results_dict[setpoint]))
                if setpoint_shape not in [(), required_shape]:
                    raise ValueError(f'Incompatible shapes. Parameter '
                                     f"{toplevel_param.name} has shape "
                                     f"{required_shape}, but its setpoint "
                                     f"{setpoint.name} has shape "
                                     f"{setpoint_shape}.")

    @staticmethod
    def _validate_result_types(
            results_dict: Mapping[ParamSpecBase, np.ndarray]) -> None:
        """
        Validate the type of the results
        """

        allowed_kinds = {'numeric': 'iuf', 'text': 'SU', 'array': 'iufcSUmM',
                         'complex': 'c'}

        for ps, vals in results_dict.items():
                if vals.dtype.kind not in allowed_kinds[ps.type]:
                    raise ValueError(f'Parameter {ps.name} is of type '
                                     f'"{ps.type}", but got a result of '
                                     f'type {vals.dtype} ({vals}).')

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
        self.dataset.export()

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
        experiment: Optional[Experiment] = None,
        station: Optional[Station] = None,
        write_period: Optional[float] = None,
        interdeps: InterDependencies_ = InterDependencies_(),
        name: str = "",
        subscribers: Optional[Sequence[SubscriberType]] = None,
        parent_datasets: Sequence[Mapping[Any, Any]] = (),
        extra_log_info: str = "",
        write_in_background: bool = False,
        shapes: Optional[Shapes] = None,
        in_memory_cache: bool = True,
        dataset_class: DataSetType = DataSetType.DataSet,
    ) -> None:

        self._dataset_class = dataset_class
        self.write_period = self._calculate_write_period(write_in_background,
                                                         write_period)

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
        self._shapes: Shapes = shapes
        self.name = name if name else 'results'
        self._parent_datasets = parent_datasets
        self._extra_log_info = extra_log_info
        self._write_in_background = write_in_background
        self._in_memory_cache = in_memory_cache
        self.ds: DataSetProtocol

    @staticmethod
    def _calculate_write_period(
            write_in_background: bool,
            write_period: Optional[float]
    ) -> float:
        write_period_changed_from_default = (
                write_period is not None and
                write_period != qc.config.defaults.dataset.write_period
        )
        if write_in_background and write_period_changed_from_default:
            warnings.warn(f"The specified write period of {write_period} s "
                          "will be ignored, since write_in_background==True")
        if write_in_background:
            return 0.0
        if write_period is None:
            write_period = qc.config.dataset.write_period
        return float(write_period)

    def __enter__(self) -> DataSaver:
        # TODO: should user actions really precede the dataset?
        # first do whatever bootstrapping the user specified

        for func, args in self.enteractions:
            func(*args)

        dataset_class: Type[DataSetProtocol]

        # next set up the "datasaver"
        if self.experiment is not None:
            exp_id: Optional[int] = self.experiment.exp_id
            path_to_db: Optional[str] = self.experiment.path_to_db
            conn: Optional["ConnectionPlus"] = self.experiment.conn
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
            snapshot = station.snapshot()
        else:
            snapshot = {}

        self.ds.prepare(
            snapshot=snapshot,
            interdeps=self._interdependencies,
            write_in_background=self._write_in_background,
            shapes=self._shapes,
            parent_datasets=self._parent_datasets,
        )

        # register all subscribers
        if isinstance(self.ds, DataSet):
            for (callble, state) in self.subscribers:
                # We register with minimal waiting time.
                # That should make all subscribers be called when data is flushed
                # to the database
                log.debug(f"Subscribing callable {callble} with state {state}")
                self.ds.subscribe(callble, min_wait=0, min_count=1, state=state)

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
                            interdeps=self._interdependencies)

        return self.datasaver

    def __exit__(self,
                 exception_type: Optional[Type[BaseException]],
                 exception_value: Optional[BaseException],
                 traceback: Optional[TracebackType]
                 ) -> None:
        with DelayedKeyboardInterrupt():
            self.datasaver.flush_data_to_database(block=True)

            # perform the "teardown" events
            for func, args in self.exitactions:
                func(*args)

            if exception_type:
                # if an exception happened during the measurement,
                # log the exception
                stream = io.StringIO()
                tb_module.print_exception(exception_type,
                                          exception_value,
                                          traceback,
                                          file=stream)
                exception_string = stream.getvalue()
                log.warning('An exception occured in measurement with guid: '
                            f'{self.ds.guid};\nTraceback:\n{exception_string}')
                self.ds.add_metadata("measurement_exception", exception_string)

            # and finally mark the dataset as closed, thus
            # finishing the measurement
            # Note that the completion of a dataset entails waiting for the
            # write thread to terminate (iff the write thread has been started)
            self.ds.mark_completed()
            if get_data_export_automatic():
                self.datasaver.export_data()
            log.info(f'Finished measurement with guid: {self.ds.guid}. '
                     f'{self._extra_log_info}')
            if isinstance(self.ds, DataSet):
                self.ds.unsubscribe_all()


T = TypeVar('T', bound='Measurement')


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
        exp: Optional[Experiment] = None,
        station: Optional[Station] = None,
        name: str = "",
    ) -> None:
        self.exitactions: List[ActionType] = []
        self.enteractions: List[ActionType] = []
        self.subscribers: List[SubscriberType] = []

        self.experiment = exp
        self.station = station
        self.name = name
        self.write_period: float = qc.config.dataset.write_period
        self._interdeps = InterDependencies_()
        self._shapes: Shapes = None
        self._parent_datasets: List[Dict[str, str]] = []
        self._extra_log_info: str = ''

    @property
    def parameters(self) -> Dict[str, ParamSpecBase]:
        return deepcopy(self._interdeps._id_to_paramspec)

    @property
    def write_period(self) -> float:
        return self._write_period

    @write_period.setter
    def write_period(self, wp: float) -> None:
        if not isinstance(wp, Number):
            raise ValueError('The write period must be a number (of seconds).')
        wp_float = float(wp)
        if wp_float < 1e-3:
            raise ValueError('The write period must be at least 1 ms.')
        self._write_period = wp_float

    def _paramspecbase_from_strings(
            self, name: str, setpoints: Optional[Sequence[str]] = None,
            basis: Optional[Sequence[str]] = None
            ) -> Tuple[Tuple[ParamSpecBase, ...], Tuple[ParamSpecBase, ...]]:
        """
        Helper function to look up and get ParamSpecBases and to give a nice
        error message if the user tries to register a parameter with reference
        (setpoints, basis) to a parameter not registered with this measurement

        Called by _register_parameter only.

        Args:
            name: Name of the parameter to register
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
                    raise ValueError(f'Unknown setpoint: {sp}.'
                                     ' Please register that parameter first.')

        # now handle inferred parameters
        inf_from = []
        if basis:
            for inff in basis:
                try:
                    inff_psb = idps._id_to_paramspec[inff]
                    inf_from.append(inff_psb)
                except KeyError:
                    raise ValueError(f'Unknown basis parameter: {inff}.'
                                     ' Please register that parameter first.')

        return tuple(depends_on), tuple(inf_from)

    def register_parent(
        self: T, parent: DataSetProtocol, link_type: str, description: str = ""
    ) -> T:
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
        parent_dict = {'tail': parent.guid, 'edge_type': link_type,
                       'description': description}
        self._parent_datasets.append(parent_dict)

        return self

    def register_parameter(
            self: T, parameter: _BaseParameter,
            setpoints: Optional[setpoints_type] = None,
            basis: Optional[setpoints_type] = None,
            paramtype: Optional[str] = None) -> T:
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
        if not isinstance(parameter, _BaseParameter):
            raise ValueError('Can not register object of type {}. Can only '
                             'register a QCoDeS Parameter.'
                             ''.format(type(parameter)))

        paramtype = self._infer_paramtype(parameter, paramtype)
        # default to numeric
        if paramtype is None:
            paramtype = 'numeric'

        # now the parameter type must be valid
        if paramtype not in ParamSpec.allowed_types:
            raise RuntimeError("Trying to register a parameter with type "
                               f"{paramtype}. However, only "
                               f"{ParamSpec.allowed_types} are supported.")

        if isinstance(parameter, ArrayParameter):
            self._register_arrayparameter(parameter,
                                          setpoints,
                                          basis,
                                          paramtype)
        elif isinstance(parameter, ParameterWithSetpoints):
            self._register_parameter_with_setpoints(parameter,
                                                    setpoints,
                                                    basis,
                                                    paramtype)
        elif isinstance(parameter, MultiParameter):
            self._register_multiparameter(parameter,
                                          setpoints,
                                          basis,
                                          paramtype,
                                          )
        elif isinstance(parameter, Parameter):
            self._register_parameter(parameter.full_name,
                                     parameter.label,
                                     parameter.unit,
                                     setpoints,
                                     basis, paramtype)
        elif isinstance(parameter, GroupedParameter):
            self._register_parameter(parameter.full_name,
                                     parameter.label,
                                     parameter.unit,
                                     setpoints,
                                     basis, paramtype)
        else:
            raise RuntimeError("Does not know how to register a parameter"
                               f"of type {type(parameter)}")

        return self

    @staticmethod
    def _infer_paramtype(parameter: _BaseParameter,
                         paramtype: Optional[str]) -> Optional[str]:
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
        if paramtype is not None:
            return paramtype

        if isinstance(parameter.vals, vals.Arrays):
            paramtype = 'array'
        elif isinstance(parameter, ArrayParameter):
            paramtype = 'array'
        elif isinstance(parameter.vals, vals.Strings):
            paramtype = 'text'
        elif isinstance(parameter.vals, vals.ComplexNumbers):
            paramtype = 'complex'
        # TODO should we try to figure out if parts of a multiparameter are
        # arrays or something else?
        return paramtype

    def _register_parameter(self: T, name: str,
                            label: Optional[str],
                            unit: Optional[str],
                            setpoints: Optional[setpoints_type],
                            basis: Optional[setpoints_type],
                            paramtype: str) -> T:
        """
        Update the interdependencies object with a new group
        """

        parameter: Optional[ParamSpecBase]

        try:
            parameter = self._interdeps[name]
        except KeyError:
            parameter = None

        paramspec = ParamSpecBase(name=name,
                                  paramtype=paramtype,
                                  label=label,
                                  unit=unit)

        # We want to allow the registration of the exact same parameter twice,
        # the reason being that e.g. two ArrayParameters could share the same
        # setpoint parameter, which would then be registered along with each
        # dependent (array)parameter

        if parameter is not None and parameter != paramspec:
            raise ValueError("Parameter already registered "
                             "in this Measurement.")

        if setpoints is not None:
            sp_strings = [str(sp) for sp in setpoints]
        else:
            sp_strings = []

        if basis is not None:
            bs_strings = [str(bs) for bs in basis]
        else:
            bs_strings = []

        # get the ParamSpecBases
        depends_on, inf_from = self._paramspecbase_from_strings(name,
                                                                sp_strings,
                                                                bs_strings)

        if depends_on:
            self._interdeps = self._interdeps.extend(
                                  dependencies={paramspec: depends_on})
        if inf_from:
            self._interdeps = self._interdeps.extend(
                                  inferences={paramspec: inf_from})
        if not(depends_on or inf_from):
            self._interdeps = self._interdeps.extend(standalones=(paramspec,))

        log.info(f'Registered {name} in the Measurement.')

        return self

    def _register_arrayparameter(self,
                                 parameter: ArrayParameter,
                                 setpoints: Optional[setpoints_type],
                                 basis: Optional[setpoints_type],
                                 paramtype: str, ) -> None:
        """
        Register an ArrayParameter and the setpoints belonging to that
        ArrayParameter
        """
        my_setpoints = list(setpoints) if setpoints else []
        for i in range(len(parameter.shape)):
            if parameter.setpoint_full_names is not None and \
                    parameter.setpoint_full_names[i] is not None:
                spname = parameter.setpoint_full_names[i]
            else:
                spname = f'{parameter.full_name}_setpoint_{i}'
            if parameter.setpoint_labels:
                splabel = parameter.setpoint_labels[i]
            else:
                splabel = ''
            if parameter.setpoint_units:
                spunit = parameter.setpoint_units[i]
            else:
                spunit = ''

            self._register_parameter(name=spname,
                                     paramtype=paramtype,
                                     label=splabel,
                                     unit=spunit,
                                     setpoints=None,
                                     basis=None)

            my_setpoints += [spname]

        self._register_parameter(parameter.full_name,
                                 parameter.label,
                                 parameter.unit,
                                 my_setpoints,
                                 basis,
                                 paramtype)

    def _register_parameter_with_setpoints(self,
                                           parameter: ParameterWithSetpoints,
                                           setpoints: Optional[setpoints_type],
                                           basis: Optional[setpoints_type],
                                           paramtype: str) -> None:
        """
        Register an ParameterWithSetpoints and the setpoints belonging to the
        Parameter
        """
        my_setpoints = list(setpoints) if setpoints else []
        for sp in parameter.setpoints:
            if not isinstance(sp, Parameter):
                raise RuntimeError("The setpoints of a "
                                   "ParameterWithSetpoints "
                                   "must be a Parameter")
            spname = sp.full_name
            splabel = sp.label
            spunit = sp.unit

            self._register_parameter(name=spname,
                                     paramtype=paramtype,
                                     label=splabel,
                                     unit=spunit,
                                     setpoints=None,
                                     basis=None)

            my_setpoints.append(spname)

        self._register_parameter(parameter.full_name,
                                 parameter.label,
                                 parameter.unit,
                                 my_setpoints,
                                 basis,
                                 paramtype)

    def _register_multiparameter(self,
                                 multiparameter: MultiParameter,
                                 setpoints: Optional[setpoints_type],
                                 basis: Optional[setpoints_type],
                                 paramtype: str) -> None:
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
                    if multiparameter.setpoint_full_names is not None and \
                            multiparameter.setpoint_full_names[i] is not None:
                        spname = multiparameter.setpoint_full_names[i][j]
                    else:
                        spname = f'{name}_setpoint_{j}'
                    if multiparameter.setpoint_labels is not None and \
                            multiparameter.setpoint_labels[i] is not None:
                        splabel = multiparameter.setpoint_labels[i][j]
                    else:
                        splabel = ''
                    if multiparameter.setpoint_units is not None and \
                            multiparameter.setpoint_units[i] is not None:
                        spunit = multiparameter.setpoint_units[i][j]
                    else:
                        spunit = ''

                    self._register_parameter(name=spname,
                                             paramtype=paramtype,
                                             label=splabel,
                                             unit=spunit,
                                             setpoints=None,
                                             basis=None)

                    my_setpoints += [spname]

            setpoints_lists.append(my_setpoints)

        for i, setpoints in enumerate(setpoints_lists):
            self._register_parameter(multiparameter.full_names[i],
                                     multiparameter.labels[i],
                                     multiparameter.units[i],
                                     setpoints,
                                     basis,
                                     paramtype)

    def register_custom_parameter(
            self: T, name: str,
            label: Optional[str] = None, unit: Optional[str] = None,
            basis: Optional[setpoints_type] = None,
            setpoints: Optional[setpoints_type] = None,
            paramtype: str = 'numeric') -> T:
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
        return self._register_parameter(name,
                                        label,
                                        unit,
                                        setpoints,
                                        basis,
                                        paramtype)

    def unregister_parameter(self,
                             parameter: setpoints_type) -> None:
        """
        Remove a custom/QCoDeS parameter from the dataset produced by
        running this measurement
        """
        if isinstance(parameter, _BaseParameter):
            param = str(parameter)
        elif isinstance(parameter, str):
            param = parameter
        else:
            raise ValueError('Wrong input type. Must be a QCoDeS parameter or'
                             ' the name (a string) of a parameter.')

        try:
            paramspec: ParamSpecBase = self._interdeps[param]
        except KeyError:
            return

        self._interdeps = self._interdeps.remove(paramspec)

        log.info(f'Removed {param} from Measurement.')

    def add_before_run(self: T, func: Callable[..., Any], args: Sequence[Any]) -> T:
        """
        Add an action to be performed before the measurement.

        Args:
            func: Function to be performed
            args: The arguments to said function
        """
        # some tentative cheap checking
        nargs = len(signature(func).parameters)
        if len(args) != nargs:
            raise ValueError('Mismatch between function call signature and '
                             'the provided arguments.')

        self.enteractions.append((func, args))

        return self

    def add_after_run(self: T,
                      func: Callable[..., Any], args: Sequence[Any]) -> T:
        """
        Add an action to be performed after the measurement.

        Args:
            func: Function to be performed
            args: The arguments to said function
        """
        # some tentative cheap checking
        nargs = len(signature(func).parameters)
        if len(args) != nargs:
            raise ValueError('Mismatch between function call signature and '
                             'the provided arguments.')

        self.exitactions.append((func, args))

        return self

    def add_subscriber(
            self: T,
            func: Callable[..., Any],
            state: Union[MutableSequence[Any], MutableMapping[Any, Any]]
    ) -> T:
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

    def set_shapes(self, shapes: Shapes) -> None:
        """
        Set the shapes of the data to be recorded in this
        measurement.

        Args:
            shapes: Dictionary from names of dependent parameters to a tuple
                of integers describing the shape of the measurement.
        """
        RunDescriber._verify_interdeps_shape(interdeps=self._interdeps,
                                             shapes=shapes)
        self._shapes = shapes

    def run(
        self,
        write_in_background: Optional[bool] = None,
        in_memory_cache: bool = True,
        dataset_class: DataSetType = DataSetType.DataSet,
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
        """
        if write_in_background is None:
            write_in_background = qc.config.dataset.write_in_background
        return Runner(
            self.enteractions,
            self.exitactions,
            self.experiment,
            station=self.station,
            write_period=self._write_period,
            interdeps=self._interdeps,
            name=self.name,
            subscribers=self.subscribers,
            parent_datasets=self._parent_datasets,
            extra_log_info=self._extra_log_info,
            write_in_background=write_in_background,
            shapes=self._shapes,
            in_memory_cache=in_memory_cache,
            dataset_class=dataset_class,
        )
