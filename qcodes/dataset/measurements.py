import json
import logging
from time import monotonic
from collections import OrderedDict
from typing import (Callable, Union, Dict, Tuple, List, Sequence, cast,
                    MutableMapping, MutableSequence, Optional, Any)
from inspect import signature
from numbers import Number

import numpy as np

import qcodes as qc
from qcodes import Station
from qcodes.instrument.parameter import ArrayParameter, _BaseParameter, \
    Parameter, MultiParameter
from qcodes.dataset.experiment_container import Experiment
from qcodes.dataset.param_spec import ParamSpec
from qcodes.dataset.data_set import DataSet

log = logging.getLogger(__name__)

array_like_types = (tuple, list, np.ndarray)
res_type = Tuple[Union[_BaseParameter, str],
                 Union[str, int, float, np.dtype, np.ndarray]]
setpoints_type = Sequence[Union[str, _BaseParameter]]
numeric_types = Union[float, int]


class ParameterTypeError(Exception):
    pass


def is_number(thing: Any) -> bool:
    """
    Test if an object can be converted to a number
    """
    try:
        float(thing)
        return True
    except (ValueError, TypeError):
        return False


class DataSaver:
    """
    The class used byt the Runner context manager to handle the
    datasaving to the database
    """

    default_callback: Optional[dict] = None

    def __init__(self, dataset: DataSet, write_period: numeric_types,
                 parameters: Dict[str, ParamSpec]) -> None:
        self._dataset = dataset
        if DataSaver.default_callback is not None and 'run_tables_subscription_callback' in DataSaver.default_callback:
            callback = DataSaver.default_callback[
                'run_tables_subscription_callback']
            min_wait = DataSaver.default_callback[
                'run_tables_subscription_min_wait']
            min_count = DataSaver.default_callback[
                'run_tables_subscription_min_count']
            snapshot = dataset.get_metadata('snapshot')
            self._dataset.subscribe(callback, min_wait=min_wait,
                                    min_count=min_count,
                                    state={},
                                    callback_kwargs={'run_id':
                                                         self._dataset.run_id,
                                                     'snapshot': snapshot})

        self.write_period = float(write_period)
        self.parameters = parameters
        self._known_parameters = list(parameters.keys())
        self._results: List[dict] = []  # will be filled by addResult
        self._last_save_time = monotonic()
        self._known_dependencies: Dict[str, List[str]] = {}
        for param, parspec in parameters.items():
            if parspec.depends_on != '':
                self._known_dependencies.update(
                    {str(param): parspec.depends_on.split(', ')})

    def add_result(self,
                   *res_tuple: res_type) -> None:
        """
        Add a result to the measurement results. Represents a measurement
        point in the space of measurement parameters, e.g. in an experiment
        varying two voltages and measuring two currents, a measurement point
        is the four dimensional (v1, v2, c1, c2). The corresponding call
        to this function would be (e.g.)
        >> datasaver.add_result((v1, 0.1), (v2, 0.2), (c1, 5), (c2, -2.1))

        For better performance, this function does not immediately write to
        the database, but keeps the results in memory. Writing happens every
        `write_period` seconds and during the __exit__ method if this class.

        Regarding arrays: since arrays as binary blobs are (almost) worthless
        in a relational database, this function "unravels" arrays passed to it.
        That, in turn, forces us to impose rules on what can be saved in one
        go. Any number of scalars and any number of arrays OF THE SAME LENGTH
        can be passed to add_result. The scalars are duplicated to match the
        arrays.

        However, if the parameter is registered as array type the numpy arrays
        are not unraveled but stored directly for improved performance.

        Args:
            res_tuple: a tuple with the first element being the parameter name
                and the second element is the corresponding value(s) at this
                measurement point. The function takes as many tuples as there
                are results.

        Raises:
            ValueError: if a parameter name not registered in the parent
                Measurement object is encountered.
            ParameterTypeError: if a parameter is given a value not matching
                its type.
        """
        res: List[res_type] = []

        # we iterate through the input twice. First we find any array and
        # multiparameters that needs to be unbundled and collect the names
        # of all parameters. This also allows users to call
        # add_result with the arguments in any particular order, i.e. NOT
        # enforcing that setpoints come before dependent variables.
        input_size = 1
        found_parameters: List[str] = []
        inserting_as_arrays = False
        inserting_unrolled_array = False

        for partial_result in res_tuple:
            parameter = partial_result[0]
            if isinstance(parameter, MultiParameter):
                # unpack parameters and potential setpoints from MultiParameter
                # unlike regular Parameters and ArrayParameters we don't want
                # to add the parameter it self only its components.
                data = partial_result[1]
                self._unbundle_multiparameter(parameter,
                                              data,
                                              res,
                                              found_parameters)
            else:
                res.append(partial_result)
                paramstr = str(parameter)
                found_parameters.append(paramstr)
                # unpack setpoints from array parameters and add them
                # to the res list
                if isinstance(parameter, ArrayParameter):
                    self._unbundle_arrayparameter(parameter,
                                                  res,
                                                  found_parameters)

        for partial_result in res:
            parameter = partial_result[0]
            paramstr = str(parameter)
            value = partial_result[1]
            found_parameters.append(paramstr)
            if paramstr not in self._known_parameters:
                raise ValueError(f'Can not add a result for {paramstr}, no '
                                 'such parameter registered in this '
                                 'measurement.')
            param_spec = self.parameters[paramstr]
            if param_spec.type == 'array':
                inserting_as_arrays = True
            if any(isinstance(value, typ) for typ in array_like_types):

                value = cast(np.ndarray, partial_result[1])
                value = np.atleast_1d(value)
                array_size = len(value.ravel())
                if param_spec.type != 'array' and array_size > 1:
                    inserting_unrolled_array = True
                if input_size > 1 and input_size != array_size:
                    raise ValueError('Incompatible array dimensions. Trying to'
                                     f' add arrays of dimension {input_size} '
                                     f'and {array_size}')
                else:
                    input_size = array_size
            elif is_number(value):
                if inserting_as_arrays:
                    raise ValueError("Trying to insert into an ArrayType with "
                                     "a scalar value")
                if param_spec.type == 'text':
                    raise ValueError(f"It is not possible to save a numeric "
                                     f"value for parameter {paramstr!r} "
                                     f"because its type class is "
                                     f"'text', not 'numeric'.")
            elif isinstance(value, str):
                if param_spec.type != 'text':
                    raise ValueError(f"It is not possible to save a string "
                                     f"value for parameter {paramstr!r} "
                                     f"because its type class is "
                                     f"{param_spec.type!r}, not 'text'.")
            else:
                raise ValueError('Wrong value type received. '
                                 f'Got {type(value)}, but only int, float, '
                                 'str, tuple, list, and np.ndarray is '
                                 'allowed.')

            # Now check for missing setpoints
            if paramstr in self._known_dependencies.keys():
                stuffweneed = set(self._known_dependencies[paramstr])
                stuffwehave = set(found_parameters)
                if not stuffweneed.issubset(stuffwehave):
                    raise ValueError('Can not add this result; missing '
                                     f'setpoint values for {paramstr}:'
                                     f' {stuffweneed}.'
                                     f' Values only given for'
                                     f' {found_parameters}.')

        if inserting_unrolled_array and inserting_as_arrays:
            raise RuntimeError("Trying to insert multiple data values both "
                               "in array from and as numeric. This is not "
                               "possible.")
        elif inserting_as_arrays:
            input_size = 1

        self._append_results(res, input_size)

        if monotonic() - self._last_save_time > self.write_period:
            self.flush_data_to_database()
            self._last_save_time = monotonic()

    def _append_results(self, res: Sequence[res_type],
                        input_size: int) -> None:
        """
        A private method to add the data to actual queue of data to be written.

        Args:
            res: A sequence of the data to be added
            input_size: The length of the data to be added. 1 if its
             to be inserted as arrays.
        """
        for index in range(input_size):
            res_dict = {}
            for partial_result in res:
                param = str(partial_result[0])
                value = partial_result[1]
                param_spec = self.parameters[param]
                if param_spec.type == 'array' and index == 0:
                    res_dict[param] = value
                elif param_spec.type != 'array':
                    # For compatibility with the old Loop, setpoints are
                    # tuples of numbers (usually tuple(np.linspace(...))
                    if hasattr(value, '__len__') and not isinstance(value, str):
                        value = cast(Union[Sequence, np.ndarray], value)
                        if isinstance(value, np.ndarray):
                            # this is significantly faster than atleast_1d
                            # espcially for non 0D arrays
                            # because we already know that this is a numpy
                            # array and just one numpy array. atleast_1d
                            # performs additional checks.
                            if value.ndim == 0:
                                value = value.reshape(1)
                            value = value.ravel()
                        res_dict[param] = value[index]
                    else:
                        res_dict[param] = value
            if len(res_dict) > 0:
                self._results.append(res_dict)

    def _unbundle_arrayparameter(self,
                                 parameter: ArrayParameter,
                                 res: List[res_type],
                                 found_parameters: List[str]) -> None:
        """
        Extract the setpoints from an ArrayParameter and add them to results
         as a regular parameter tuple.

        Args:
            parameter: The ArrayParameter to extract setpoints from.
            res: The result list to add to. Note that this is modified inplace
            found_parameters: The list of all parameters that we know of by now
              Note that this is modified in place.
        """
        # TODO (WilliamHPNielsen): The following code block is ugly and
        # brittle and should be enough to convince us to abandon the
        # design of ArrayParameters (possibly) containing (some of) their
        # setpoints

        sp_names = parameter.setpoint_full_names
        fallback_sp_name = f"{parameter.full_name}_setpoint"
        self._unbundle_setpoints_from_param(parameter, sp_names,
                                            fallback_sp_name,
                                            parameter.setpoints,
                                            res, found_parameters)


    def _unbundle_setpoints_from_param(self, parameter: _BaseParameter,
                                       sp_names: Sequence[str],
                                       fallback_sp_name: str,
                                       setpoints: Sequence,
                                       res: List[res_type],
                                       found_parameters: List[str]):
        """
        Private function to unbundle setpoints from an ArrayParameter or
         a subset of a MultiParameter.

        Args:
            parameter:
            sp_names: Names of the setpoint axes
            fallback_sp_name:  Fallback name for setpoints in case sp_names
              is None. The axis num is appended to this name to ensure all
              setpoint axes names are unique.
            setpoints: The actual setpoints i.e. `parameter.setpoints` for an
              ArrayParameter and `parameter.setpoints[i]` for a MultiParameter
            res: The result list the unpacked setpoints are added too.
              Note that this will be modified in place.
            found_parameters: The list of all parameters that we know of by now
              This is modified in place with new parameters found here.
        """

        setpoint_axes = []
        setpoint_meta = []
        if setpoints is None:
            raise RuntimeError(f"{parameter.full_name} is an {type(parameter)} "
                               f"without setpoints. Cannot handle this.")
        for i, sps in enumerate(setpoints):
            if sp_names is not None:
                spname = sp_names[i]
            else:
                spname = f'{fallback_sp_name}_{i}'

            if spname not in self.parameters.keys():
                raise RuntimeError('No setpoints registered for '
                                   f'{type(parameter)} {parameter.full_name}!')
            sps = np.array(sps)
            while sps.ndim > 1:
                # The outermost setpoint axis or an nD param is nD
                # but the innermost is 1D. In all cases we just need
                # the axis along one dim, the innermost one.
                sps = sps[0]

            setpoint_meta.append(spname)
            found_parameters.append(spname)
            setpoint_axes.append(sps)
        output_grids = np.meshgrid(*setpoint_axes, indexing='ij')
        for grid, meta in zip(output_grids, setpoint_meta):
            res.append((meta, grid))

    def _unbundle_multiparameter(self,
                                 parameter: MultiParameter,
                                 data: Union[tuple, list, np.ndarray],
                                 res: List[res_type],
                                 found_parameters: List[str]) -> None:

        """
        Extract the subarrays and setpoints from an MultiParameter and
         add them to res as a regular parameter tuple.

        Args:
            parameter: The MultiParameter to extract from
            data: The acquired data for this parameter
            res: The result list that the unpacked data and setpoints
              is added too. Note that this will be modified in place.
            found_parameters: The list of all parameters that we know of by now
              This is modified in place with new parameters found here.
        """
        for i in range(len(parameter.shapes)):
            shape = parameter.shapes[i]
            res.append((parameter.names[i], data[i]))
            if shape != ():
                # array parameter like part of the multiparameter
                # need to find setpoints too
                fallback_sp_name = f'{parameter.full_names[i]}_setpoint'

                if parameter.setpoint_full_names[i] is not None:
                    sp_names = parameter.setpoint_full_names[i]
                else:
                    sp_names = None

                self._unbundle_setpoints_from_param(parameter, sp_names,
                                                    fallback_sp_name,
                                                    parameter.setpoints[i],
                                                    res, found_parameters)

    def flush_data_to_database(self) -> None:
        """
        Write the in-memory results to the database.
        """
        log.debug('Flushing to database')
        if self._results != []:
            try:
                write_point = self._dataset.add_results(self._results)
                log.debug(f'Successfully wrote from index {write_point}')
                self._results = []
            except Exception as e:
                log.warning(f'Could not commit to database; {e}')
        else:
            log.debug('No results to flush')

    @property
    def run_id(self) -> int:
        return self._dataset.run_id

    @property
    def points_written(self) -> int:
        return self._dataset.number_of_results

    @property
    def dataset(self):
        return self._dataset


class Runner:
    """
    Context manager for the measurement.
    Lives inside a Measurement and should never be instantiated
    outside a Measurement.

    This context manager handles all the dirty business of writing data
    to the database. Additionally, it may perform experiment bootstrapping
    and clean-up after the measurement.
    """

    def __init__(
            self, enteractions: List, exitactions: List,
            experiment: Experiment = None, station: Station = None,
            write_period: numeric_types = None,
            parameters: Dict[str, ParamSpec] = None,
            name: str = '',
            subscribers: Sequence[Tuple[Callable,
                                        Union[MutableSequence,
                                              MutableMapping]]] = None) -> None:

        self.enteractions = enteractions
        self.exitactions = exitactions
        self.subscribers: Sequence[Tuple[Callable,
                                         Union[MutableSequence,
                                               MutableMapping]]]
        if subscribers is None:
            self.subscribers = []
        else:
            self.subscribers = subscribers
        self.experiment = experiment
        self.station = station
        self.parameters = parameters
        # here we use 5 s as a sane default, but that value should perhaps
        # be read from some config file
        self.write_period = float(write_period) \
            if write_period is not None else 5.0
        self.name = name if name else 'results'

    def __enter__(self) -> DataSaver:
        # TODO: should user actions really precede the dataset?
        # first do whatever bootstrapping the user specified
        for func, args in self.enteractions:
            func(*args)

        # next set up the "datasaver"
        if self.experiment is not None:
            self.ds = qc.new_data_set(
                self.name, self.experiment.exp_id, conn=self.experiment.conn
            )
        else:
            self.ds = qc.new_data_set(self.name)

        # .. and give the dataset a snapshot as metadata
        if self.station is None:
            station = qc.Station.default
        else:
            station = self.station

        if station:
            self.ds.add_metadata('snapshot',
                                 json.dumps({'station': station.snapshot()}))

        if self.parameters is not None:
            for paramspec in self.parameters.values():
                self.ds.add_parameter(paramspec)
        else:
            raise RuntimeError("No parameters supplied")

        # register all subscribers
        for (callble, state) in self.subscribers:
            # We register with minimal waiting time.
            # That should make all subscribers be called when data is flushed
            # to the database
            log.debug(f'Subscribing callable {callble} with state {state}')
            self.ds.subscribe(callble, min_wait=0, min_count=1, state=state)

        print(f'Starting experimental run with id: {self.ds.run_id}')

        self.datasaver = DataSaver(dataset=self.ds,
                                   write_period=self.write_period,
                                   parameters=self.parameters)

        return self.datasaver

    def __exit__(self, exception_type, exception_value, traceback) -> None:

        self.datasaver.flush_data_to_database()

        # perform the "teardown" events
        for func, args in self.exitactions:
            func(*args)

        # and finally mark the dataset as closed, thus
        # finishing the measurement
        self.ds.mark_complete()

        self.ds.unsubscribe_all()


class Measurement:
    """
    Measurement procedure container

    Attributes:
        name (str): The name of this measurement/run. Is used by the dataset
            to give a name to the results_table.
    """

    def __init__(self, exp: Optional[Experiment] = None,
                 station: Optional[qc.Station] = None) -> None:
        """
        Init

        Args:
            exp: Specify the experiment to use. If not given
                the default one is used.
            station: The QCoDeS station to snapshot. If not given, the
                default one is used.
        """
        self.exitactions: List[Tuple[Callable, Sequence]] = []
        self.enteractions: List[Tuple[Callable, Sequence]] = []
        self.subscribers: List[Tuple[Callable, Union[MutableSequence,
                                                     MutableMapping]]] = []
        self.experiment = exp
        self.station = station
        self.parameters: Dict[str, ParamSpec] = OrderedDict()
        self._write_period: Optional[float] = None
        self.name = ''

    @property
    def write_period(self) -> float:
        return self._write_period

    @write_period.setter
    def write_period(self, wp: numeric_types) -> None:
        if not isinstance(wp, Number):
            raise ValueError('The write period must be a number (of seconds).')
        wp_float = float(wp)
        if wp_float < 1e-3:
            raise ValueError('The write period must be at least 1 ms.')
        self._write_period = wp_float

    def _registration_validation(
            self, name: str, setpoints: Sequence[str] = None,
            basis: Sequence[str] = None) -> Tuple[List[str], List[str]]:
        """
        Helper function to do all the validation in terms of dependencies
        when adding parameters, e.g. that no setpoints have setpoints etc.

        Called by register_parameter and register_custom_parameter

        Args:
            name: Name of the parameter to register
            setpoints: name(s) of the setpoint parameter(s)
            basis: name(s) of the parameter(s) that this parameter is
                inferred from
        """

        # now handle setpoints
        depends_on = []
        if setpoints:
            for sp in setpoints:
                if sp not in list(self.parameters.keys()):
                    raise ValueError(f'Unknown setpoint: {sp}.'
                                     ' Please register that parameter first.')
                elif sp == name:
                    raise ValueError('A parameter can not have itself as '
                                     'setpoint.')
                elif self.parameters[sp].depends_on != '':
                    raise ValueError("A parameter's setpoints can not have "
                                     f"setpoints themselves. {sp} depends on"
                                     f" {self.parameters[sp].depends_on}")
                else:
                    depends_on.append(sp)

        # now handle inferred parameters
        inf_from = []
        if basis:
            for inff in basis:
                if inff not in list(self.parameters.keys()):
                    raise ValueError(f'Unknown basis parameter: {inff}.'
                                     ' Please register that parameter first.')
                elif inff == name:
                    raise ValueError('A parameter can not be inferred from'
                                     'itself.')
                else:
                    inf_from.append(inff)

        return depends_on, inf_from

    def register_parameter(
            self, parameter: _BaseParameter,
            setpoints: setpoints_type = None,
            basis: setpoints_type = None,
            paramtype: str = 'numeric') -> None:
        """
        Add QCoDeS Parameter to the dataset produced by running this
        measurement.

        TODO: Does not handle metadata yet

        Args:
            parameter: The parameter to add
            setpoints: The Parameter representing the setpoints for this
                parameter. If this parameter is a setpoint,
                it should be left blank
            basis: The parameters that this parameter is inferred from. If
                this parameter is not inferred from any other parameters,
                this should be left blank.
            paramtype: type of the parameter, i.e. the SQL storage class
        """
        # input validation
        if paramtype not in ParamSpec.allowed_types:
            raise RuntimeError("Trying to register a parameter with type "
                               f"{paramtype}. However, only "
                               f"{ParamSpec.allowed_types} are supported.")
        if not isinstance(parameter, _BaseParameter):
            raise ValueError('Can not register object of type {}. Can only '
                             'register a QCoDeS Parameter.'
                             ''.format(type(parameter)))
        # perhaps users will want a different name? But the name must be unique
        # on a per-run basis
        # we also use the name below, but perhaps is is better to have
        # a more robust Parameter2String function?
        name = str(parameter)
        if isinstance(parameter, ArrayParameter):
            self._register_arrayparameter(parameter,
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
            self._register_parameter(name,
                                     parameter.label,
                                     parameter.unit,
                                     setpoints,
                                     basis, paramtype)
        else:
            raise RuntimeError("Does not know how to register a parameter"
                               f"of type {type(parameter)}")

    def _register_parameter(self, name: str,
                            label: str,
                            unit: str,
                            setpoints: setpoints_type,
                            basis: setpoints_type,
                            paramtype: str) -> None:
        """
        Generate ParamSpecs and register them for an individual parameter
        """

        if setpoints is not None:
            sp_strings = [str(sp) for sp in setpoints]
        else:
            sp_strings = []

        if basis is not None:
            bs_strings = [str(bs) for bs in basis]
        else:
            bs_strings = []

        # validate all dependencies
        depends_on, inf_from = self._registration_validation(name, sp_strings,
                                                             bs_strings)
        paramspec = ParamSpec(name=name,
                              paramtype=paramtype,
                              label=label,
                              unit=unit,
                              inferred_from=inf_from,
                              depends_on=depends_on)
        # ensure the correct order
        if name in self.parameters.keys():
            self.parameters.pop(name)
        self.parameters[name] = paramspec
        log.info(f'Registered {name} in the Measurement.')

    def _register_arrayparameter(self,
                                 parameter: ArrayParameter,
                                 setpoints: setpoints_type,
                                 basis: setpoints_type,
                                 paramtype: str, ) -> None:
        """
        Register an Array paramter and the setpoints belonging to the
        ArrayParameter
        """
        name = str(parameter)
        my_setpoints = list(setpoints) if setpoints else []
        for i in range(len(parameter.shape)):
            if parameter.setpoint_full_names is not None and \
                    parameter.setpoint_full_names[i] is not None:
                spname = parameter.setpoint_full_names[i]
            else:
                spname = f'{name}_setpoint_{i}'
            if parameter.setpoint_labels:
                splabel = parameter.setpoint_labels[i]
            else:
                splabel = ''
            if parameter.setpoint_units:
                spunit = parameter.setpoint_units[i]
            else:
                spunit = ''

            sp = ParamSpec(name=spname, paramtype=paramtype,
                           label=splabel, unit=spunit)

            self.parameters[spname] = sp

            my_setpoints += [spname]

        self._register_parameter(name,
                                 parameter.label,
                                 parameter.unit,
                                 my_setpoints,
                                 basis,
                                 paramtype)

    def _register_multiparameter(self,
                                 multiparameter: MultiParameter,
                                 setpoints: setpoints_type,
                                 basis: setpoints_type,
                                 paramtype: str) -> None:
        """
        Find the individual multiparameter components and their setpoints
        and register these
        """
        setpoints_lists = []
        for i in range(len(multiparameter.shapes)):
            shape = multiparameter.shapes[i]
            name = multiparameter.full_names[i]
            if shape is ():
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

                    sp = ParamSpec(name=spname, paramtype=paramtype,
                                   label=splabel, unit=spunit)

                    self.parameters[spname] = sp
                    my_setpoints += [spname]

            setpoints_lists.append(my_setpoints)

        for i, setpoints in enumerate(setpoints_lists):
            self._register_parameter(multiparameter.names[i],
                                     multiparameter.labels[i],
                                     multiparameter.units[i],
                                     setpoints,
                                     basis,
                                     paramtype)

    def register_custom_parameter(
            self, name: str,
            label: str = None, unit: str = None,
            basis: setpoints_type = None,
            setpoints: setpoints_type = None,
            paramtype: str = 'numeric') -> None:
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
            paramtype: type of the parameter, i.e. the SQL storage class
        """
        self._register_parameter(name,
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

        if param not in self.parameters:
            log.info(f'Tried to unregister {param}, but it was not'
                     'registered.')
            return

        for name, paramspec in self.parameters.items():
            if param in paramspec.depends_on:
                raise ValueError(f'Can not unregister {param}, it is a '
                                 f'setpoint for {name}')
            if param in paramspec.inferred_from:
                raise ValueError(f'Can not unregister {param}, it is a '
                                 f'basis for {name}')

        self.parameters.pop(param)
        log.info(f'Removed {param} from Measurement.')

    def add_before_run(self, func: Callable, args: tuple) -> None:
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

    def add_after_run(self, func: Callable, args: tuple) -> None:
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

    def add_subscriber(self,
                       func: Callable,
                       state: Union[MutableSequence, MutableMapping]) -> None:
        """
        Add a subscriber to the dataset of the measurement.

        Args:
            func: A function taking three positional arguments: a list of
                tuples of parameter values, an integer, a mutable variable
                (list or dict) to hold state/writes updates to.
            state: The variable to hold the state.
        """
        self.subscribers.append((func, state))

    def run(self) -> Runner:
        """
        Returns the context manager for the experimental run
        """
        return Runner(self.enteractions, self.exitactions,
                      self.experiment, station=self.station,
                      write_period=self._write_period,
                      parameters=self.parameters,
                      name=self.name,
                      subscribers=self.subscribers)
