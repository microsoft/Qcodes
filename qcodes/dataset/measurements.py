import json
import logging
from time import monotonic
from collections import OrderedDict
from typing import (Callable, Union, Dict, Tuple, List, Sequence, cast,
                    MutableMapping, MutableSequence, Optional)
from inspect import signature
from numbers import Number

import numpy as np

import qcodes as qc
from qcodes import Station
from qcodes.instrument.parameter import ArrayParameter, _BaseParameter, Parameter
from qcodes.dataset.experiment_container import Experiment
from qcodes.dataset.param_spec import ParamSpec
from qcodes.dataset.data_set import DataSet

log = logging.getLogger(__name__)

array_like_types = (tuple, list, np.ndarray)
non_array_like_types = (int, float, str)


class ParameterTypeError(Exception):
    pass


class DataSaver:
    """
    The class used byt the Runner context manager to handle the
    datasaving to the database
    """

    default_callback: Optional[dict] = None

    def __init__(self, dataset: DataSet, write_period: float,
                 parameters: Dict[str, ParamSpec]) -> None:
        self._dataset = dataset
        if DataSaver.default_callback is not None and 'run_tables_subscription_callback' in DataSaver.default_callback:
            callback = DataSaver.default_callback['run_tables_subscription_callback']
            min_wait = DataSaver.default_callback['run_tables_subscription_min_wait']
            min_count = DataSaver.default_callback['run_tables_subscription_min_count']
            snapshot = dataset.get_metadata('snapshot')
            self._dataset.subscribe(callback, min_wait= min_wait, min_count= min_count, state={}, callback_kwargs={'run_id': self._dataset.run_id, 'snapshot': snapshot })
        self.write_period = write_period
        self.parameters = parameters
        self._known_parameters = list(parameters.keys())
        self._results: List[dict] = []  # will be filled by addResult
        self._last_save_time = monotonic()
        self._known_dependencies: Dict[str, List[str]] = {}
        for param, parspec in parameters.items():
            if parspec.depends_on != '':
                self._known_dependencies.update({str(param):
                                                parspec.depends_on.split(', ')})

    def add_result(self,
                   *res_tuple: Tuple[Union[_BaseParameter, str],
                                     Union[str, int, float, np.ndarray]])-> None:
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

        Args:
            res: a dictionary with keys that are parameter names and items
                that are the corresponding values at this measurement point.

        Raises:
            ValueError: if a parameter name not registered in the parent
                Measurement object is encountered.
            ParameterTypeError: if a parameter is given a value not matching
                its type.
        """
        res = list(res_tuple)  # ArrayParameters cause us to mutate the results

        # we iterate through the input twice in order to allow users to call
        # add_result with the arguments in any particular order, i.e. NOT
        # enforcing that setpoints come before dependent variables.
        # Also, we pre-check that array dimensions are compatible before
        # proceeding.
        input_size = 1
        params = []
        for partial_result in res:
            parameter = partial_result[0]
            paramstr = str(partial_result[0])
            value = partial_result[1]
            params.append(paramstr)
            if paramstr not in self._known_parameters:
                raise ValueError(f'Can not add a result for {paramstr}, no '
                                 'such parameter registered in this '
                                 'measurement.')
            if any(isinstance(value, typ) for typ in array_like_types):
                value = cast(np.ndarray, partial_result[1])
                value = np.atleast_1d(value)
                array_size = len(value)
                if input_size > 1 and input_size != array_size:
                    raise ValueError('Incompatible array dimensions. Trying to'
                                     f' add arrays of dimension {input_size} '
                                     f'and {array_size}')
                else:
                    input_size = array_size
            elif any(isinstance(value, t) for t in non_array_like_types):
                pass
            else:
                raise ValueError('Wrong value type received. '
                                 f'Got {type(value)}, but only int, float, '
                                 'str, tuple, list, and np.ndarray is '
                                 'allowed.')

            # TODO (WilliamHPNielsen): The following code block is ugly and
            # brittle and should be enough to convince us to abandon the
            # design of ArrayParameters (possibly) containing (some of) their
            # setpoints
            if isinstance(parameter, ArrayParameter):
                if parameter.setpoints is not None:
                    sps = parameter.setpoints[0]
                else:
                    raise RuntimeError("Got an array parameter without "
                                       "setpoints. Cannot handle this")
                inst_name = getattr(parameter._instrument, 'name', '')
                sp_name_parts = []
                if inst_name is not None:
                    sp_name_parts.append(inst_name)
                if parameter.setpoint_names is not None:
                    sp_name_parts.append(parameter.setpoint_names[0])
                spname = '_'.join(sp_name_parts)

                if f'{paramstr}_setpoint' in self.parameters.keys():
                    res.append((f'{paramstr}_setpoint', sps))
                elif spname in self.parameters.keys():
                        res.append((spname, sps))
                else:
                    raise RuntimeError('No setpoints registered for '
                                       f'ArrayParameter {paramstr}!')

        # Now check for missing setpoints
        for partial_result in res:
            param = str(partial_result[0])
            if param in self._known_dependencies.keys():
                stuffweneed = set(self._known_dependencies[param])
                stuffwehave = set(params)
                if not stuffweneed.issubset(stuffwehave):
                    raise ValueError('Can not add this result; missing '
                                     f'setpoint values for {param}:'
                                     f' {stuffweneed}.'
                                     f' Values only given for {params}.')

        for index in range(input_size):
            res_dict = {}
            for partial_result in res:
                param = str(partial_result[0])
                value = partial_result[1]
                # For compatibility with the old Loop, setpoints are
                # tuples of numbers (usually tuple(np.linspace(...))
                if hasattr(value, '__len__') and not(isinstance(value, str)):
                    value = cast(Union[Sequence,np.ndarray], value)
                    if isinstance(value, np.ndarray):
                        value = np.atleast_1d(value)
                    res_dict.update({param: value[index]})
                else:
                    res_dict.update({param: value})

            self._results.append(res_dict)

        if monotonic() - self._last_save_time > self.write_period:
            self.flush_data_to_database()
            self._last_save_time = monotonic()

    def flush_data_to_database(self):
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
    def run_id(self):
        return self._dataset.run_id

    @property
    def points_written(self):
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
            experiment: Experiment=None, station: Station=None,
            write_period: float=None,
            parameters: Dict[str, ParamSpec]=None,
            name: str='',
            subscribers: List=[]) -> None:

        self.enteractions = enteractions
        self.exitactions = exitactions
        self.subscribers = subscribers
        self.experiment = experiment
        self.station = station
        self.parameters = parameters
        # here we use 5 s as a sane default, but that value should perhaps
        # be read from some config file
        self.write_period = write_period if write_period is not None else 5
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
    def __init__(self, exp: Optional[Experiment]=None,
                 station: Optional[qc.Station]=None) -> None:
        """
        Init

        Args:
            exp: Specify the experiment to use. If not given
                the default one is used.
            station: The QCoDeS station to snapshot. If not given, the
                default one is used.
        """
        self.exp = exp
        self.exitactions: List[Tuple[Callable, Sequence]] = []
        self.enteractions: List[Tuple[Callable, Sequence]] = []
        self.subscribers: List[Tuple[Callable, Union[MutableSequence,
                                                     MutableMapping]]] = []
        self.experiment = exp
        self.station = station
        self.parameters: Dict[str, ParamSpec] = OrderedDict()
        self._write_period: Optional[Number] = None
        self.name = ''

    @property
    def write_period(self):
        return self._write_period

    @write_period.setter
    def write_period(self, wp: Number) -> None:
        if not isinstance(wp, Number):
            raise ValueError('The write period must be a number (of seconds).')
        wp_float = cast(float, wp)
        if wp_float < 1e-3:
            raise ValueError('The write period must be at least 1 ms.')
        self._write_period = wp

    def _registration_validation(
            self, name: str, setpoints: Sequence[str]=None,
            basis: Sequence[str]=None) -> Tuple[List[str], List[str]]:
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

        return (depends_on, inf_from)

    def register_parameter(
            self, parameter: _BaseParameter,
            setpoints: Sequence[_BaseParameter]=None,
            basis: Sequence[_BaseParameter]=None) -> None:
        """
        Add QCoDeS Parameter to the dataset produced by running this
        measurement.

        TODO: Does not handle metadata yet

        Args:
            parameter: The parameter to add
            setpoints: The setpoints for this parameter. If this parameter
                is a setpoint, it should be left blank
            basis: The parameters that this parameter is inferred from. If
                this parameter is not inferred from any other parameters,
                this should be left blank.
        """
        # input validation
        if not isinstance(parameter, _BaseParameter):
            raise ValueError('Can not register object of type {}. Can only '
                             'register a QCoDeS Parameter.'
                             ''.format(type(parameter)))
        # perhaps users will want a different name? But the name must be unique
        # on a per-run basis
        # we also use the name below, but perhaps is is better to have
        # a more robust Parameter2String function?
        name = str(parameter)
        my_setpoints: Optional[Sequence[Union[str, _BaseParameter]]]
        if isinstance(parameter, ArrayParameter):
            parameter = cast(ArrayParameter, parameter)
            spname_parts = []
            if parameter.instrument is not None:
                inst_name = parameter.instrument.name
                if inst_name is not None:
                    spname_parts.append(inst_name)
            if parameter.setpoint_names is not None:
                spname_parts.append(parameter.setpoint_names[0])
            if len(spname_parts) > 0:
                spname = '_'.join(spname_parts)
            else:
                spname = f'{name}_setpoint'
            if parameter.setpoint_labels:
                splabel = parameter.setpoint_labels[0]
            else:
                splabel = ''
            if parameter.setpoint_units:
                spunit = parameter.setpoint_units[0]
            else:
                spunit = ''

            sp = ParamSpec(name=spname, paramtype='numeric',
                           label=splabel, unit=spunit)

            self.parameters[spname] = sp
            my_setpoints = list(setpoints) if setpoints else []
            my_setpoints += [spname]
        else:
            my_setpoints = setpoints

        # We currently treat ALL parameters as 'numeric' and fail to add them
        # to the dataset if they can not be unraveled to fit that description
        # (except strings, we just let those through)
        # this is indeed a limitation, but a sane one. We might loosen that
        # requirement later and start saving binary blobs with the datasaver,
        # but for now binary blob saving is referred to using the DataSet
        # API directly
        parameter = cast(Union[Parameter, ArrayParameter], parameter)
        paramtype = 'numeric'
        label = parameter.label
        unit = parameter.unit

        if my_setpoints is not None:
            sp_strings = [str(sp) for sp in my_setpoints]
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

    def register_custom_parameter(
            self, name: str,
            label: str=None, unit: str=None,
            basis: Sequence[Union[str, _BaseParameter]]=None,
            setpoints: Sequence[Union[str, _BaseParameter]]=None) -> None:
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
        """

        # validate dependencies
        if setpoints:
            sp_strings = [str(sp) for sp in setpoints]
        else:
            sp_strings = []
        if basis:
            bs_strings = [str(bs) for bs in basis]
        else:
            bs_strings = []

        # validate all dependencies
        depends_on, inf_from = self._registration_validation(name, sp_strings,
                                                             bs_strings)

        parspec = ParamSpec(name=name, paramtype='numeric',
                            label=label, unit=unit,
                            inferred_from=inf_from,
                            depends_on=depends_on)

        # ensure the correct order
        if name in self.parameters.keys():
            self.parameters.pop(name)

        self.parameters[name] = parspec

    def unregister_parameter(self,
                             parameter: Union[_BaseParameter, str]) -> None:
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
            name: The name of the subscriber.
            func: A function taking three positional arguments: a list of
                tuples of parameter values, an integer, a mutable variable
                (list or dict) to hold state/writes updates to.
            state: The variable to hold the state.
        """
        # TODO: Should we protect users from registering two subscribers
        # with the same state?
        self.subscribers.append((func, state))

    def run(self):
        """
        Returns the context manager for the experimental run
        """
        return Runner(self.enteractions, self.exitactions,
                      self.experiment, station=self.station,
                      write_period=self._write_period,
                      parameters=self.parameters,
                      name=self.name,
                      subscribers=self.subscribers)
