import json
import logging
from time import monotonic
from collections import OrderedDict
from typing import Callable, Union, Dict, Tuple, List, Sequence, Type
from inspect import signature
import numpy as np

import qcodes as qc
from qcodes import Station, Parameter
from qcodes.instrument.parameter import ArrayParameter
from qcodes.dataset.experiment_container import Experiment
from qcodes.dataset.param_spec import ParamSpec
from qcodes.dataset.data_set import DataSet

log = logging.getLogger(__name__)


class ParameterTypeError(Exception):
    pass


class DataSaver:
    """
    The class used byt the Runner context manager to handle the
    datasaving to the database
    """

    def __init__(self, dataset: DataSet, write_period: float,
                 known_parameters: List[str]) -> None:
        self._dataset = dataset
        self.write_period = write_period
        self._known_parameters = known_parameters
        self._results: List[dict] = []  # will be filled by addResult
        self._last_save_time = monotonic()

    def addResult(self,
                  *res: Tuple[Union[Parameter, str],
                              Union[str, int, float, np.ndarray]])-> None:
        """
        Add a result to the measurement results. Represents a measurement
        point in the space of measurement parameters, e.g. in an experiment
        varying two voltages and measuring two currents, a measurement point
        is the four dimensional (v1, v2, c1, c2). The corresponding call
        to this function would be (e.g.)
        >> addResult((v1, 0.1), (v2, 0.2), (c1, 5), (c2, -2.1))

        For better performance, this function does not immediately write to
        the database, but keeps the results in memory. Writing happens every
        `write_period` seconds and during the __exit__ method if this class.

        Args:
            res: a dictionary with keys that are parameter names and items
                that are the corresponding values at this measurement point.

        Raises:
            ValueError: if a parameter name not registered in the parent
                Measurement object is encountered.
            ParameterTypeError: if a parameter is given a value not matching
                its type.
        """
        res_dict = {}

        for partial_result in res:
            # TODO: Here we again use the str(), which may not be terrific
            res_dict.update({str(partial_result[0]): partial_result[1]})

        self._results.append(res_dict)
        if monotonic() - self._last_save_time > self.write_period:
            self.flushDataToDatabase()
            self._last_save_time = monotonic()

    def flushDataToDatabase(self):
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

    @property
    def id(self):
        return self._dataset.id


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
            self, enteractions: OrderedDict, exitactions: OrderedDict,
            experiment: Experiment=None, station: Station=None,
            write_period: float=None,
            parameters: Dict[str, ParamSpec]=None,
            saver_class: Type=DataSaver) -> None:

        self.enteractions = enteractions
        self.exitactions = exitactions
        self.experiment = experiment
        self.station = station
        self.parameters = parameters
        # here we use 5 s as a sane default, but that value should perhaps
        # be read from some config file
        self.write_period = write_period if write_period is not None else 5
        self._saver_class = saver_class

    def __enter__(self) -> DataSaver:
        # TODO: should user actions really precede the dataset?
        # first do whatever bootstrapping the user specified
        for func, args in self.enteractions.items():
            func(*args)

        # next set up the "datasaver"
        if self.experiment:
            eid = self.experiment.id
        else:
            eid = None

        self.ds = qc.new_data_set('name', eid)

        # .. and give it a snapshot as metadata
        if self.station is None:
            station = qc.Station.default
        else:
            station = self.station

        self.ds.add_metadata('snapshot', json.dumps(station.snapshot()))

        for paramspec in self.parameters.values():
            self.ds.add_parameter(paramspec)

        print(f'Starting experimental run with id: {self.ds.id}')

        self.datasaver = self._saver_class(
            self.ds, self.write_period, list(self.parameters.keys())
        )

        return self.datasaver

    def __exit__(self, exception_type, exception_value, traceback) -> None:

        self.datasaver.flushDataToDatabase()

        # perform the "teardown" events
        for func, args in self.exitactions.items():
            func(*args)

        # and finally mark the dataset as closed, thus
        # finishing the measurement
        self.ds.mark_complete()


class Measurement:
    """
    Measurement procedure container
    """
    def __init__(self, exp: Experiment=None, station=None) -> None:
        """
        Init

        Args:
            exp: Specify the experiment to use. If not given
                the default one is used
            station: The QCoDeS station to snapshot
        """
        self.exp = exp
        self.exitactions: Dict[Callable, Sequence] = OrderedDict()
        self.enteractions: Dict[Callable, Sequence] = OrderedDict()
        self.experiment = exp
        self.station = station
        self.parameters: Dict[str, ParamSpec] = OrderedDict()

    def registerParameter(
            self, parameter: Parameter,
            setpoints: Tuple[Parameter]=None,
            basis: Tuple[Parameter]=None) -> None:
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
        # perhaps users will want a different name? But the name must be unique
        # on a per-run basis
        # we also use str(parameter) below, but perhaps is is better to have
        # a more robust Parameter2String function?
        name = str(parameter)
        # the next one is tricky and deserves some thought
        # TODO: How to handle types?
        if isinstance(parameter, ArrayParameter):
            paramtype = 'array'
        else:
            paramtype = 'number'
        label = parameter.label
        unit = parameter.unit

        # now handle setpoints
        depends_on = []
        if setpoints:
            for sp in setpoints:
                if str(sp) not in list(self.parameters.keys()):
                    raise ValueError(f'Unknown setpoint: {str(sp)}.'
                                     ' Please register that parameter first.')
                else:
                    depends_on.append(str(sp))

        # now handle inferred parameters
        inf_from = []
        if basis:
            for inff in basis:
                if str(inff) not in list(self.parameters.keys()):
                    raise ValueError(f'Unknown basis parameter: {str(inff)}.'
                                     ' Please register that parameter first.')
                else:
                    inf_from.append(str(inff))

        paramspec = ParamSpec(name=name,
                              paramtype=paramtype,
                              label=label,
                              unit=unit,
                              inferred_from=inf_from,
                              depends_on=depends_on)

        self.parameters[str(parameter)] = paramspec

    def addBeforeRun(self, func: Callable, args: tuple) -> None:
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

        self.enteractions[func] = args

    def addAfterRun(self, func: Callable, args: tuple) -> None:
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

        self.exitactions[func] = args

    def run(self):
        """
        Returns the context manager for the experimental run
        """
        return Runner(self.enteractions, self.exitactions,
                      self.experiment,
                      parameters=self.parameters)
