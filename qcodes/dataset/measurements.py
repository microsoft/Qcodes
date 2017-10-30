import json
from collections import OrderedDict
from typing import Callable
from inspect import signature

import qcodes as qc
from qcodes import Station
from qcodes.dataset.experiment_container import Experiment


class Runner:
    """
    Context manager for the measurement.
    Lives inside a Measurement and should never be instantiated
    outside a Measurement.
    """
    def __init__(self, enteractions: OrderedDict, exitactions: OrderedDict,
                 experiment: Experiment=None, station: Station=None) -> None:
        self.enteractions = enteractions
        self.exitactions = exitactions
        self.experiment = experiment
        self.station = station

    def __enter__(self) -> None:
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

        return self.ds

    def __exit__(self, exception_type, exception_value, traceback) -> None:
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
        # TODO: The sequence of actions probably matters A LOT
        self.exp = exp
        self.exitactions = OrderedDict()  # key: function, item: args
        self.enteractions = OrderedDict()  # key: function, item: args
        self.experiment = exp
        self.station = station

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
                      self.experiment)
