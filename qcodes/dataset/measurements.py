import json
from collections import OrderedDict

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

        print('-'*25)
        print('Finished dataset')
        print(self.ds)
