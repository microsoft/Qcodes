from datetime import datetime, timedelta
import asyncio
import multiprocessing as mp

from qcodes.utils.metadata import Metadatable
from qcodes.utils.helpers import is_function, make_unique
from qcodes.storage import get_storage_manager


class Station(Metadatable):
    default = None

    def __init__(self, *instruments, storage=None, monitor=None, default=True):
        # when a new station is defined, store it in a class variable
        # so it becomes the globally accessible default station for
        # new sweeps etc after that. You can still have multiple stations
        # defined, but to use other than the default one you must specify
        # it explicitly in the MeasurementSet
        # but if for some reason you want this new Station NOT to be the
        # default, just specify default=False
        if default:
            Station.default = self

        self.instruments = {}
        for instrument in instruments:
            self.add_instrument(instrument, instrument.name)

        self.storage = storage or get_storage_manager()
        self.monitor = monitor

        self._share_manager = mp.Manager()
        self._sweep_data = self._share_manager.dict()

    def add_instrument(self, instrument, name):
        name = make_unique(str(name), self.instruments)
        self.instruments[name] = instrument

    def set_measurement(self, *args, **kwargs):
        '''
        create a MeasurementSet linked to this Station
        and set it as the default for this station
        '''
        ms = MeasurementSet(*args, station=self, **kwargs)
        self._measurement_set = ms
        return ms

    def sweep(self, *args, **kwargs):
        '''
        run a sweep using the default MeasurementSet for this Station
        '''
        return self._measurement_set.sweep(*args, **kwargs)

    def __getitem__(self, key):
        '''
        station['someinstrument']
          is a shortcut to:
        station.instruments['someinstrument']
        '''
        return self.instruments[key]


class MeasurementSet(object):
    '''
    create a collection of parameters to measure and sweep

    args are a list of parameters, which can be instrument Parameters
    or any other object with `get` and `get_async` methods
    or they can be strings of the form '{instrument}:{parameter}'
    as they are known to this MeasurementSet's linked Station
    '''
    def __init__(self, *args, station=None, storage=None, monitor=None):
        self._station = station or Station.default
        self._storage = storage or self._station.storage
        self._monitor = monitor or self._station.monitor
        self._storage_class = self._storage.storage_class
        self._parameters = [self._pick_param(arg) for arg in args]

    def _pick_param(self, param_or_string):
        if isinstance(param_or_string, str):
            instrument_name, param_name = param_or_string.split(':')
            return self._station[instrument_name][param_name]
        else:
            return param_or_string

    def get(self):
        return [p.get() for p in self._parameters]

    async def get_async(self):
        outputs = (p.get_async() for p in self._parameters)
        return await asyncio.gather(*outputs)

    def sweep(self, *args, location=None):
        '''
        execute a sweep, measuring this MeasurementSet at each point
        args:
            SweepValues1, delay1, SweepValues2, delay2, ...
            The first two are the outer loop, the next two are
            nested inside it, etc
        location: the location of the dataset, a string whose meaning
            depends on the particular SweepStorage class we're using
        returns:
            a SweepStorage object that we can use to plot
        '''
        self._init_sweep(args, location=location)
        self._sweep()
        # TODO: this returns at the end, but if we put the sweep in
        # another process, we can return this immediately
        return self._storage_class(self.location, self._param_names,
                                   self._dim_size, self._storage)

    async def sweep_async(self, *args, location=None):
        self._init_sweep(args, location=location)
        await self._sweep_async(())
        return self._storage_class(self.location, self._param_names,
                                   self._dim_size, self._storage)

    def _init_sweep(self, args, location=None):
        if len(args) < 2:
            raise TypeError('need at least one SweepValues, delay pair')
        sweep_vals, delays = args[::2], args[1::2]
        if len(sweep_vals) != len(delays):
            raise TypeError('SweepValues without matching delay')

        # find the output array size we need
        self._dim_size = [len(vals) for vals in sweep_vals]
        self._param_names = [vals.name for vals in sweep_vals]
        self.location = self._storage.ask('new_sweep', location,
                                          self._param_names, self._dim_size)

        self._sweep_def = zip(sweep_vals, delays)
        self._sweep_depth = len(sweep_vals)

    def _sweep(self, indices=(), values=()):
        current_depth = len(indices)
        if current_depth == self._sweep_depth:
            self._store(self.get(), indices)
        else:
            values, delay = self._sweep_def[current_depth]
            for i, value in enumerate(values):
                if i or not current_depth:
                    finish_datetime = datetime.now() + timedelta(seconds=delay)
                    values.set(value)

                    # if we're changing an outer loop variable,
                    # also change any inner variables before waiting
                    for inner_values, _ in self._sweep_def[current_depth + 1:]:
                        inner_values.set(inner_values[0])

                    self._station.monitor(finish_by=finish_datetime)

                    time.sleep(wait_secs(finish_datetime))

                # sweep the next level
                self._sweep(indices + (value,))

    def _sweep_async(self, indices):
        current_depth = len(indices)
        if current_depth == self._sweep_depth:
            self._store(await self.get_async(), indices)
        else:
            values, delay = self._sweep_def[current_depth]
            for i, value in enumerate(values):
                if i or not current_depth:
                    finish_datetime = datetime.now() + timedelta(seconds=delay)
                    setters = [values.set_async(value)]

                    # if we're changing an outer loop variable,
                    # also change any inner variables before waiting
                    for inner_values, _ in self._sweep_def[current_depth + 1:]:
                        setters.append(inner_values.set_async(inner_values[0]))

                    await asyncio.gather(setters)
                    await asyncio.sleep(wait_secs(finish_datetime))

                # sweep the next level
                await self._sweep_async(indices + (value,))

    def _store(self, vals, indices):
        pass  # TODO
