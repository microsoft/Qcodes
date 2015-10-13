import time
from datetime import datetime, timedelta
import asyncio
import multiprocessing as mp

from qcodes.utils.metadata import Metadatable
from qcodes.utils.helpers import make_unique, wait_secs
from qcodes.utils.sync_async import mock_sync
from qcodes.storage import get_storage_manager
from qcodes.sweep_storage import MergedCSVStorage


class Station(Metadatable):
    default = None

    def __init__(self, *instruments, storage_manager=None,
                 storage_class=MergedCSVStorage, monitor=None, default=True):
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

        self.storage_manager = storage_manager or get_storage_manager()
        self.storage_class = storage_class
        self.monitor = monitor

    def add_instrument(self, instrument, name):
        name = make_unique(str(name), self.instruments)
        self.instruments[name] = instrument
        return name

    def set_measurement(self, *args, **kwargs):
        '''
        create a MeasurementSet linked to this Station
        and set it as the default for this station
        '''
        self._measurement_set = MeasurementSet(*args, station=self, **kwargs)
        return self._measurement_set

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


def get_bg_sweep():
    processes = mp.active_children()
    sweeps = [p for p in processes if getattr(p, 'is_sweep', False)]

    if len(sweeps) == 1:
        return sweeps[0]

    if len(sweeps):
        raise RuntimeError(
            'Oops, multiple sweeps are running, how did that happen?')

    return None


class MeasurementSet(object):
    '''
    create a collection of parameters to measure and sweep

    args are a list of parameters, which can be instrument Parameters
    or any other object with `get` and `get_async` methods
    or they can be strings of the form '{instrument}:{parameter}'
    as they are known to this MeasurementSet's linked Station
    '''

    HALT = 'HALT SWEEP'

    def __init__(self, *args, station=None, storage_manager=None,
                 monitor=None, storage_class=None):
        self._station = station
        if station:
            self._storage_manager = storage_manager or station.storage_manager
            self._monitor = monitor or station.monitor
            self._storage_class = storage_class or station.storage_class
        else:
            self._storage_manager = storage_manager
            self._monitor = monitor
            self._storage_class = storage_class
        self._parameters = [self._pick_param(arg) for arg in args]

        self.sweep_process = None

        self.signal_queue = mp.Queue()  # for communicating with bg sweep

    def _pick_param(self, param_or_string):
        if isinstance(param_or_string, str):
            instrument_name, param_name = param_or_string.split(':')
            return self._station[instrument_name][param_name]
        else:
            return param_or_string

    def get(self):
        return tuple(p.get() for p in self._parameters)

    @asyncio.coroutine
    def get_async(self):
        outputs = (p.get_async() for p in self._parameters)
        return (yield from asyncio.gather(*outputs))

    def sweep(self, *args, location=None, storage_class=None,
              background=True, use_async=True, enqueue=False):
        '''
        execute a sweep, measuring this MeasurementSet at each point

        args:
            SweepValues1, delay1, SweepValues2, delay2, ...
            The first two are the outer loop, the next two are
            nested inside it, etc
        location: the location of the dataset, a string whose meaning
            depends on the particular SweepStorage class we're using
        storage_class: subclass of SweepStorage to use for storing this sweep
        background: (default True) run this sweep in a separate process
            so we can have live plotting and other analysis in the main process
        use_async: (default True): execute the sweep asynchronously as much
            as possible
        enqueue: (default False): wait for a previous background sweep to
            finish? If false, will raise an error if another sweep is running

        returns:
            a SweepStorage object that we can use to plot
        '''

        prev_sweep = get_bg_sweep()

        if prev_sweep:
            if enqueue:
                prev_sweep.join()  # wait until previous sweep finishes
            else:
                raise RuntimeError(
                    'a sweep is already running in the background')

        self._init_sweep(args, location=location, storage_class=storage_class)

        sweep_fn = mock_sync(self._sweep_async) if use_async else self._sweep

        # clear any lingering signal queue items
        while not self.signal_queue.empty():
            self.signal_queue.get()

        if background:
            if not self._storage_manager:
                raise RuntimeError('sweep can only run in the background '
                                   'if it has a storage manager running')

            # start the sweep in a new process
            # TODO: in notebooks, errors in a background sweep will just appear
            # the next time a command is run. Do something better?
            # (like log them somewhere, show in monitoring window)?
            p = mp.Process(target=sweep_fn, daemon=True)
            p.start()

            # flag this as a sweep process, and connect in its storage object
            # so you can always find running sweeps and data even if you
            # don't have a Station or MeasurementSet
            p.is_sweep = True
            p.storage = self._storage
            p.measurement = self
            self.sweep_process = p

        else:
            sweep_fn()

        self._storage.sync_live()
        return self._storage

    def _init_sweep(self, args, location=None, storage_class=None):
        if len(args) < 2:
            raise TypeError('need at least one SweepValues, delay pair')
        sweep_vals, delays = args[::2], args[1::2]
        if len(sweep_vals) != len(delays):
            raise TypeError('SweepValues without matching delay')

        # find the output array size we need
        self._dim_size = [len(v) for v in sweep_vals]
        self._sweep_params = [v.name for v in sweep_vals]
        all_params = self._sweep_params + [p.name for p in self._parameters]

        # do any of the sweep params support feedback?
        self._feedback = [v for v in sweep_vals if hasattr(v, 'feedback')
                          and callable(v.feedback)]

        if storage_class is None:
            storage_class = self._storage_class
        self._storage = storage_class(location,
                                      param_names=all_params,
                                      dim_sizes=self._dim_size,
                                      storage_manager=self._storage_manager,
                                      passthrough=True)

        self._sweep_def = tuple(zip(sweep_vals, delays))
        self._sweep_depth = len(sweep_vals)

    def _store(self, indices, set_values, measured):
        self._storage.set_point(indices, set_values + tuple(measured))

        # for adaptive sampling - pass this measurement back to
        # any sweep param that supports feedback
        for vals in self._feedback:
            vals.feedback(set_values, measured)

    def _check_signal(self):
        while not self.signal_queue.empty():
            signal = self.signal_queue.get()
            if signal == self.HALT:
                raise KeyboardInterrupt('sweep was halted')

    def _sweep(self, indices=(), current_values=()):
        self._check_signal()

        current_depth = len(indices)

        if current_depth == self._sweep_depth:
            measured = self.get()
            self._store(indices, current_values, measured)

        else:
            values, delay = self._sweep_def[current_depth]
            for i, value in enumerate(values):
                if i or not current_depth:
                    values.set(value)

                    # if we're changing an outer loop variable,
                    # also change any inner variables before waiting
                    for inner_values, _ in self._sweep_def[current_depth + 1:]:
                        inner_values.set(inner_values[0])

                    finish_datetime = datetime.now() + timedelta(seconds=delay)

                    if self._monitor:
                        self._monitor.call(finish_by=finish_datetime)

                    self._check_signal()
                    time.sleep(wait_secs(finish_datetime))

                # sweep the next level
                self._sweep(indices + (i,), current_values + (value,))

        if not current_depth:
            self._storage.close()

    def _sweep_async(self, indices=(), current_values=()):
        self._check_signal()

        current_depth = len(indices)

        if current_depth == self._sweep_depth:
            measured = yield from self.get_async()
            self._store(indices, current_values, measured)

        else:
            values, delay = self._sweep_def[current_depth]
            for i, value in enumerate(values):
                if i or not current_depth:
                    setters = [values.set_async(value)]

                    # if we're changing an outer loop variable,
                    # also change any inner variables before waiting
                    for inner_values, _ in self._sweep_def[current_depth + 1:]:
                        setters.append(inner_values.set_async(inner_values[0]))

                    yield from asyncio.gather(*setters)

                    finish_datetime = datetime.now() + timedelta(seconds=delay)

                    if self._monitor:
                        yield from self._monitor.call_async(
                            finish_by=finish_datetime)

                    self._check_signal()
                    yield from asyncio.sleep(wait_secs(finish_datetime))

                # sweep the next level
                yield from self._sweep_async(indices + (i,),
                                             current_values + (value,))

        if not current_depth:
            self._storage.close()

    def halt_sweep(self, timeout=5):
        sweep = get_bg_sweep()
        if not sweep:
            print('No sweep running')
            return

        self.signal_queue.put(self.HALT)
        sweep.join(timeout)

        if sweep.is_alive():
            sweep.terminate()
            print('Background sweep did not respond, terminated')
