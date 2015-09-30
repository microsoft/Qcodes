from datetime import datetime, timedelta
import time
import asyncio
import logging

from qcodes.utils.helpers import is_sequence, permissive_range, wait_secs
from qcodes.utils.metadata import Metadatable
from qcodes.utils.sync_async import (mock_async, mock_sync, syncable_command,
                                     NoCommandError)
from qcodes.utils.validators import Validator, Numbers, Ints


class Parameter(Metadatable):
    def __init__(self, instrument, name,
                 get_cmd=None, async_get_cmd=None, parse_function=None,
                 set_cmd=None, async_set_cmd=None, vals=None,
                 sweep_step=None, sweep_delay=None, max_val_age=3600,
                 **kwargs):
        '''
        defines one measurement parameter

        instrument: an instrument that handles this parameter
        name: the local name of this parameter
        get_cmd: a string or function to get this parameter
        async_get_cmd: a function to use for async get, or for both sync
            and async if get_cmd is missing or None
        parse_function: function to transform the response from get
            to the final output value.
            NOTE: only applies if get_cmd is a string. The function forms
            of get_cmd and async_get_cmd should do their own parsing
        set_cmd: command to set this parameter, either:
            - a string (containing one field to .format, like "{}" etc)
            - a function (of one parameter)
        async_set_cmd: a function to use for async set, or for both sync
            and async if set_cmd is missing or None
        vals: a Validator object for this parameter
        sweep_step: max increment of parameter value - larger changes
            are broken into steps this size
        sweep_delay: time (in seconds) to wait after each sweep step
        max_val_age: max time (in seconds) to trust a saved value from
            this parameter as the starting point of a sweep
        '''
        super().__init__(**kwargs)

        self._instrument = instrument
        self.name = name

        # stored value from last .set() or .get()
        # normally only used by set with a sweep, to avoid
        # having to call .get() for every .set()
        self._max_val_age = 0
        self._last_value = None
        self._last_ts = None

        self.has_get = False
        self.has_set = False

        self._set_get(get_cmd, async_get_cmd, parse_function)
        self._set_vals(vals)
        self._set_set(set_cmd, async_set_cmd)
        self.set_sweep(sweep_step, sweep_delay, max_val_age)

        if not (self.has_get or self.has_set):
            raise NoCommandError('neither set nor get cmd found in' +
                                 ' Parameter {}'.format(self.name))

    def snapshot_base(self):
        snap = {}
        if self._last_value is not None:
            snap['value'] = self._last_value
            snap['ts'] = self._last_ts.strftime('%Y-%m-%d %H:%M:%S')
        return snap

    def _save_val(self, value):
        self._last_value = value
        self._last_ts = datetime.now()

    def get(self):
        value = self._get()
        self._save_val(value)
        return value

    @asyncio.coroutine
    def get_async(self):
        value = yield from self._get_async()
        self._save_val(value)
        return value

    def _set_get(self, get_cmd, async_get_cmd, parse_function):
        def no_get():
            raise NotImplementedError(
                'parameter {} has no getter defined'.format(self.name))

        self._get, self._get_async = syncable_command(
            0, get_cmd, async_get_cmd, self._instrument.ask,
            self._instrument.ask_async, parse_function, no_get)

        if self._get is not no_get:
            self.has_get = True

    def _set_vals(self, vals):
        if vals is None:
            self._vals = Numbers()
        elif isinstance(vals, Validator):
            self._vals = vals
        else:
            raise TypeError('vals must be a Validator')

    def _set_set(self, set_cmd, async_set_cmd):
        # note: this does not set the final setter functions. that's handled
        # in self.set_sweep, when we choose a swept or non-swept setter.
        def no_set(value):
            raise NotImplementedError(
                'parameter {} has no setter defined'.format(self.name))

        self._set, self._set_async = syncable_command(
            1, set_cmd, async_set_cmd, self._instrument.write,
            self._instrument.write_async, no_cmd_function=no_set)

        if self._set is not no_set:
            self.has_set = True

    def validate(self, value):
        if not self._vals.is_valid(value):
            raise ValueError(
                '{} is not a valid value for {}'.format(value, self.name))

    def _validate_and_set(self, value):
        self.validate(value)
        self._set(value)
        self._save_val(value)

    @asyncio.coroutine
    def _validate_and_set_async(self, value):
        self.validate(value)
        yield from self._set_async(value)
        self._save_val(value)

    def _sweep_steps(self, value):
        oldest_ok_val = datetime.now() - timedelta(seconds=self._max_val_age)
        if self._last_ts is None or self._last_ts < oldest_ok_val:
            self.get()
        start_value = self._last_value

        self.validate(start_value)

        if not (isinstance(start_value, (int, float)) and
                isinstance(value, (int, float))):
            # something weird... parameter is numeric but one of the ends
            # isn't, even though it's valid.
            # probably a MultiType with a mix of numeric and non-numeric types
            # just set the endpoint and move on
            logging.warning('cannot sweep {} from {} to {} - jumping.'.format(
                self.name, start_value, value))
            return []

        # drop the initial value, we're already there
        return permissive_range(start_value, value, self._sweep_step)[1:]

    def _validate_and_sweep(self, value):
        self.validate(value)
        step_finish_ts = datetime.now()

        for step_val in self._sweep_steps(value):
            self._set(step_val)
            self._save_val(step_val)
            step_finish_ts += timedelta(seconds=self._sweep_delay)
            time.sleep(wait_secs(step_finish_ts))

        self._set(value)
        self._save_val(value)

    @asyncio.coroutine
    def _validate_and_sweep_async(self, value):
        self.validate(value)
        step_finish_ts = datetime.now()

        for step_val in self._sweep_steps(value):
            yield from self._set_async(step_val)
            self._save_val(step_val)
            step_finish_ts += timedelta(seconds=self._sweep_delay)
            yield from asyncio.sleep(wait_secs(step_finish_ts))

        yield from self._set_async(value)
        self._save_val(value)

    def set_sweep(self, sweep_step, sweep_delay, max_val_age=None):
        if sweep_step is not None or sweep_delay is not None:
            if not self._vals.is_numeric:
                raise TypeError('you can only sweep numeric parameters')

            if (isinstance(self._vals, Ints) and
                    not isinstance(sweep_step, int)):
                raise TypeError(
                    'sweep_step must be a positive int for an Ints parameter')
            elif not isinstance(sweep_step, (int, float)):
                raise TypeError('sweep_step must be a positive number')
            if sweep_step <= 0:
                raise ValueError('sweep_step must be positive')

            if not isinstance(sweep_delay, (int, float)):
                raise TypeError('sweep_delay must be a positive number')
            if sweep_delay <= 0:
                raise ValueError('sweep_delay must be positive')

            self._sweep_step = sweep_step
            self._sweep_delay = sweep_delay

            # assign the setters with a sweep
            self.set = self._validate_and_sweep
            self.set_async = self._validate_and_sweep_async
        else:
            # assign the setters as immediate jumps
            self.set = self._validate_and_set
            self.set_async = self._validate_and_set_async

        if max_val_age is not None:
            if not isinstance(max_val_age, (int, float)):
                raise TypeError('max_val_age must be a non-negative number')
            if max_val_age < 0:
                raise ValueError('max_val_age must be non-negative')
            self._max_val_age = max_val_age

    def __getitem__(self, keys):
        '''
        slice a Parameter to get a SweepValues object
        to iterate over during a sweep
        '''
        return SweepValues(self, keys)


class SweepValues(object):
    '''
    a collection of parameter values that can be iterated over
    during a sweep.

    inputs:
        parameter: the target of the sweep, an object with
            set (and/or set_async), and optionally validate methods
        keys: one or a sequence of items, each of which can be:
            - a single parameter value
            - a sequence of parameter values
            - a slice object, which MUST include all three args

    intended use:
        a SweepValues object is normally created by slicing a Parameter p:

        sv = p[1.2:2:0.01]  # slice notation
        sv = p[1, 1.1, 1.3, 1.6]  # explicit individual values
        sv = p[1.2:2:0.01, 2:3:0.02]  # sequence of slices
        sv = p[logrange(1,10,.01)]  # make a function that returns a sequence

        then it is iterated over in a sweep:

        for value in sv:
            sv.set(value)  # or (await / yield from) sv.set_async(value)
                           # set(_async) just shortcuts sv._parameter.set
            sleep(delay)
            measure()

    you can also extend, reverse, add, and copy SweepValues objects:

    sv += p[2:3:.01] (must be the same parameter)
    sv += [4, 5, 6] (a bare sequence)
    sv.extend(p[2:3:.01])
    sv.append(3.2)
    sv.reverse()
    sv2 = reversed(sv)
    sv3 = sv + sv2
    sv4 = sv.copy()

    note though that sweeps should only require set, set_async, and
    __iter__ - ie "for val in sv", so any class that implements these
    may be used in sweeps. That allows things like adaptive sampling,
    where you don't know ahead of time what the values will be or even
    how many there are.
    '''
    def __init__(self, parameter, keys):
        self._parameter = parameter
        self.name = parameter.name
        self._values = []
        keyset = keys if is_sequence(keys) else (keys,)

        for key in keyset:
            if is_sequence(key):
                self._values.extend(key)
            elif isinstance(key, slice):
                if key.start is None or key.stop is None or key.step is None:
                    raise TypeError('all 3 slice parameters are required, ' +
                                    '{} is missing some'.format(key))
                self._values.extend(permissive_range(key.start, key.stop,
                                                     key.step))
            else:
                # assume a single value
                self._values.append(key)

        self._validate(self._values)

        # create the set and set_async shortcuts
        if hasattr(parameter, 'set'):
            self.set = parameter.set
        else:
            self.set = mock_sync(parameter.set_async)

        if hasattr(parameter, 'set_async'):
            self.set_async = parameter.set_async
        else:
            self.set_async = mock_async(parameter.set)

    def _validate(self, values):
        if hasattr(self._parameter, 'validate'):
            for value in values:
                self._parameter.validate(value)

    def append(self, value):
        self._validate((value,))
        self._values.append(value)

    def extend(self, values):
        if hasattr(values, '_parameter') and hasattr(values, '_values'):
            if values._parameter is not self._parameter:
                raise TypeError(
                    'can only extend SweepValues of the same parameters')
            # these values are already validated
            self._values.extend(values._values)
        elif is_sequence(values):
            self._validate(values)
            self._values.extend(values)
        else:
            raise TypeError('cannot extend SweepValues with {}'.format(values))

    def copy(self):
        new_sv = SweepValues(self._parameter, [])
        # skip validation by adding values separately instead of on init
        new_sv._values = self._values[:]
        return new_sv

    def reverse(self):
        self._values.reverse()

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, key):
        return self._values[key]

    def __len__(self):
        return len(self._values)

    def __add__(self, other):
        new_sv = self.copy()
        new_sv.extend(other)
        return new_sv

    def __iadd__(self, values):
        self.extend(values)
        return self

    def __contains__(self, value):
        return value in self._values

    def __reversed__(self):
        new_sv = self.copy()
        new_sv.reverse()
        return new_sv
