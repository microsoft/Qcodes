from datetime import datetime, timedelta
import time
import asyncio
import logging

from qcodes.utils.helpers import permissive_range, wait_secs
from qcodes.utils.metadata import Metadatable
from qcodes.utils.sync_async import syncable_command, NoCommandError
from qcodes.utils.validators import Validator, Numbers, Ints
from qcodes.sweep_values import SweepFixedValues


def no_func(*args, **kwargs):
    raise NotImplementedError('no function defined')


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
        self._get, self._get_async = syncable_command(
            0, get_cmd, async_get_cmd, self._instrument.ask,
            self._instrument.ask_async, parse_function, no_func)

        if self._get is not no_func:
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
        self._set, self._set_async = syncable_command(
            1, set_cmd, async_set_cmd, self._instrument.write,
            self._instrument.write_async, no_cmd_function=no_func)

        if self._set is not no_func:
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
        return SweepFixedValues(self, keys)
