'''
Measured and/or controlled parameters

the Parameter class is meant for direct parameters of instruments (ie
subclasses of Instrument) but elsewhere in Qcodes we can use anything
as a parameter if it has the right attributes:

To use Parameters in data acquisition loops, they should have:
    .name - like a variable name, ie no spaces or weird characters
    .label - string to use as an axis label (optional, defaults to .name)
    (except for composite measurements, see below)

Controlled parameters should have a .set(value) (and/or .set_async(value))
method, which takes a single value to apply to this parameter.
To use this parameter for sweeping, also connect its __getitem__ to
SweepFixedValues as below.

Measured parameters should have .get() (and/or .get_async()) which can return:
- a single value:
    parameter should have .name and optional .label as above
- several values of different meaning (raw and measured, I and Q,
  a set of fit parameters, that sort of thing, that all get measured/calculated
  at once):
    parameter should have .names and optional .labels, each a sequence with
    the same length as returned by .get()
- an array of values of one type:
    parameter should have .name and optional .label as above, but also
    .size attribute, which is an integer (or tuple of integers) describing
    the size of the returned array (which must be fixed)
    optionally also .setpoints, array(s) of setpoint values for this data
    otherwise we will use integers from 0 in each direction as the setpoints
- several arrays of values (all the same size):
    define .names (and .labels) AND .size (and .setpoints)
'''

from datetime import datetime, timedelta
import time
import asyncio
import logging

from qcodes.utils.helpers import (permissive_range, wait_secs,
                                  DelegateAttributes)
from qcodes.utils.metadata import Metadatable
from qcodes.utils.sync_async import syncable_command, NoCommandError
from qcodes.utils.validators import Validator, Numbers, Ints, Enum
from qcodes.instrument.sweep_values import SweepFixedValues


def no_setter(*args, **kwargs):
    raise NotImplementedError('This Parameter has no setter defined.')


def no_getter(*args, **kwargs):
    raise NotImplementedError(
        'This Parameter has no getter, use .get_latest to get the most recent '
        'set value.')


class Parameter(Metadatable):
    '''
    defines one generic parameter, not necessarily part of
    an instrument. can be settable and/or gettable.

    A settable Parameter has a .set and/or a .set_async method,
    and supports only a single value at a time (see below)

    A gettable Parameter has a .get and/or a .get_async method,
    which may return:
    1.  a single value
    2.  a sequence of values with different names (for example,
        raw and interpreted, I and Q, several fit parameters...)
    3.  an array of values all with the same name, but at different
        setpoints (for example, a time trace or fourier transform that
        was acquired in the hardware and all sent to the computer at once)
    4.  2 & 3 together: a sequence of arrays. All arrays should be the same
        size.
    5.  a sequence of differently sized items

    Because .set only supports a single value, if a Parameter is both
    gettable AND settable, .get should return a single value too (case 1)

    Parameters have a .get_latest method that simply returns the most recent
    set or measured value. This can either be called ( param.get_latest() )
    or used in a Loop as if it were a (gettable-only) parameter itself:
        Loop(...).each(param.get_latest)


    The constructor arguments change somewhat between these cases:

    name: (1&3) the local name of this parameter, should be a valid
        identifier, ie no spaces or special characters
    names: (2,4,5) a tuple of names

    label: (1&3) string to use as an axis label for this parameter
        defaults to name
    labels: (2,4,5) a tuple of labels

    units: (1&3) string that indicates units of parameter for use in axis
        label and snapshot
           (2,4,5) a tuple of units

    size: (3&4) an integer or tuple of integers for the size of array
        returned by .get(). Can be an integer only if the array is 1D, but
        as a tuple it can describe any dimensionality (including 1D)
        If size is an integer then setpoints, setpoint_names,
        and setpoint_labels should also not be wrapped in tuples.
    sizes: (5) a tuple of integers or tuples, each one as in `size`.

    setpoints: (3,4,5) the setpoints for the returned array of values.
        3&4 - This should be an array if `size` is an integer, or a
            tuple of arrays if `size` is a tuple
            The first array should be 1D, the second 2D, etc.
        5 - This should be a tuple of arrays or tuples, each item as above
            Single values should be denoted by None or (), not 1 (because 1
            would be a length-1 array)
        Defaults to integers from zero in each respective direction
        Each may be either a DataArray, a numpy array, or a sequence
        (sequences will be converted to numpy arrays)
        NOTE: if the setpoints will be different each measurement, leave
        this out and return the setpoints (with extra names) in the get.
    setpoint_names: (3,4,5) one identifier (like `name`) per setpoint
        array.
        Ignored if `setpoints` are DataArrays, which already have names.
    setpoint_labels: (3&4) one label (like `label`) per setpoint array.
        Overridden if `setpoints` are DataArrays and already have labels.

    vals: allowed values for setting this parameter (only relevant
        if it has a setter)
        defaults to Numbers()
    '''
    def __init__(self,
                 name=None, names=None,
                 label=None, labels=None,
                 units=None,
                 size=None, sizes=None,
                 setpoints=None, setpoint_names=None, setpoint_labels=None,
                 vals=None, **kwargs):
        super().__init__(**kwargs)

        if names is not None:
            # check for names first - that way you can provide both name
            # AND names for instrument parameters - name is how you get the
            # object (from the parameters dict or the delegated attributes),
            # and names are the items it returns
            self.names = names
            self.labels = names if labels is None else names
            self.units = units if units is not None else ['']*len(names)

        elif name is not None:
            self.name = name
            self.label = name if label is None else label
            self.units = units if units is not None else ''

            # vals / validate only applies to simple single-value parameters
            self._set_vals(vals)

        else:
            raise ValueError('either name or names is required')

        if size is not None or sizes is not None:
            if size is not None:
                self.size = size
            else:
                self.sizes = sizes

            self.setpoints = setpoints
            self.setpoint_names = setpoint_names
            self.setpoint_labels = setpoint_labels

        # record of latest value and when it was set or measured
        # what exactly this means is different for different subclasses
        # but they all use the same attributes so snapshot is consistent.
        self._last_value = None
        self._last_ts = None
        self.get_latest = GetLatest(self)

    def snapshot_base(self):
        '''
        json state of the Parameter
        '''
        if self._last_ts is None:
            ts = None
        else:
            ts = self._last_ts.strftime('%Y-%m-%d %H:%M:%S')

        return {
            'value': self._last_value,
            'ts': ts
        }

    def _save_val(self, value):
        self._last_value = value
        self._last_ts = datetime.now()

    def _set_vals(self, vals):
        if vals is None:
            self._vals = Numbers()
        elif isinstance(vals, Validator):
            self._vals = vals
        else:
            raise TypeError('vals must be a Validator')

    def validate(self, value):
        '''
        raises a ValueError if this value is not allowed for this Parameter
        '''
        if not self._vals.is_valid(value):
            raise ValueError(
                '{} is not a valid value for {}'.format(value, self.name))

    def __getitem__(self, keys):
        '''
        slice a Parameter to get a SweepValues object
        to iterate over during a sweep
        '''
        return SweepFixedValues(self, keys)


class StandardParameter(Parameter):
    '''
    defines one measurement parameter

    name: the local name of this parameter
    instrument: an instrument that handles this parameter
        default None

    get_cmd: a string or function to get this parameter
        you can only use a string if an instrument is provided,
        this string will be passed to instrument.ask
    async_get_cmd: a function to use for async get, or for both sync
        and async if get_cmd is missing or None
    get_parser: function to transform the response from get
        to the final output value.
        NOTE: only applies if get_cmd is a string. The function forms
        of get_cmd and async_get_cmd should do their own parsing
        See also val_mapping

    set_cmd: command to set this parameter, either:
        - a string (containing one field to .format, like "{}" etc)
          you can only use a string if an instrument is provided,
          this string will be passed to instrument.write
        - a function (of one parameter)
    async_set_cmd: a function to use for async set, or for both sync
        and async if set_cmd is missing or None
    set_parser: function to transform the input set value to an encoded
        value sent to the instrument.
        NOTE: only applies if set_cmd is a string. The function forms
        of set_cmd and async_set_cmd should do their own parsing
        See also val_mapping

    parse_function: DEPRECATED - use get_parser instead

    val_mapping: a bidirectional map from data/readable values to
        instrument codes, expressed as a dict {data_val: instrument_code}
        For example, if the instrument uses '0' to mean 1V and '1' to mean
        10V, set val_mapping={1: '0', 10: '1'} and on the user side you
        only see 1 and 10, never the coded '0' and '1'

        If vals is omitted, will also construct a matching Enum validator.
        NOTE: only applies to get if get_cmd is a string, and to set if
        set_cmd is a string.

    vals: a Validator object for this parameter

    sweep_step: max increment of parameter value - larger changes
        are broken into steps this size
    sweep_delay: time (in seconds) to wait after each sweep step
    max_val_age: max time (in seconds) to trust a saved value from
        this parameter as the starting point of a sweep
    '''
    def __init__(self, name, instrument=None,
                 get_cmd=None, async_get_cmd=None, get_parser=None,
                 parse_function=None, val_mapping=None,
                 set_cmd=None, async_set_cmd=None, set_parser=None,
                 sweep_step=None, sweep_delay=None, max_sweep_delay=None,
                 max_val_age=3600, vals=None, **kwargs):
        # handle val_mapping before super init because it impacts
        # vals / validation in the base class
        if val_mapping:
            if vals is None:
                vals = Enum(*val_mapping.keys())

            if get_parser is None:
                self._get_mapping = {v: k for k, v in val_mapping.items()}
                get_parser = self._get_mapping.__getitem__

            if set_parser is None:
                self._set_mapping = val_mapping
                set_parser = self._set_mapping.__getitem__

        super().__init__(name=name, vals=vals, **kwargs)

        self._instrument = instrument

        # stored value from last .set() or .get()
        # normally only used by set with a sweep, to avoid
        # having to call .get() for every .set()
        self._max_val_age = 0

        self.has_get = False
        self.has_set = False

        # push deprecated parse_function argument to get_parser
        if get_parser is None:
            get_parser = parse_function

        self._set_get(get_cmd, async_get_cmd, get_parser)
        self._set_set(set_cmd, async_set_cmd, set_parser)
        self.set_sweep(sweep_step, sweep_delay,
                       max_sweep_delay=max_sweep_delay,
                       max_val_age=max_val_age)

        if not (self.has_get or self.has_set):
            raise NoCommandError('neither set nor get cmd found in' +
                                 ' Parameter {}'.format(self.name))

    def get(self):
        value = self._get()
        self._save_val(value)
        return value

    @asyncio.coroutine
    def get_async(self):
        value = yield from self._get_async()
        self._save_val(value)
        return value

    def _set_get(self, get_cmd, async_get_cmd, get_parser):
        self._get, self._get_async = syncable_command(
            param_count=0, cmd=get_cmd, acmd=async_get_cmd,
            exec_str=self._instrument.ask if self._instrument else None,
            aexec_str=self._instrument.ask_async if self._instrument else None,
            output_parser=get_parser, no_cmd_function=no_getter)

        if self._get is not no_getter:
            self.has_get = True

    def _set_set(self, set_cmd, async_set_cmd, set_parser):
        # note: this does not set the final setter functions. that's handled
        # in self.set_sweep, when we choose a swept or non-swept setter.
        self._set, self._set_async = syncable_command(
            param_count=1, cmd=set_cmd, acmd=async_set_cmd,
            exec_str=self._instrument.write if self._instrument else None,
            aexec_str=(self._instrument.write_async if self._instrument
                       else None),
            input_parser=set_parser, no_cmd_function=no_setter)

        if self._set is not no_setter:
            self.has_set = True

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

    def _update_sweep_ts(self, step_clock):
        # calculate the delay time to the *max* delay,
        # then take off up to the tolerance
        tolerance = self._sweep_delay_tolerance
        step_clock += self._sweep_delay
        remainder = wait_secs(step_clock + tolerance)
        if remainder <= tolerance:
            # don't allow extra delays to compound
            step_clock = time.perf_counter()
            remainder = 0
        else:
            remainder -= tolerance
        return step_clock, remainder

    def _validate_and_sweep(self, value):
        self.validate(value)
        step_clock = time.perf_counter()

        for step_val in self._sweep_steps(value):
            self._set(step_val)
            self._save_val(step_val)
            step_clock, remainder = self._update_sweep_ts(step_clock)
            time.sleep(remainder)

        self._set(value)
        self._save_val(value)

    @asyncio.coroutine
    def _validate_and_sweep_async(self, value):
        self.validate(value)
        step_clock = time.perf_counter()

        for step_val in self._sweep_steps(value):
            yield from self._set_async(step_val)
            self._save_val(step_val)
            step_clock, remainder = self._update_sweep_ts(step_clock)
            yield from asyncio.sleep(remainder)

        yield from self._set_async(value)
        self._save_val(value)

    def set_sweep(self, sweep_step, sweep_delay, max_sweep_delay=None,
                  max_val_age=None):
        '''
        configure this Parameter to set using a stair-step sweep

        This means a single .set call will generate many instrument writes
        so that the value only changes at most sweep_step in a time sweep_delay

        sweep_step: the biggest change in value allowed at once

        sweep_delay: the target time between steps. The actual time will not be
            shorter than this, but may be longer if the underlying set call
            takes longer than this time.

        max_sweep_delay: if given, the longest time allowed between steps (due
            to a slow set call, presumably) before we emit a warning

        max_val_age: max time (in seconds) to trust a saved value. Important
        since we need to know what value we're starting from in this sweep.
        '''
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

            if max_sweep_delay is not None:
                if not isinstance(max_sweep_delay, (int, float)):
                    raise TypeError(
                        'max_sweep_delay must be a positive number')
                if max_sweep_delay < sweep_delay:
                    raise ValueError(
                        'max_sweep_delay must not be shorter than sweep_delay')
                self._sweep_delay_tolerance = max_sweep_delay - sweep_delay
            else:
                self._sweep_delay_tolerance = 0

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


class ManualParameter(Parameter):
    '''
    defines one parameter that reflects a manual setting / configuration

    name: the local name of this parameter

    instrument: the instrument this applies to. Not actually used for
        anything, just required so this class can be used with
        Instrument.add_parameter(name, parameter_class=ManualParameter)

    initial_value: optional starting value. Default is None, which is the
        only invalid value allowed (and None is only allowed as an initial
        value, it cannot be set later)
    '''
    def __init__(self, name, instrument=None, initial_value=None, **kwargs):
        super().__init__(name=name, **kwargs)
        if initial_value is not None:
            self.validate(initial_value)
            self._save_val(initial_value)

    def set(self, value):
        self.validate(value)
        self._save_val(value)

    @asyncio.coroutine
    def set_async(self, value):
        return self.set(value)

    def get(self):
        return self._last_value

    @asyncio.coroutine
    def get_async(self):
        return self.get()


class GetLatest(DelegateAttributes):
    '''
    wrapper for a Parameter that just returns the last set or measured value
    stored in the Parameter itself.

    Can be called:
        param.get_latest()

    Or used as if it were a gettable-only parameter itself:
        Loop(...).each(param.get_latest)
    '''
    def __init__(self, parameter):
        self.parameter = parameter

    delegate_attr_objects = ['parameter']
    omit_delegate_attrs = ['set', 'set_async']

    def get(self):
        return self.parameter._last_value

    @asyncio.coroutine
    def get_async(self):
        return self.get()

    def __call__(self):
        return self.get()
