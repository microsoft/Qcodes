"""
Measured and/or controlled parameters

The Parameter class is meant for direct parameters of instruments (ie
subclasses of Instrument) but elsewhere in Qcodes we can use anything
as a parameter if it has the right attributes:

To use Parameters in data acquisition loops, they should have:
    .name - like a variable name, ie no spaces or weird characters
    .label - string to use as an axis label (optional, defaults to .name)
    (except for composite measurements, see below)

Controlled parameters should have a .set(value) method, which takes a single
value to apply to this parameter. To use this parameter for sweeping, also
connect its __getitem__ to SweepFixedValues as below.

Measured parameters should have .get() which can return:

- a single value:
    parameter should have .name and optional .label as above

- several values of different meaning (raw and measured, I and Q,
  a set of fit parameters, that sort of thing, that all get measured/calculated
  at once):
    parameter should have .names and optional .labels, each a sequence with
    the same length as returned by .get()

- an array of values of one type:
    parameter should have .name and optional .label as above, but also
    .shape attribute, which is an integer (or tuple of integers) describing
    the shape of the returned array (which must be fixed)
    optionally also .setpoints, array(s) of setpoint values for this data
    otherwise we will use integers from 0 in each direction as the setpoints

- several arrays of values (all the same shape):
    define .names (and .labels) AND .shape (and .setpoints)
"""

from datetime import datetime, timedelta
from copy import copy
import time
import logging
import os
import collections

import numpy

from qcodes.utils.deferred_operations import DeferredOperations
from qcodes.utils.helpers import (permissive_range, wait_secs, is_sequence_of,
                                  DelegateAttributes, full_class, named_repr)
from qcodes.utils.metadata import Metadatable
from qcodes.utils.command import Command, NoCommandError
from qcodes.utils.validators import Validator, Numbers, Ints, Enum, Anything
from qcodes.instrument.sweep_values import SweepFixedValues
from qcodes.data.data_array import DataArray


def no_setter(*args, **kwargs):
    raise NotImplementedError('This Parameter has no setter defined.')


def no_getter(*args, **kwargs):
    raise NotImplementedError(
        'This Parameter has no getter, use .get_latest to get the most recent '
        'set value.')


class Parameter(Metadatable, DeferredOperations):
    """
    Define one generic parameter, not necessarily part of
    an instrument. can be settable and/or gettable.

    A settable Parameter has a .set method, and supports only a single value
    at a time (see below)

    A gettable Parameter has a .get method, which may return:

    1.  a single value
    2.  a sequence of values with different names (for example,
        raw and interpreted, I and Q, several fit parameters...)
    3.  an array of values all with the same name, but at different
        setpoints (for example, a time trace or fourier transform that
        was acquired in the hardware and all sent to the computer at once)
    4.  2 & 3 together: a sequence of arrays. All arrays should be the same
        shape.
    5.  a sequence of differently shaped items

    Because .set only supports a single value, if a Parameter is both
    gettable AND settable, .get should return a single value too (case 1)

    Parameters have a .get_latest method that simply returns the most recent
    set or measured value. This can either be called ( param.get_latest() )
    or used in a Loop as if it were a (gettable-only) parameter itself:
        Loop(...).each(param.get_latest)


    The constructor arguments change somewhat between these cases:

    Todo:
        no idea how to document such a constructor

    Args:
        name: (1&3) the local name of this parameter, should be a valid
            identifier, ie no spaces or special characters

        names: (2,4,5) a tuple of names

        label: (1&3) string to use as an axis label for this parameter
            defaults to name

        labels: (2,4,5) a tuple of labels

        units: (1&3) string that indicates units of parameter for use in axis
            label and snapshot

        shape: (3&4) a tuple of integers for the shape of array returned by
            .get().

        shapes: (5) a tuple of tuples, each one as in `shape`.
            Single values should be denoted by None or ()

        setpoints: (3,4,5) the setpoints for the returned array of values.
            3&4 - a tuple of arrays. The first array is be 1D, the second 2D,
                etc.
            5 - a tuple of tuples of arrays
            Defaults to integers from zero in each respective direction
            Each may be either a DataArray, a numpy array, or a sequence
            (sequences will be converted to numpy arrays)
            NOTE: if the setpoints will be different each measurement, leave
            this out and return the setpoints (with extra names) in the get.

        setpoint_names: (3,4,5) one identifier (like `name`) per setpoint
            array. Ignored if `setpoints` are DataArrays, which already have
            names.

        setpoint_labels: (3&4) one label (like `label`) per setpoint array.
            Overridden if `setpoints` are DataArrays and already have labels.

        vals: allowed values for setting this parameter (only relevant
            if it has a setter),  defaults to Numbers()

        docstring (Optional[string]): documentation string for the __doc__
            field of the object. The __doc__ field of the instance is used by
            some help systems, but not all

        snapshot_get (bool): Prevent any update to the parameter
          for example if it takes too long to update

    """
    def __init__(self,
                 name=None, names=None,
                 label=None, labels=None,
                 units=None,
                 shape=None, shapes=None,
                 setpoints=None, setpoint_names=None, setpoint_labels=None,
                 vals=None, docstring=None, snapshot_get=True, **kwargs):
        super().__init__(**kwargs)
        self._snapshot_get = snapshot_get

        self.has_get = hasattr(self, 'get')
        self.has_set = hasattr(self, 'set')
        self._meta_attrs = ['setpoint_names', 'setpoint_labels']

        # always let the parameter have a single name (in fact, require this!)
        # even if it has names too
        self.name = str(name)

        if names is not None:
            # check for names first - that way you can provide both name
            # AND names for instrument parameters - name is how you get the
            # object (from the parameters dict or the delegated attributes),
            # and names are the items it returns
            self.names = names
            self.labels = names if labels is None else names
            self.units = units if units is not None else [''] * len(names)

            self.set_validator(vals or Anything())
            self.__doc__ = os.linesep.join((
                'Parameter class:',
                '* `names` %s' % ', '.join(self.names),
                '* `labels` %s' % ', '.join(self.labels),
                '* `units` %s' % ', '.join(self.units)))
            self._meta_attrs.extend(['names', 'labels', 'units'])

        elif name is not None:
            self.label = name if label is None else label
            self.units = units if units is not None else ''

            # vals / validate only applies to simple single-value parameters
            self.set_validator(vals)

            # generate default docstring
            self.__doc__ = os.linesep.join((
                'Parameter class:',
                '* `name` %s' % self.name,
                '* `label` %s' % self.label,
                # TODO is this unit s a typo? shouldnt that be unit?
                '* `units` %s' % self.units,
                '* `vals` %s' % repr(self._vals)))
            self._meta_attrs.extend(['name', 'label', 'units', 'vals'])

        else:
            raise ValueError('either name or names is required')

        if shape is not None or shapes is not None:
            nt = type(None)

            if shape is not None:
                if not is_sequence_of(shape, int):
                    raise ValueError('shape must be a tuple of ints, not ' +
                                     repr(shape))
                self.shape = shape
                depth = 1
                container_str = 'tuple'
            else:
                if not is_sequence_of(shapes, int, depth=2):
                    raise ValueError('shapes must be a tuple of tuples '
                                     'of ints, not ' + repr(shape))
                self.shapes = shapes
                depth = 2
                container_str = 'tuple of tuples'

            sp_types = (nt, DataArray, collections.Sequence,
                        collections.Iterator)
            if (setpoints is not None and
                    not is_sequence_of(setpoints, sp_types, depth)):
                raise ValueError(
                    'setpoints must be a {} of arrays'.format(container_str))
            if (setpoint_names is not None and
                    not is_sequence_of(setpoint_names, (nt, str), depth)):
                raise ValueError('setpoint_names must be a {} '
                                 'of strings'.format(container_str))
            if (setpoint_labels is not None and
                    not is_sequence_of(setpoint_labels, (nt, str), depth)):
                raise ValueError('setpoint_labels must be a {} '
                                 'of strings'.format(container_str))

            self.setpoints = setpoints
            self.setpoint_names = setpoint_names
            self.setpoint_labels = setpoint_labels

        # record of latest value and when it was set or measured
        # what exactly this means is different for different subclasses
        # but they all use the same attributes so snapshot is consistent.
        self._latest_value = None
        self._latest_ts = None

        if docstring is not None:
            self.__doc__ = docstring + os.linesep + self.__doc__

        self.get_latest = GetLatest(self)

    def __repr__(self):
        return named_repr(self)

    def __call__(self, *args):
        if len(args) == 0:
            if self.has_get:
                return self.get()
            else:
                raise NoCommandError('no get cmd found in' +
                                     ' Parameter {}'.format(self.name))
        else:
            if self.has_set:
                self.set(*args)
            else:
                raise NoCommandError('no set cmd found in' +
                                     ' Parameter {}'.format(self.name))

    def _latest(self):
        return {
            'value': self._latest_value,
            'ts': self._latest_ts
        }

    # get_attrs ignores leading underscores, unless they're in this list
    _keep_attrs = ['__doc__', '_vals']

    def get_attrs(self):
        """
        Attributes recreated as properties in the RemoteParameter proxy.

        Grab the names of all attributes that the RemoteParameter needs
        to function like the main one (in loops etc)

        Returns:
            list: All public attribute names, plus docstring and _vals
        """
        out = []

        for attr in dir(self):
            if ((attr[0] == '_' and attr not in self._keep_attrs) or
                    callable(getattr(self, attr))):
                continue
            out.append(attr)

        return out

    def snapshot_base(self, update=False):
        """
        State of the parameter as a JSON-compatible dict.

        Args:
            update (bool): If True, update the state by calling
                    parameter.get().
                    If False, just use the latest values in memory.

        Returns:
            dict: base snapshot
        """

        if self.has_get and self._snapshot_get and update:
            self.get()

        state = self._latest()
        state['__class__'] = full_class(self)

        if isinstance(state['ts'], datetime):
            state['ts'] = state['ts'].strftime('%Y-%m-%d %H:%M:%S')

        for attr in set(self._meta_attrs):
            if attr == 'instrument' and getattr(self, '_instrument', None):
                state.update({
                    'instrument': full_class(self._instrument),
                    'instrument_name': self._instrument.name
                })

            elif hasattr(self, attr):
                state[attr] = getattr(self, attr)

        return state

    def _save_val(self, value):
        self._latest_value = value
        self._latest_ts = datetime.now()

    def set_validator(self, vals):
        """
        Set a validator `vals` for this parameter.

        Args:
            vals (Validator):  validator to set
        """
        if vals is None:
            self._vals = Numbers()
        elif isinstance(vals, Validator):
            self._vals = vals
        else:
            raise TypeError('vals must be a Validator')

    def validate(self, value):
        """
        Validate value

        Args:
            value (any): value to validate

        """
        if hasattr(self, '_instrument'):
            context = (getattr(self._instrument, 'name', '') or
                       str(self._instrument.__class__)) + '.' + self.name
        else:
            context = self.name

        self._vals.validate(value, 'Parameter: ' + context)

    def sweep(self, start, stop, step=None, num=None):
        """
        Create a collection of parameter values to be iterated over.
        Requires `start` and `stop` and (`step` or `num`)
        The sign of `step` is not relevant.

        Args:
            start (Union[int, float]): The starting value of the sequence.
            stop (Union[int, float]): The end value of the sequence.
            step (Optional[Union[int, float]]):  Spacing between values.
            num (Optional[int]): Number of values to generate.

        Returns:
            SweepFixedValues: collection of parameter values to be
                iterated over

        Examples:
            >>> sweep(0, 10, num=5)
             [0.0, 2.5, 5.0, 7.5, 10.0]
            >>> sweep(5, 10, step=1)
            [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            >>> sweep(15, 10.5, step=1.5)
            >[15.0, 13.5, 12.0, 10.5]
        """
        return SweepFixedValues(self, start=start, stop=stop,
                                step=step, num=num)

    def __getitem__(self, keys):
        """
        Slice a Parameter to get a SweepValues object
        to iterate over during a sweep
        """
        return SweepFixedValues(self, keys)

    @property
    def full_name(self):
        """Include the instrument name with the Parameter name if possible."""
        if getattr(self, 'name', None) is None:
            return None

        try:
            inst_name = self._instrument.name
            if inst_name:
                return inst_name + '_' + self.name
        except AttributeError:
            pass

        return self.name

    @property
    def full_names(self):
        """Include the instrument name with the Parameter names if possible."""
        if getattr(self, 'names', None) is None:
            return None

        try:
            inst_name = self._instrument.name
            if inst_name:
                return [inst_name + '_' + name for name in self.names]
        except AttributeError:
            pass

        return self.names


class StandardParameter(Parameter):
    """
    Define one measurement parameter.

    Args:
        name (string): the local name of this parameter
        instrument (Optional[Instrument]): an instrument that handles this
            function. Default None.

        get_cmd (Optional[Union[string, function]]): a string or function to
            get this parameter. You can only use a string if an instrument is
            provided, then this string will be passed to instrument.ask

        get_parser ( Optional[function]): function to transform the response
            from get to the final output value.
            See also val_mapping

        set_cmd (Optional[Union[string, function]]): command to set this
            parameter, either:
            - a string (containing one field to .format, like "{}" etc)
              you can only use a string if an instrument is provided,
              this string will be passed to instrument.write
            - a function (of one parameter)

        set_parser (Optional[function]): function to transform the input set
            value to an encoded value sent to the instrument.
            See also val_mapping

        val_mapping (Optional[dict]): a bidirectional map data/readable values
            to instrument codes, expressed as a dict:
            ``{data_val: instrument_code}``
            For example, if the instrument uses '0' to mean 1V and '1' to mean
            10V, set val_mapping={1: '0', 10: '1'} and on the user side you
            only see 1 and 10, never the coded '0' and '1'

            If vals is omitted, will also construct a matching Enum validator.
            NOTE: only applies to get if get_cmd is a string, and to set if
            set_cmd is a string.

            You can use ``val_mapping`` with ``get_parser``, in which case
            ``get_parser`` acts on the return value from the instrument first,
            then ``val_mapping`` is applied (in reverse).

            You CANNOT use ``val_mapping`` and ``set_parser`` together - that
            would just provide too many ways to do the same thing.

        vals (Optional[Validator]): a Validator object for this parameter

        delay (Optional[Union[int, float]]): time (in seconds) to wait after
            the *start* of each set, whether part of a sweep or not. Can be
            set to 0 to go maximum speed with no errors.

        max_delay (Optional[Union[int, float]]): If > delay, we don't emit a
            warning unless the time taken during a single set is greater than
            this, even though we aim for delay.

        step (Optional[Union[int, float]]): max increment of parameter value.
            Larger changes are broken into multiple steps this size.

        max_val_age (Optional[Union[int, float]]): max time (in seconds) to
            trust a saved value from this parameter as the starting point of
            a sweep.

        **kwargs: Passed to Parameter parent class

    Raises:
        NoCommandError: if get and set are not found
    """
    def __init__(self, name, instrument=None,
                 get_cmd=None, get_parser=None,
                 set_cmd=None, set_parser=None,
                 delay=None, max_delay=None, step=None, max_val_age=3600,
                 vals=None, val_mapping=None, **kwargs):
        # handle val_mapping before super init because it impacts
        # vals / validation in the base class
        if val_mapping:
            if vals is None:
                vals = Enum(*val_mapping.keys())

            self._get_mapping = {v: k for k, v in val_mapping.items()}

            if get_parser is None:
                get_parser = self._valmapping_get_parser
            else:
                # First run get_parser, then run the result through
                # val_mapping
                self._get_preparser = get_parser
                get_parser = self._valmapping_with_preparser

            if set_parser is None:
                self._set_mapping = val_mapping
                set_parser = self._set_mapping.__getitem__
            else:
                raise TypeError(
                    'You cannot use set_parser and val_mapping together.')

        if get_parser is not None and not isinstance(get_cmd, str):
            logging.warning('get_parser is set, but will not be used ' +
                            '(name %s)' % name)
        super().__init__(name=name, vals=vals, **kwargs)

        self._instrument = instrument

        self._meta_attrs.extend(['instrument', 'sweep_step', 'sweep_delay',
                                'max_sweep_delay'])

        # stored value from last .set() or .get()
        # normally only used by set with a sweep, to avoid
        # having to call .get() for every .set()
        self._max_val_age = 0

        self._set_get(get_cmd, get_parser)
        self._set_set(set_cmd, set_parser)
        self.set_delay(delay, max_delay)
        self.set_step(step, max_val_age)

        if not (self.has_get or self.has_set):
            raise NoCommandError('neither set nor get cmd found in' +
                                 ' Parameter {}'.format(self.name))

    def get(self):
        try:
            value = self._get()
            self._save_val(value)
            return value
        except Exception as e:
            e.args = e.args + (
                'getting {}:{}'.format(self._instrument.name, self.name),)
            raise e

    def _valmapping_get_parser(self, val):
        """
        Get parser to be used in the case that a val_mapping is defined
        and a separate get_parser is not defined.

        Tries to match against defined strings in the mapping dictionary. If
        there are no matches, we try to convert the val into an integer.
        """

        # Try and match the raw value from the instrument directly
        try:
            return self._get_mapping[val]
        except KeyError:
            pass

        # If there is no match, we can try to convert the parameter into a
        # numeric value
        try:
            val = int(val)
            return self._get_mapping[val]
        except (ValueError, KeyError):
            raise KeyError("Unmapped value from instrument: {!r}".format(val))

    def _valmapping_with_preparser(self, val):
        return self._valmapping_get_parser(self._get_preparser(val))

    def _set_get(self, get_cmd, get_parser):
        exec_str = self._instrument.ask if self._instrument else None
        self._get = Command(arg_count=0, cmd=get_cmd, exec_str=exec_str,
                            output_parser=get_parser,
                            no_cmd_function=no_getter)

        self.has_get = (get_cmd is not None)

    def _set_set(self, set_cmd, set_parser):
        # note: this does not set the final setter functions. that's handled
        # in self.set_sweep, when we choose a swept or non-swept setter.
        exec_str = self._instrument.write if self._instrument else None
        self._set = Command(arg_count=1, cmd=set_cmd, exec_str=exec_str,
                            input_parser=set_parser, no_cmd_function=no_setter)

        self.has_set = set_cmd is not None

    def _validate_and_set(self, value):
        try:
            clock = time.perf_counter()
            self.validate(value)
            self._set(value)
            self._save_val(value)
            if self._delay is not None:
                clock, remainder = self._update_set_ts(clock)
                time.sleep(remainder)
        except Exception as e:
            e.args = e.args + (
                'setting {}:{} to {}'.format(self._instrument.name,
                                             self.name, repr(value)),)
            raise e

    def _sweep_steps(self, value):
        oldest_ok_val = datetime.now() - timedelta(seconds=self._max_val_age)
        state = self._latest()
        if state['ts'] is None or state['ts'] < oldest_ok_val:
            start_value = self.get()
        else:
            start_value = state['value']

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
        return permissive_range(start_value, value, self._step)[1:]

    def _update_set_ts(self, step_clock):
        # calculate the delay time to the *max* delay,
        # then take off up to the tolerance
        tolerance = self._delay_tolerance
        step_clock += self._delay
        remainder = wait_secs(step_clock + tolerance)
        if remainder <= tolerance:
            # don't allow extra delays to compound
            step_clock = time.perf_counter()
            remainder = 0
        else:
            remainder -= tolerance
        return step_clock, remainder

    def _validate_and_sweep(self, value):
        try:
            self.validate(value)
            step_clock = time.perf_counter()

            for step_val in self._sweep_steps(value):
                self._set(step_val)
                self._save_val(step_val)
                if self._delay is not None:
                    step_clock, remainder = self._update_set_ts(step_clock)
                    time.sleep(remainder)

            self._set(value)
            self._save_val(value)

            if self._delay is not None:
                step_clock, remainder = self._update_set_ts(step_clock)
                time.sleep(remainder)
        except Exception as e:
            e.args = e.args + (
                'setting {}:{} to {}'.format(self._instrument.name,
                                             self.name, repr(value)),)
            raise e

    def set_step(self, step, max_val_age=None):
        """
        Configure whether this Parameter uses steps during set operations.
        If step is a positive number, this is the maximum value change
        allowed in one hardware call, so a single set can result in many
        calls to the hardware if the starting value is far from the target.

        Args:
            step (Union[int, float]): A positive number, the largest change
                allowed in one call. All but the final change will attempt to
                change by +/- step exactly

            max_val_age (Optional[int]): Only used with stepping, the max time
                (in seconds) to trust a saved value. If this parameter has not
                been set or measured more recently than this, it will be
                measured before starting to step, so we're confident in the
                value we're starting from.

        Raises:
            TypeError: if step is not numeric
            ValueError: if step is negative
            TypeError:  if step is not integer for an integer parameter
            TypeError: if step is not a number
            TypeError: if max_val_age is not numeric
            ValueError: if max_val_age is negative
        """
        if not step:
            # single-command setting
            self.set = self._validate_and_set

        elif not self._vals.is_numeric:
            raise TypeError('you can only step numeric parameters')
        elif step <= 0:
            raise ValueError('step must be positive')
        elif (isinstance(self._vals, Ints) and
                not isinstance(step, int)):
            raise TypeError(
                'step must be a positive int for an Ints parameter')
        elif not isinstance(step, (int, float)):
            raise TypeError('step must be a number')

        else:
            # stepped setting
            if max_val_age is not None:
                if not isinstance(max_val_age, (int, float)):
                    raise TypeError(
                        'max_val_age must be a number')
                if max_val_age < 0:
                    raise ValueError('max_val_age must be non-negative')
                self._max_val_age = max_val_age

            self._step = step
            self.set = self._validate_and_sweep

    def get_delay(self):
        """Return the delay time of this parameter. Also see `set_delay` """
        return self._delay

    def set_delay(self, delay, max_delay=None):
        """
        Configure this parameter with a delay between set operations.

        Typically used in conjunction with set_step to create an effective
        ramp rate, but can also be used without a step to enforce a delay
        after every set.
        If delay and max_delay are both None or 0, we never emit warnings
        no matter how long the set takes.

        Args:
            delay(Union[int, float]): the target time between set calls. The
                actual time will not be shorter than this, but may be longer
                if the underlying set call takes longer.

            max_delay(Optional[Union[int, float]]): if given, the longest time
                allowed for the underlying set call before we emit a warning.

        Raises:
            TypeError: If delay is not int nor float
            TypeError: If max_delay is not int nor float
            ValueError: If delay is negative
            ValueError: If max_delay is smaller than delay
        """
        if delay is None:
            delay = 0
        if not isinstance(delay, (int, float)):
            raise TypeError('delay must be a number')
        if delay < 0:
            raise ValueError('delay must not be negative')
        self._delay = delay

        if max_delay is not None:
            if not isinstance(max_delay, (int, float)):
                raise TypeError(
                    'max_delay must be a either  int or a float')
            if max_delay < delay:
                raise ValueError('max_delay must be no shorter than delay')
            self._delay_tolerance = max_delay - delay
        else:
            self._delay_tolerance = 0

        if not (self._delay or self._delay_tolerance):
            # denotes that we shouldn't follow the wait code or
            # emit any warnings
            self._delay = None


class ManualParameter(Parameter):
    """
    Define one parameter that reflects a manual setting / configuration.

    Args:
        name (string): the local name of this parameter

        instrument (Optional[Instrument]): the instrument this applies to,
            if any.

        initial_value (Optional[string]): starting value, the
            only invalid value allowed, and None is only allowed as an initial
            value, it cannot be set later

        **kwargs: Passed to Parameter parent class
    """
    def __init__(self, name, instrument=None, initial_value=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._instrument = instrument
        self._meta_attrs.extend(['instrument', 'initial_value'])

        if initial_value is not None:
            self.validate(initial_value)
            self._save_val(initial_value)

    def set(self, value):
        """
        Validate and saves value
        Args:
            value (any): value to validate and save
        """
        self.validate(value)
        self._save_val(value)

    def get(self):
        """ Return latest value"""
        return self._latest()['value']


class GetLatest(DelegateAttributes, DeferredOperations):
    """
    Wrapper for a Parameter that just returns the last set or measured value
    stored in the Parameter itself.

    Examples:
        >>> # Can be called:
        >>> param.get_latest()
        >>> # Or used as if it were a gettable-only parameter itself:
        >>> Loop(...).each(param.get_latest)

    Args:
        parameter (Parameter): Parameter to be wrapped
    """
    def __init__(self, parameter):
        self.parameter = parameter

    delegate_attr_objects = ['parameter']
    omit_delegate_attrs = ['set']

    def get(self):
        """ Return latest value"""
        return self.parameter._latest()['value']

    def __call__(self):
        return self.get()


def combine(*parameters, name, label=None, units=None, aggregator=None):
    """Combine parameters into one swepable parameter

    Args:
        *paramters (qcodes.Parameter): the parameters to combine
        name (str): the name of the paramter
        label (Optional[str]): the label of the combined parameter
        unit (Optional[str]): the unit of the combined parameter
        aggregator (Optional[Callable[list[any]]]): a function to aggregate
            the set values into one

    A combined parameter sets all the combined parameters at every point of the
    sweep.
    The sets are called in the same order the parameters are, and
    sequantially.
    """
    parameters = list(parameters)
    multi_par = CombinedParameter(parameters, name, label, units, aggregator)
    return multi_par


class CombinedParameter(Metadatable):
    """ A combined parameter

    Args:
        *paramters (qcodes.Parameter): the parameters to combine
        name (str): the name of the paramter
        label (Optional[str]): the label of the combined parameter
        unit (Optional[str]): the unit of the combined parameter
        aggregator (Optional[Callable[list[any]]]): a function to aggregate
            the set values into one

    A combined parameter sets all the combined parameters at every point of the
    sweep.
    The sets are called in the same order the parameters are, and
    sequantially.
    """

    def __init__(self, parameters, name, label=None,
                 units=None, aggregator=None):
        super().__init__()
        # TODO(giulioungaretti)temporary hack
        # starthack
        # this is a dummy parameter
        # that mimicks the api that a normal parameter has
        self.parameter = lambda: None
        self.parameter.full_name = name
        self.parameter.name = name
        self.parameter.label = label
        self.parameter.units = units
        # endhack
        self.parameters = parameters
        self.sets = [parameter.set for parameter in self.parameters]
        self.dimensionality = len(self.sets)

        if aggregator:
            self.f = aggregator
            setattr(self, 'aggregate', self._aggregate)

    def set(self, index: int):
        """
        Set multiple parameters.

        Args:
            index (int): the index of the setpoints one wants to set
        Returns:
            list: values that where actually set
        """
        values = self.setpoints[index]
        for setFunction, value in zip(self.sets, values):
            setFunction(value)
        return values

    def sweep(self, *array: numpy.ndarray):
        """
        Creates a new combined parameter to be iterated over.
        One can sweep over either:
             - n array of lenght m
             - one nxm array

        where n is the number of combined parameters
        and m is the number of setpoints

        Args:
            *array(numpy.ndarray): array(s) of setopoints

        Returns:
            MultiPar: combined parameter
        """
        # if it's a list of arrays, convert to one array
        if len(array) > 1:
            dim = set([len(a) for a in array])
            if len(dim) != 1:
                raise ValueError("Arrays have different number of setpoints")
            array = numpy.array(array).transpose()
        else:
            # cast to array in case users
            # decide to not read docstring
            # and pass a 2d list
            array = numpy.array(array[0])
        new = copy(self)
        _error_msg = """ Dimensionality of array does not match\
                        the number of parameter combined. Expected a \
                        {} dimensional array, got a {} dimensional array. \
                        """
        try:
            if array.shape[1] != self.dimensionality:
                raise ValueError(_error_msg.format(self.dimensionality,
                                                   array.shape[1]))
        except KeyError:
            # this means the array is 1d
            raise ValueError(_error_msg.format(self.dimensionality, 1))

        new.setpoints = array.tolist()
        return new

    def _aggregate(self, *vals):
        # check f args
        return self.f(*vals)

    def __iter__(self):
        return iter(range(len(self.setpoints)))

    def __len__(self):
        # dimension of the sweep_values
        # i.e. how many setpoint
        return numpy.shape(self.setpoints)[0]

    def snapshot_base(self, update=False):
        """
        State of the combined parameter as a JSON-compatible dict.

        Args:
            update (bool):

        Returns:
            dict: base snapshot
        """
        meta_data = collections.OrderedDict()
        meta_data['__class__'] = full_class(self)
        meta_data["units"] = self.parameter.units
        meta_data["label"] = self.parameter.label
        meta_data["full_name"] = self.parameter.full_name
        meta_data["aggreagator"] = repr(getattr(self, 'f', None))
        for param in self.parameters:
            meta_data[param.full_name] = param.snapshot()

        return meta_data
