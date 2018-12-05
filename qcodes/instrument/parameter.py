"""
Measured and/or controlled parameters

Anything that you want to either measure or control within QCoDeS should
satisfy the Parameter interface. Most of the time that is easiest to do
by either using or subclassing one of the classes defined here, but you can
also use any class with the right attributes.

All parameter classes are subclassed from _BaseParameter (except
CombinedParameter). The _BaseParameter provides functionality that is common
to all parameter types, such as ramping and scaling of values, adding delays
(see documentation for details).

This file defines four classes of parameters:

- ``Parameter`` is the base class for scalar-valued parameters.
    Two primary ways in which it can be used:

    1. As an ``Instrument`` parameter that sends/receives commands. Provides a
       standardized interface to construct strings to pass to the
       instrument's ``write`` and ``ask`` methods
    2. As a variable that stores and returns a value. For instance, for storing
       of values you want to keep track of but cannot set or get electronically.

    Provides ``sweep`` and ``__getitem__`` (slice notation) methods to use a
    settable parameter as the swept variable in a ``Loop``.
    The get/set functionality can be modified.

- ``ArrayParameter`` is a base class for array-valued parameters, ie anything
    for which each ``get`` call returns an array of values that all have the
    same type and meaning. Currently not settable, only gettable. Can be used
    in ``Measure``, or in ``Loop`` - in which case these arrays are nested
    inside the loop's setpoint array. To use, provide a ``get`` method that
    returns an array or regularly-shaped sequence, and describe that array in
    ``super().__init__``.

- ``MultiParameter`` is the base class for multi-valued parameters. Currently
    not settable, only gettable, but can return an arbitrary collection of
    scalar and array values and can be used in ``Measure`` or ``Loop`` to
    feed data to a ``DataSet``. To use, provide a ``get`` method
    that returns a sequence of values, and describe those values in
    ``super().__init__``.

    ``CombinedParameter`` Combines several parameters into a ``MultiParameter``.
    can be easily used via the ``combine`` function.
    Note that it is not yet a subclass of BaseParameter.


"""

# TODO (alexcjohnson) update this with the real duck-typing requirements or
# create an ABC for Parameter and MultiParameter - or just remove this statement
# if everyone is happy to use these classes.

from datetime import datetime, timedelta
from copy import copy
from operator import xor
import time
import logging
import os
import collections
import warnings
import enum
from typing import Optional, Sequence, TYPE_CHECKING, Union, Callable, List, \
    Dict, Any, Sized, Iterable, cast, Type
from functools import partial, wraps
import numpy


from qcodes.utils.helpers import (permissive_range, is_sequence_of,
                                  DelegateAttributes, full_class, named_repr,
                                  warn_units)
from qcodes.utils.metadata import Metadatable
from qcodes.utils.command import Command
from qcodes.utils.validators import Validator, Ints, Strings, Enum
from qcodes.instrument.sweep_values import SweepFixedValues
from qcodes.data.data_array import DataArray

if TYPE_CHECKING:
    from .base import Instrument, InstrumentBase

Number = Union[float, int]


class _SetParamContext:
    """
    This class is returned by the set method of parameters

    Example usage:
    >>> v = dac.voltage()
    >>> with dac.voltage.set_to(-1):
        ...     # Do stuff with the DAC output set to -1 V.
        ...
    >>> assert abs(dac.voltage() - v) <= tolerance
    """
    def __init__(self, parameter):
        self._parameter = parameter
        self._original_value = self._parameter._latest["value"]

    def __enter__(self):
        pass

    def __exit__(self, typ, value, traceback):
        self._parameter.set(self._original_value)


class _BaseParameter(Metadatable):
    """
    Shared behavior for all parameters. Not intended to be used
    directly, normally you should use ``Parameter``, ``ArrayParameter``,
    ``MultiParameter``, or ``CombinedParameter``.
    Note that ``CombinedParameter`` is not yet a subclass of ``_BaseParameter``

    Args:
        name (str): the local name of the parameter. Must be a valid
            identifier, ie no spaces or special characters or starting with a
            number. If this parameter is part of an Instrument or Station,
            this should match how it will be referenced from that parent,
            ie ``instrument.name`` or ``instrument.parameters[name]``

        instrument (Optional[Instrument]): the instrument this parameter
            belongs to, if any

        snapshot_get (Optional[bool]): False prevents any update to the
            parameter during a snapshot, even if the snapshot was called with
            ``update=True``, for example if it takes too long to update.
            Default True.

        snapshot_value (Optional[bool]): False prevents parameter value to be
            stored in the snapshot. Useful if the value is large.

        step (Optional[Union[int, float]]): max increment of parameter value.
            Larger changes are broken into multiple steps this size.
            When combined with delays, this acts as a ramp.

        scale (Optional[float]): Scale to multiply value with before
            performing set. the internally multiplied value is stored in
            `raw_value`. Can account for a voltage divider.

        offset: Compensate for a parameter specific offset. (just as scale)
            get value = raw value - offset.
            set value = argument + offset.
            If offset and scale are used in combination, when getting a value,
            first an offset is added, then the scale is applied.

        inter_delay (Optional[Union[int, float]]): Minimum time (in seconds)
            between successive sets. If the previous set was less than this,
            it will wait until the condition is met.
            Can be set to 0 to go maximum speed with no errors.

        post_delay (Optional[Union[int, float]]): time (in seconds) to wait
            after the *start* of each set, whether part of a sweep or not.
            Can be set to 0 to go maximum speed with no errors.

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

        get_parser ( Optional[function]): function to transform the response
            from get to the final output value. See also val_mapping

        set_parser (Optional[function]): function to transform the input set
            value to an encoded value sent to the instrument.
            See also val_mapping.

        vals (Optional[Validator]): a Validator object for this parameter

        max_val_age (Optional[float]): The max time (in seconds) to trust a
            saved value obtained from get_latest(). If this parameter has not
            been set or measured more recently than this, perform an
            additional measurement.

        metadata (Optional[dict]): extra information to include with the
            JSON snapshot of the parameter
    """
    get_raw = None  # type: Optional[Callable]
    set_raw = None  # type: Optional[Callable]

    def __init__(self, name: str,
                 instrument: Optional['Instrument'],
                 snapshot_get: bool=True,
                 metadata: Optional[dict]=None,
                 step: Optional[Union[int, float]]=None,
                 scale: Optional[Union[Number, Iterable[Number]]]=None,
                 offset: Optional[Union[Number, Iterable[Number]]]=None,
                 inter_delay: Union[int, float]=0,
                 post_delay: Union[int, float]=0,
                 val_mapping: Optional[dict]=None,
                 get_parser: Optional[Callable]=None,
                 set_parser: Optional[Callable]=None,
                 snapshot_value: bool=True,
                 max_val_age: Optional[float]=None,
                 vals: Optional[Validator]=None,
                 **kwargs) -> None:
        super().__init__(metadata)
        if not str(name).isidentifier():
            raise ValueError(f"Parameter name must be a valid identifier "
                             f"got {name} which is not. Parameter names "
                             f"cannot start with a number and "
                             f"must not contain spaces or special characters")
        self.name = str(name)
        self.short_name = str(name)
        self._instrument = instrument
        self._snapshot_get = snapshot_get
        self._snapshot_value = snapshot_value

        if not isinstance(vals, (Validator, type(None))):
            raise TypeError('vals must be None or a Validator')
        elif val_mapping is not None:
            vals = Enum(*val_mapping.keys())
        self.vals = vals

        self.step = step
        self.scale = scale
        self.offset = offset
        self.raw_value = None

        self.inter_delay = inter_delay
        self.post_delay = post_delay

        self.val_mapping = val_mapping
        if val_mapping is None:
            self.inverse_val_mapping = None
        else:
            self.inverse_val_mapping = {v: k for k, v in val_mapping.items()}

        self.get_parser = get_parser
        self.set_parser = set_parser

        # record of latest value and when it was set or measured
        # what exactly this means is different for different subclasses
        # but they all use the same attributes so snapshot is consistent.
        self._latest = {'value': None, 'ts': None, 'raw_value': None}
        self.get_latest = GetLatest(self, max_val_age=max_val_age)

        if hasattr(self, 'get_raw') and self.get_raw is not None:
            self.get = self._wrap_get(self.get_raw)
        elif hasattr(self, 'get'):
            warnings.warn('Wrapping get method, original get method will not '
                          'be directly accessible. It is recommended to '
                          'define get_raw in your subclass instead.' )
            self.get = self._wrap_get(self.get)
        if hasattr(self, 'set_raw') and self.set_raw is not None:
            self.set = self._wrap_set(self.set_raw)
        elif hasattr(self, 'set'):
            warnings.warn('Wrapping set method, original set method will not '
                          'be directly accessible. It is recommended to '
                          'define set_raw in your subclass instead.' )
            self.set = self._wrap_set(self.set)

        # subclasses should extend this list with extra attributes they
        # want automatically included in the snapshot
        self._meta_attrs = ['name', 'instrument', 'step', 'scale', 'offset',
                            'inter_delay', 'post_delay', 'val_mapping', 'vals']

        # Specify time of last set operation, used when comparing to delay to
        # check if additional waiting time is needed before next set
        self._t_last_set = time.perf_counter()

    def __str__(self):
        """Include the instrument name with the Parameter name if possible."""
        inst_name = getattr(self._instrument, 'name', '')
        if inst_name:
            return '{}_{}'.format(inst_name, self.name)
        else:
            return self.name

    def __repr__(self):
        return named_repr(self)

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            if hasattr(self, 'get'):
                return self.get()
            else:
                raise NotImplementedError('no get cmd found in' +
                                          ' Parameter {}'.format(self.name))
        else:
            if hasattr(self, 'set'):
                self.set(*args, **kwargs)
            else:
                raise NotImplementedError('no set cmd found in' +
                                          ' Parameter {}'.format(self.name))

    def snapshot_base(self, update: bool=False,
                      params_to_skip_update: Sequence[str]=None) -> Dict[str, Any]:
        """
        State of the parameter as a JSON-compatible dict.

        Args:
            update (bool): If True, update the state by calling
                parameter.get().
                If False, just use the latest values in memory.
            params_to_skip_update: No effect but may be passed from super Class:

        Returns:
            dict: base snapshot
        """

        if hasattr(self, 'get') and self._snapshot_get \
                and self._snapshot_value and update:
            self.get()

        state = copy(self._latest) # type: Dict[str, Any]
        state['__class__'] = full_class(self)
        state['full_name'] = str(self)

        if not self._snapshot_value:
            state.pop('value')
            state.pop('raw_value', None)

        if isinstance(state['ts'], datetime):
            dttime = state['ts'] # type: datetime
            state['ts'] = dttime.strftime('%Y-%m-%d %H:%M:%S')

        for attr in set(self._meta_attrs):
            if attr == 'instrument' and self._instrument:
                state.update({
                    'instrument': full_class(self._instrument),
                    'instrument_name': self._instrument.name
                })
            else:
                val = getattr(self, attr, None)
                if val is not None:
                    attr_strip = attr.lstrip('_')  # strip leading underscores
                    if isinstance(val, Validator):
                        state[attr_strip] = repr(val)
                    else:
                        state[attr_strip] = val

        return state

    def _save_val(self, value, validate=False):
        """
        Update latest
        """
        if validate:
            self.validate(value)
        if (self.get_parser is None and
            self.set_parser is None and
            self.val_mapping is None and
            self.scale is None and
            self.offset is None):
                self.raw_value = value
        self._latest = {'value': value, 'ts': datetime.now(),
                        'raw_value': self.raw_value}

    def _wrap_get(self, get_function):
        @wraps(get_function)
        def get_wrapper(*args, **kwargs):
            try:
                # There might be cases where a .get also has args/kwargs
                value = get_function(*args, **kwargs)
                self.raw_value = value

                if self.get_parser is not None:
                    value = self.get_parser(value)

                # apply offset first (native scale)
                if self.offset is not None:
                    # offset values
                    if isinstance(self.offset, collections.abc.Iterable):
                        # offset contains multiple elements, one for each value
                        value = tuple(value - offset for value, offset
                                      in zip(value, self.offset))
                    elif isinstance(value, collections.abc.Iterable):
                        # Use single offset for all values
                        value = tuple(value - self.offset for value in value)
                    else:
                        value -= self.offset

                # scale second
                if self.scale is not None:
                    # Scale values
                    if isinstance(self.scale, collections.abc.Iterable):
                        # Scale contains multiple elements, one for each value
                        value = tuple(value / scale for value, scale
                                      in zip(value, self.scale))
                    elif isinstance(value, collections.abc.Iterable):
                        # Use single scale for all values
                        value = tuple(value / self.scale for value in value)
                    else:
                        value /= self.scale

                if self.val_mapping is not None:
                    if value in self.inverse_val_mapping:
                        value = self.inverse_val_mapping[value]
                    else:
                        try:
                            value = self.inverse_val_mapping[int(value)]
                        except (ValueError, KeyError):
                            raise KeyError("'{}' not in val_mapping".format(value))
                self._save_val(value)
                return value
            except Exception as e:
                e.args = e.args + ('getting {}'.format(self),)
                raise e

        return get_wrapper

    def _wrap_set(self, set_function):
        @wraps(set_function)
        def set_wrapper(value, **kwargs):
            try:
                self.validate(value)

                # In some cases intermediate sweep values must be used.
                # Unless `self.step` is defined, get_sweep_values will return
                # a list containing only `value`.
                steps = self.get_ramp_values(value, step=self.step)

                for step_index, val_step in enumerate(steps):
                    # even if the final value is valid we may be generating
                    # steps that are not so validate them too
                    self.validate(val_step)
                    if self.val_mapping is not None:
                        # Convert set values using val_mapping dictionary
                        raw_value = self.val_mapping[val_step]
                    else:
                        raw_value = val_step

                    # transverse transformation in reverse order as compared to
                    # getter:
                    # apply scale first
                    if self.scale is not None:
                        if isinstance(self.scale, collections.abc.Iterable):
                            # Scale contains multiple elements, one for each value
                            raw_value = tuple(val * scale for val, scale
                                              in zip(raw_value, self.scale))
                        else:
                            # Use single scale for all values
                            raw_value *= self.scale

                    # apply offset next
                    if self.offset is not None:
                        if isinstance(self.offset, collections.abc.Iterable):
                            # offset contains multiple elements, one for each value
                            raw_value = tuple(val + offset for val, offset
                                              in zip(raw_value, self.offset))
                        else:
                            # Use single offset for all values
                            raw_value += self.offset

                    # parser last
                    if self.set_parser is not None:
                        raw_value = self.set_parser(raw_value)

                    # Check if delay between set operations is required
                    t_elapsed = time.perf_counter() - self._t_last_set
                    if t_elapsed < self.inter_delay:
                        # Sleep until time since last set is larger than
                        # self.inter_delay
                        time.sleep(self.inter_delay - t_elapsed)

                    # Start timer to measure execution time of set_function
                    t0 = time.perf_counter()

                    set_function(raw_value, **kwargs)
                    self.raw_value = raw_value
                    self._save_val(val_step,
                                   validate=False)

                    # Update last set time (used for calculating delays)
                    self._t_last_set = time.perf_counter()

                    # Check if any delay after setting is required
                    t_elapsed = self._t_last_set - t0
                    if t_elapsed < self.post_delay:
                        # Sleep until total time is larger than self.post_delay
                        time.sleep(self.post_delay - t_elapsed)

            except Exception as e:
                e.args = e.args + ('setting {} to {}'.format(self, value),)
                raise e

        return set_wrapper

    def get_ramp_values(self, value: Union[float, int, Sized],
                        step: Union[float, int]=None) -> List[Union[float,
                                                                    int,
                                                                    Sized]]:
        """
        Return values to sweep from current value to target value.
        This method can be overridden to have a custom sweep behaviour.
        It can even be overridden by a generator.
        Args:
            value: target value
            step: maximum step size

        Returns:
            List of stepped values, including target value.
        """
        if step is None:
            return [value]
        else:
            if isinstance(value, collections.abc.Sized) and len(value) > 1:
                raise RuntimeError("Don't know how to step a parameter with more than one value")
            if self.get_latest() is None:
                self.get()
            start_value = self.get_latest()

            if not (isinstance(start_value, (int, float)) and
                    isinstance(value, (int, float))):
                # something weird... parameter is numeric but one of the ends
                # isn't, even though it's valid.
                # probably MultiType with a mix of numeric and non-numeric types
                # just set the endpoint and move on
                logging.warning(
                    'cannot sweep {} from {} to {} - jumping.'.format(
                        self.name, start_value, value))
                return []

            # drop the initial value, we're already there
            return permissive_range(start_value, value, step)[1:] + [value]

    def validate(self, value):
        """
        Validate value

        Args:
            value (any): value to validate

        """
        if self._instrument:
            context = (getattr(self._instrument, 'name', '') or
                       str(self._instrument.__class__)) + '.' + self.name
        else:
            context = self.name
        if self.vals is not None:
            self.vals.validate(value, 'Parameter: ' + context)

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, step: Union[int, float]):
        """
        Configure whether this Parameter uses steps during set operations.
        If step is a positive number, this is the maximum value change
        allowed in one hardware call, so a single set can result in many
        calls to the hardware if the starting value is far from the target.

        Args:
            step (Union[int, float]): A positive number, the largest change
                allowed in one call. All but the final change will attempt to
                change by +/- step exactly

        Raises:
            TypeError: if step is not numeric
            ValueError: if step is negative
            TypeError:  if step is not integer for an integer parameter
            TypeError: if step is not a number
        """
        if step is None:
            self._step = step # type: Optional[Union[float, int]]
        elif not getattr(self.vals, 'is_numeric', True):
            raise TypeError('you can only step numeric parameters')
        elif not isinstance(step, (int, float)):
            raise TypeError('step must be a number')
        elif step == 0:
            self._step = None
        elif step <= 0:
            raise ValueError('step must be positive')
        elif isinstance(self.vals, Ints) and not isinstance(step, int):
            raise TypeError('step must be a positive int for an Ints parameter')
        else:
            self._step = step

    @property
    def post_delay(self):
        """Delay time after *start* of set operation, for each set"""
        return self._post_delay

    @post_delay.setter
    def post_delay(self, post_delay):
        """
        Configure this parameter with a delay after the *start* of every set
        operation.

        Typically used in conjunction with `step` to create an effective
        ramp rate, but can also be used without a `step` to enforce a delay
        *after* every set. One might think of post_delay as how long a set
        operation is supposed to take. For example, there might be an
        instrument that needs extra time after setting a parameter although
        the command for setting the parameter returns quickly.

        Args:
            post_delay(Union[int, float]): the target time after the *start*
                of a set operation. The actual time will not be shorter than
                this, but may be longer if the underlying set call takes longer.

        Raises:
            TypeError: If delay is not int nor float
            ValueError: If delay is negative
        """
        if not isinstance(post_delay, (int, float)):
            raise TypeError(
                'post_delay ({}) must be a number'.format(post_delay))
        if post_delay < 0:
            raise ValueError(
                'post_delay ({}) must not be negative'.format(post_delay))
        self._post_delay = post_delay

    @property
    def inter_delay(self):
        """Delay time between consecutive set operations"""
        return self._inter_delay

    @inter_delay.setter
    def inter_delay(self, inter_delay):
        """
        Configure this parameter with a delay between set operations.

        Typically used in conjunction with `step` to create an effective
        ramp rate, but can also be used without a `step` to enforce a delay
        *between* sets.

        Args:
            inter_delay(Union[int, float]): the minimum time between set calls.
                The actual time will not be shorter than this, but may be longer
                if the underlying set call takes longer.

        Raises:
            TypeError: If delay is not int nor float
            ValueError: If delay is negative
        """
        if not isinstance(inter_delay, (int, float)):
            raise TypeError(
                'inter_delay ({}) must be a number'.format(inter_delay))
        if inter_delay < 0:
            raise ValueError(
                'inter_delay ({}) must not be negative'.format(inter_delay))
        self._inter_delay = inter_delay

    @property
    def full_name(self):
        return "_".join(self.name_parts)

    def set_validator(self, vals):
        """
            Deprecated Set a validator `vals` for this parameter.
                Args:
                    vals (Validator):  validator to set

        """
        warnings.warn(
            "set_validator is deprected use `inst.vals = MyValidator` instead")
        if isinstance(vals, Validator):
            self.vals = vals
        else:
            raise TypeError('vals must be a Validator')

    @property
    def instrument(self) -> Optional['InstrumentBase']:
        """
        Return the first instrument that this parameter is bound to.
        E.g if this is bound to a channel it will return the channel
        and not the instrument that the channel is bound too. Use
        :meth:`root_instrument` to get the real instrument.
        """
        return self._instrument

    @property
    def root_instrument(self) -> Optional['InstrumentBase']:
        """
        Return the fundamental instrument that this parameter belongs too.
        E.g if the parameter is bound to a channel this will return the
        fundamental instrument that that channel belongs to. Use
        :meth:`instrument` to get the channel.
        """
        if self._instrument is not None:
            return self._instrument.root_instrument
        else:
            return None

    def set_to(self, value):
        """
        Use a context manager to temporarily set the value of a parameter to
        a value. Example:

        >>> from qcodes import Parameter
        >>> p = Parameter("p", set_cmd=None, get_cmd=None)
        >>> with p.set_to(3):
        ...    print(f"p value in with block {p.get()}")
        >>> print(f"p value outside with block {p.get()}")
        """
        context_manager = _SetParamContext(self)
        self.set(value)
        return context_manager

    @property
    def name_parts(self) -> List[str]:
        if self.instrument is not None:
            name_parts = getattr(self.instrument, 'name_parts', [])
            if name_parts == []:
                # add fallback for the case where someone has bound
                # the parameter to something that is not an instrument
                # but perhaps it has a name anyway?
                name = getattr(self.instrument, 'name', None)
                if name is not None:
                    name_parts = [name]
        else:
            name_parts = []

        name_parts.append(self.short_name)
        return name_parts


class Parameter(_BaseParameter):
    """
    A parameter that represents a single degree of freedom.
    This is the standard parameter for Instruments, though it can also be
    used as a variable, i.e. storing/retrieving a value, or be subclassed for
    more complex uses.

    By default only gettable, returning its last value.
    This behaviour can be modified in two ways:

    1. Providing a ``get_cmd``/``set_cmd``, which can of the following:

       a. callable, with zero args for get_cmd, one arg for set_cmd
       b. VISA command string
       c. None, in which case it retrieves its last value for ``get_cmd``,
          and stores a value for ``set_cmd``
       d. False, in which case trying to get/set will raise an error.

    2. Creating a subclass with an explicit ``get_raw``/``set_raw`` method.
       This enables more advanced functionality. The ``get_raw`` and
       ``set_raw`` methods are automatically wrapped to provide ``get`` and
       ``set``.

    Parameters have a ``.get_latest`` method that simply returns the most
    recent set or measured value. This can be called ( ``param.get_latest()`` )
    or used in a ``Loop`` as if it were a (gettable-only) parameter itself:

        ``Loop(...).each(param.get_latest)``


    Args:
        name (str): the local name of the parameter. Should be a valid
            identifier, ie no spaces or special characters. If this parameter
            is part of an Instrument or Station, this is how it will be
            referenced from that parent, ie ``instrument.name`` or
            ``instrument.parameters[name]``

        instrument (Optional[Instrument]): the instrument this parameter
            belongs to, if any

        label (Optional[str]): Normally used as the axis label when this
            parameter is graphed, along with ``unit``.

        unit (Optional[str]): The unit of measure. Use ``''`` for unitless.

        snapshot_get (Optional[bool]): False prevents any update to the
            parameter during a snapshot, even if the snapshot was called with
            ``update=True``, for example if it takes too long to update.
            Default True.

        snapshot_value (Optional[bool]): False prevents parameter value to be
            stored in the snapshot. Useful if the value is large.

        step (Optional[Union[int, float]]): max increment of parameter value.
            Larger changes are broken into multiple steps this size.
            When combined with delays, this acts as a ramp.

        scale (Optional[float]): Scale to multiply value with before
            performing set. the internally multiplied value is stored in
            `raw_value`. Can account for a voltage divider.

        inter_delay (Optional[Union[int, float]]): Minimum time (in seconds)
            between successive sets. If the previous set was less than this,
            it will wait until the condition is met.
            Can be set to 0 to go maximum speed with no errors.

        post_delay (Optional[Union[int, float]]): time (in seconds) to wait
            after the *start* of each set, whether part of a sweep or not.
            Can be set to 0 to go maximum speed with no errors.

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

        get_parser ( Optional[function]): function to transform the response
            from get to the final output value. See also val_mapping

        set_parser (Optional[function]): function to transform the input set
            value to an encoded value sent to the instrument.
            See also val_mapping.

        vals (Optional[Validator]): Allowed values for setting this parameter.
            Only relevant if settable. Defaults to ``Numbers()``

        max_val_age (Optional[float]): The max time (in seconds) to trust a
            saved value obtained from get_latest(). If this parameter has not
            been set or measured more recently than this, perform an
            additional measurement.

        docstring (Optional[str]): documentation string for the __doc__
            field of the object. The __doc__ field of the instance is used by
            some help systems, but not all

        metadata (Optional[dict]): extra information to include with the
            JSON snapshot of the parameter

    """

    def __init__(self, name: str,
                 instrument: Optional['Instrument']=None,
                 label: Optional[str]=None,
                 unit: Optional[str]=None,
                 get_cmd: Optional[Union[str, Callable, bool]]=None,
                 set_cmd:  Optional[Union[str, Callable, bool]]=False,
                 initial_value: Optional[Union[float, int, str]]=None,
                 max_val_age: Optional[float]=None,
                 vals: Optional[Validator]=None,
                 docstring: Optional[str]=None,
                 **kwargs) -> None:
        super().__init__(name=name, instrument=instrument, vals=vals, **kwargs)

        # Enable set/get methods if get_cmd/set_cmd is given
        # Called first so super().__init__ can wrap get/set methods
        if not hasattr(self, 'get') and get_cmd is not False:
            if get_cmd is None:
                if max_val_age is not None:
                    raise SyntaxError('Must have get method or specify get_cmd '
                                      'when max_val_age is set')
                self.get_raw = lambda: self._latest['raw_value']
            else:
                exec_str_ask = instrument.ask if instrument else None
                self.get_raw = Command(arg_count=0, cmd=get_cmd, exec_str=exec_str_ask)
            self.get = self._wrap_get(self.get_raw)

        if not hasattr(self, 'set') and set_cmd is not False:
            if set_cmd is None:
                self.set_raw = partial(self._save_val, validate=False)# type: Callable
            else:
                exec_str_write = instrument.write if instrument else None
                self.set_raw = Command(arg_count=1, cmd=set_cmd, exec_str=exec_str_write)# type: Callable
            self.set = self._wrap_set(self.set_raw)

        self._meta_attrs.extend(['label', 'unit', 'vals'])

        self.label = name if label is None else label
        self.unit = unit if unit is not None else ''

        if initial_value is not None:
            self.set(initial_value)

        # generate default docstring
        self.__doc__ = os.linesep.join((
            'Parameter class:',
            '',
            '* `name` %s' % self.name,
            '* `label` %s' % self.label,
            '* `unit` %s' % self.unit,
            '* `vals` %s' % repr(self.vals)))

        if docstring is not None:
            self.__doc__ = os.linesep.join((
                docstring,
                '',
                self.__doc__))

    def __getitem__(self, keys):
        """
        Slice a Parameter to get a SweepValues object
        to iterate over during a sweep
        """
        return SweepFixedValues(self, keys)

    def increment(self, value):
        """ Increment the parameter with a value

        Args:
            value (float): value to be added to the parameter
        """
        self.set(self.get() + value)

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


class ArrayParameter(_BaseParameter):
    """
    A gettable parameter that returns an array of values.
    Not necessarily part of an instrument.

    Subclasses should define a ``.get_raw`` method, which returns an array.
    This method is automatically wrapped to provide a ``.get``` method.
    When used in a ``Loop`` or ``Measure`` operation, this will be entered
    into a single ``DataArray``, with extra dimensions added by the ``Loop``.
    The constructor args describe the array we expect from each ``.get`` call
    and how it should be handled.

    For now you must specify upfront the array shape, and this cannot change
    from one call to the next. Later we intend to require only that you specify
    the dimension, and the size of each dimension can vary from call to call.

    Note: If you want ``.get`` to save the measurement for ``.get_latest``,
    you must explicitly call ``self._save_val(items)`` inside ``.get_raw``.

    Args:
        name (str): the local name of the parameter. Should be a valid
            identifier, ie no spaces or special characters. If this parameter
            is part of an Instrument or Station, this is how it will be
            referenced from that parent, ie ``instrument.name`` or
            ``instrument.parameters[name]``

        shape (Tuple[int]): The shape (as used in numpy arrays) of the array
            to expect. Scalars should be denoted by (), 1D arrays as (n,),
            2D arrays as (n, m), etc.

        instrument (Optional[Instrument]): the instrument this parameter
            belongs to, if any

        label (Optional[str]): Normally used as the axis label when this
            parameter is graphed, along with ``unit``.

        unit (Optional[str]): The unit of measure. Use ``''`` for unitless.

        setpoints (Optional[Tuple[setpoint_array]]):
            ``setpoint_array`` can be a DataArray, numpy.ndarray, or sequence.
            The setpoints for each dimension of the returned array. An
            N-dimension item should have N setpoint arrays, where the first is
            1D, the second 2D, etc.
            If omitted for any or all items, defaults to integers from zero in
            each respective direction.
            Note: if the setpoints will be different each measurement, leave
            this out and return the setpoints (with extra names) in ``.get``.

        setpoint_names (Optional[Tuple[str]]): one identifier (like
            ``name``) per setpoint array. Ignored if a setpoint is a
            DataArray, which already has a name.

        setpoint_labels (Optional[Tuple[str]]): one label (like ``labels``)
            per setpoint array. Ignored if a setpoint is a DataArray, which
            already has a label.

        setpoint_units (Optional[Tuple[str]]): one label (like ``v``)
            per setpoint array. Ignored if a setpoint is a DataArray, which
            already has a unit.

        docstring (Optional[str]): documentation string for the __doc__
            field of the object. The __doc__ field of the instance is used by
            some help systems, but not all

        snapshot_get (bool): Prevent any update to the parameter, for example
            if it takes too long to update. Default True.

        snapshot_value: Should the value of the parameter be stored in the
            snapshot. Unlike Parameter this defaults to False as
            ArrayParameters are potentially huge.

        metadata (Optional[dict]): extra information to include with the
            JSON snapshot of the parameter
    """

    def __init__(self,
                 name: str,
                 shape: Sequence[int],
                 instrument: Optional['Instrument']=None,
                 label: Optional[str]=None,
                 unit: Optional[str]=None,
                 setpoints: Optional[Sequence]=None,
                 setpoint_names: Optional[Sequence[str]]=None,
                 setpoint_labels: Optional[Sequence[str]]=None,
                 setpoint_units: Optional[Sequence[str]]=None,
                 docstring: Optional[str]=None,
                 snapshot_get: bool=True,
                 snapshot_value: bool=False,
                 metadata: Optional[dict]=None) -> None:
        super().__init__(name, instrument, snapshot_get, metadata,
                         snapshot_value=snapshot_value)

        if hasattr(self, 'set'):
            # TODO (alexcjohnson): can we support, ala Combine?
            raise AttributeError('ArrayParameters do not support set '
                                 'at this time.')

        self._meta_attrs.extend(['setpoint_names', 'setpoint_labels',
                                 'setpoint_units', 'label', 'unit'])

        self.label = name if label is None else label
        self.unit = unit if unit is not None else ''

        nt: Type[None] = type(None)

        if not is_sequence_of(shape, int):
            raise ValueError('shapes must be a tuple of ints, not ' +
                             repr(shape))
        self.shape = shape

        # require one setpoint per dimension of shape
        sp_shape = (len(shape),)

        sp_types = (nt, DataArray, collections.abc.Sequence,
                    collections.abc.Iterator, numpy.ndarray)
        if (setpoints is not None and
                not is_sequence_of(setpoints, sp_types, shape=sp_shape)):
            raise ValueError('setpoints must be a tuple of arrays')
        if (setpoint_names is not None and
                not is_sequence_of(setpoint_names, (nt, str), shape=sp_shape)):
            raise ValueError('setpoint_names must be a tuple of strings')
        if (setpoint_labels is not None and
                not is_sequence_of(setpoint_labels, (nt, str),
                                   shape=sp_shape)):
            raise ValueError('setpoint_labels must be a tuple of strings')
        if (setpoint_units is not None and
                not is_sequence_of(setpoint_units, (nt, str),
                                   shape=sp_shape)):
            raise ValueError('setpoint_units must be a tuple of strings')

        self.setpoints = setpoints
        self.setpoint_names = setpoint_names
        self.setpoint_labels = setpoint_labels
        self.setpoint_units = setpoint_units

        self.__doc__ = os.linesep.join((
            'Parameter class:',
            '',
            '* `name` %s' % self.name,
            '* `label` %s' % self.label,
            '* `unit` %s' % self.unit,
            '* `shape` %s' % repr(self.shape)))

        if docstring is not None:
            self.__doc__ = os.linesep.join((
                docstring,
                '',
                self.__doc__))

        if not hasattr(self, 'get') and not hasattr(self, 'set'):
            raise AttributeError('ArrayParameter must have a get, set or both')

    @property
    def setpoint_full_names(self):
        """
        Full names of setpoints including instrument names if available
        """
        if self.setpoint_names is None:
            return None
        # omit the last part of name_parts which is the parameter name
        # and not part of the setpoint names
        inst_name = "_".join(self.name_parts[:-1])
        if inst_name != '':
            spnames = []
            for spname in self.setpoint_names:
                if spname is not None:
                    spnames.append(inst_name + '_' + spname)
                else:
                    spnames.append(None)
            return tuple(spnames)
        else:
            return self.setpoint_names


def _is_nested_sequence_or_none(obj, types, shapes):
    """Validator for MultiParameter setpoints/names/labels"""
    if obj is None:
        return True

    if not is_sequence_of(obj, tuple, shape=(len(shapes),)):
        return False

    for obji, shapei in zip(obj, shapes):
        if not is_sequence_of(obji, types, shape=(len(shapei),)):
            return False

    return True


class MultiParameter(_BaseParameter):
    """
    A gettable parameter that returns multiple values with separate names,
    each of arbitrary shape.
    Not necessarily part of an instrument.

    Subclasses should define a ``.get_raw`` method, which returns a sequence of
    values. This method is automatically wrapped to provide a ``.get``` method.
    When used in a ``Loop`` or ``Measure`` operation, each of these
    values will be entered into a different ``DataArray``. The constructor
    args describe what data we expect from each ``.get`` call and how it
    should be handled. ``.get`` should always return the same number of items,
    and most of the constructor arguments should be tuples of that same length.

    For now you must specify upfront the array shape of each item returned by
    ``.get_raw``, and this cannot change from one call to the next. Later we
    intend to require only that you specify the dimension of each item returned,
    and the size of each dimension can vary from call to call.

    Note: If you want ``.get`` to save the measurement for ``.get_latest``,
    you must explicitly call ``self._save_val(items)`` inside ``.get_raw``.

    Args:
        name (str): the local name of the whole parameter. Should be a valid
            identifier, ie no spaces or special characters. If this parameter
            is part of an Instrument or Station, this is how it will be
            referenced from that parent, ie ``instrument.name`` or
            ``instrument.parameters[name]``

        names (Tuple[str]): A name for each item returned by a ``.get``
            call. Will be used as the basis of the ``DataArray`` names
            when this parameter is used to create a ``DataSet``.

        shapes (Tuple[Tuple[int]]): The shape (as used in numpy arrays) of
            each item. Scalars should be denoted by (), 1D arrays as (n,),
            2D arrays as (n, m), etc.

        instrument (Optional[Instrument]): the instrument this parameter
            belongs to, if any

        labels (Optional[Tuple[str]]): A label for each item. Normally used
            as the axis label when a component is graphed, along with the
            matching entry from ``units``.

        units (Optional[Tuple[str]]): The unit of measure for each item.
            Use ``''`` or ``None`` for unitless values.

        setpoints (Optional[Tuple[Tuple[setpoint_array]]]):
            ``setpoint_array`` can be a DataArray, numpy.ndarray, or sequence.
            The setpoints for each returned array. An N-dimension item should
            have N setpoint arrays, where the first is 1D, the second 2D, etc.
            If omitted for any or all items, defaults to integers from zero in
            each respective direction.
            Note: if the setpoints will be different each measurement, leave
            this out and return the setpoints (with extra names) in ``.get``.

        setpoint_names (Optional[Tuple[Tuple[str]]]): one identifier (like
            ``name``) per setpoint array. Ignored if a setpoint is a
            DataArray, which already has a name.

        setpoint_labels (Optional[Tuple[Tuple[str]]]): one label (like
            ``labels``) per setpoint array. Ignored if a setpoint is a
            DataArray, which already has a label.

        setpoint_units (Optional[Tuple[Tuple[str]]]): one unit (like
            ``V``) per setpoint array. Ignored if a setpoint is a
            DataArray, which already has a unit.

        docstring (Optional[str]): documentation string for the __doc__
            field of the object. The __doc__ field of the instance is used by
            some help systems, but not all

        snapshot_get (bool): Prevent any update to the parameter, for example
            if it takes too long to update. Default True.

        snapshot_value: Should the value of the parameter be stored in the
            snapshot. Unlike Parameter this defaults to False as
            MultiParameters are potentially huge.

        metadata (Optional[dict]): extra information to include with the
            JSON snapshot of the parameter
    """

    def __init__(self,
                 name: str,
                 names: Sequence[str],
                 shapes: Sequence[Sequence[Optional[int]]],
                 instrument: Optional['Instrument']=None,
                 labels: Optional[Sequence[str]]=None,
                 units: Optional[Sequence[str]]=None,
                 setpoints: Optional[Sequence[Sequence]]=None,
                 setpoint_names: Optional[Sequence[Sequence[str]]]=None,
                 setpoint_labels: Optional[Sequence[Sequence[str]]]=None,
                 setpoint_units: Optional[Sequence[Sequence[str]]]=None,
                 docstring: str=None,
                 snapshot_get: bool=True,
                 snapshot_value: bool=False,
                 metadata: Optional[dict]=None) -> None:
        super().__init__(name, instrument, snapshot_get, metadata,
                         snapshot_value=snapshot_value)

        if hasattr(self, 'set'):
            # TODO (alexcjohnson): can we support, ala Combine?
            warnings.warn('MultiParameters do not support set at this time.')

        self._meta_attrs.extend(['setpoint_names', 'setpoint_labels',
                                 'setpoint_units', 'names', 'labels', 'units'])

        if not is_sequence_of(names, str):
            raise ValueError('names must be a tuple of strings, not ' +
                             repr(names))

        self.names = names
        self.labels = labels if labels is not None else names
        self.units = units if units is not None else [''] * len(names)

        nt: Type[None] = type(None)

        if (not is_sequence_of(shapes, int, depth=2) or
                len(shapes) != len(names)):
            raise ValueError('shapes must be a tuple of tuples '
                             'of ints, not ' + repr(shapes))
        self.shapes = shapes

        sp_types = (nt, DataArray, collections.abc.Sequence,
                    collections.abc.Iterator, numpy.ndarray)
        if not _is_nested_sequence_or_none(setpoints, sp_types, shapes):
            raise ValueError('setpoints must be a tuple of tuples of arrays')

        if not _is_nested_sequence_or_none(setpoint_names, (nt, str), shapes):
            raise ValueError(
                'setpoint_names must be a tuple of tuples of strings')

        if not _is_nested_sequence_or_none(setpoint_labels, (nt, str), shapes):
            raise ValueError(
                'setpoint_labels must be a tuple of tuples of strings')

        if not _is_nested_sequence_or_none(setpoint_units, (nt, str), shapes):
            raise ValueError(
                'setpoint_units must be a tuple of tuples of strings')

        self.setpoints = setpoints
        self.setpoint_names = setpoint_names
        self.setpoint_labels = setpoint_labels
        self.setpoint_units = setpoint_units

        self.__doc__ = os.linesep.join((
            'MultiParameter class:',
            '',
            '* `name` %s' % self.name,
            '* `names` %s' % ', '.join(self.names),
            '* `labels` %s' % ', '.join(self.labels),
            '* `units` %s' % ', '.join(self.units)))

        if docstring is not None:
            self.__doc__ = os.linesep.join((
                docstring,
                '',
                self.__doc__))

        if not hasattr(self, 'get') and not hasattr(self, 'set'):
            raise AttributeError('MultiParameter must have a get, set or both')

    @property
    def short_names(self):
        """
        short_names is indentical to names i.e. the names of the paramter parts
        but does not add the intrument name.

        It exists for consistency with instruments and other parameters.
        """

        return self.names

    @property
    def full_names(self):
        """Include the instrument name with the Parameter names if possible."""
        inst_name = "_".join(self.name_parts[:-1])
        if inst_name != '':
            return [inst_name + '_' + name for name in self.names]
        else:
            return self.names

    @property
    def setpoint_full_names(self):
        """
        Full names of setpoints including instrument names if available
        """
        if self.setpoint_names is None:
            return None
        # omit the last part of name_parts which is the parameter name
        # and not part of the setpoint names
        inst_name = "_".join(self.name_parts[:-1])
        if inst_name != '':
            full_sp_names = []
            for sp_group in self.setpoint_names:
                full_sp_names_subgroupd = []
                for spname in sp_group:
                    if spname is not None:
                        full_sp_names_subgroupd.append(inst_name + '_' + spname)
                    else:
                        full_sp_names_subgroupd.append(None)
                full_sp_names.append(tuple(full_sp_names_subgroupd))

            return tuple(full_sp_names)
        else:
            return self.setpoint_names


class GetLatest(DelegateAttributes):
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

        max_val_age (Optional[int]): The max time (in seconds) to trust a
            saved value obtained from get_latest(). If this parameter has not
            been set or measured more recently than this, perform an
            additional measurement.
    """
    def __init__(self, parameter, max_val_age=None):
        self.parameter = parameter
        self.max_val_age = max_val_age

    delegate_attr_objects = ['parameter']
    omit_delegate_attrs = ['set']

    def get(self):
        """Return latest value if time since get was less than
        `self.max_val_age`, otherwise perform `get()` and return result
        """
        state = self.parameter._latest
        if self.max_val_age is None:
            # Return last value since max_val_age is not specified
            return state['value']
        else:
            oldest_ok_val = datetime.now() - timedelta(seconds=self.max_val_age)
            if state['ts'] is None or state['ts'] < oldest_ok_val:
                # Time of last get exceeds max_val_age seconds, need to
                # perform new .get()
                return self.parameter.get()
            else:
                return state['value']

    def __call__(self):
        return self.get()


def combine(*parameters, name, label=None, unit=None, units=None,
            aggregator=None):
    """
    Combine parameters into one sweepable parameter

    Args:
        *parameters (qcodes.Parameter): the parameters to combine
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
    multi_par = CombinedParameter(parameters, name, label, unit, units,
                                  aggregator)
    return multi_par


class CombinedParameter(Metadatable):
    """ A combined parameter

    Args:
        *parameters (qcodes.Parameter): the parameters to combine
        name (str): the name of the parameter
        label (Optional[str]): the label of the combined parameter
        unit (Optional[str]): the unit of the combined parameter
        aggregator (Optional[Callable[list[any]]]): a function to aggregate
            the set values into one

    A combined parameter sets all the combined parameters at every point of the
    sweep.
    The sets are called in the same order the parameters are, and
    sequentially.
    """

    def __init__(self, parameters: Sequence[Parameter], name: str,
                 label: str = None, unit: str=None, units: str=None,
                 aggregator: Callable=None) -> None:
        super().__init__()
        # TODO(giulioungaretti)temporary hack
        # starthack
        # this is a dummy parameter
        # that mimicks the api that a normal parameter has
        if not name.isidentifier():
            raise ValueError(f"Parameter name must be a valid identifier "
                             f"got {name} which is not. Parameter names "
                             f"cannot start with a number and "
                             f"must not contain spaces or special characters")

        self.parameter = lambda: None
        # mypy will complain that a callable does not have these attributes
        # but you can still create them here.
        self.parameter.full_name = name  # type: ignore
        self.parameter.name = name  # type: ignore
        self.parameter.label = label  # type: ignore

        if units is not None:
            warn_units('CombinedParameter', self)
            if unit is None:
                unit = units
        self.parameter.unit = unit  # type: ignore
        self.setpoints: List[Any] = []
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

    def sweep(self, *array: numpy.ndarray) -> 'CombinedParameter':
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
                raise ValueError('Arrays have different number of setpoints')
            nparray = numpy.array(array).transpose()
        else:
            # cast to array in case users
            # decide to not read docstring
            # and pass a 2d list
            nparray = numpy.array(array[0])
        new = copy(self)
        _error_msg = """ Dimensionality of array does not match\
                        the number of parameter combined. Expected a \
                        {} dimensional array, got a {} dimensional array. \
                        """
        try:
            if nparray.shape[1] != self.dimensionality:
                raise ValueError(_error_msg.format(self.dimensionality,
                                                   nparray.shape[1]))
        except KeyError:
            # this means the array is 1d
            raise ValueError(_error_msg.format(self.dimensionality, 1))

        new.setpoints = nparray.tolist()
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
        meta_data['unit'] = self.parameter.unit
        meta_data['label'] = self.parameter.label
        meta_data['full_name'] = self.parameter.full_name
        meta_data['aggregator'] = repr(getattr(self, 'f', None))
        for param in self.parameters:
            meta_data[str(param)] = param.snapshot()

        return meta_data


class InstrumentRefParameter(Parameter):
    """
    An InstrumentRefParameter

    Args:
        name (string): the name of the parameter that one wants to add.

        instrument (Optional[Instrument]): the "parent" instrument this
            parameter is attached to, if any.

        initial_value (Optional[str]): starting value, may be None even if
            None does not pass the validator. None is only allowed as an
            initial value and cannot be set after initiation.

        **kwargs: Passed to InstrumentRefParameter parent class

    This parameter is useful when one needs a reference to another instrument
    from within an instrument, e.g., when creating a meta instrument that
    sets parameters on instruments it contains.
    """

    def __init__(self, *args, **kwargs):
        kwargs['vals'] = kwargs.get('vals', Strings())
        super().__init__(set_cmd=None, *args, **kwargs)

    # TODO(nulinspiratie) check class works now it's subclassed from Parameter
    def get_instr(self):
        """
        Returns the instance of the instrument with the name equal to the
        value of this parameter.
        """
        ref_instrument_name = self.get()
        # note that _instrument refers to the instrument this parameter belongs
        # to, while the ref_instrument_name is the instrument that is the value
        # of this parameter.
        return self._instrument.find_instrument(ref_instrument_name)


# Deprecated parameters
class StandardParameter(Parameter):
    def __init__(self, name, instrument=None,
                 get_cmd=False, get_parser=None,
                 set_cmd=False, set_parser=None,
                 delay=0, max_delay=None, step=None, max_val_age=3600,
                 vals=None, val_mapping=None, **kwargs):
        super().__init__(name, instrument=instrument,
                 get_cmd=get_cmd, get_parser=get_parser,
                 set_cmd=set_cmd, set_parser=set_parser,
                 post_delay=delay, step=step, max_val_age=max_val_age,
                 vals=vals, val_mapping=val_mapping, **kwargs)
        warnings.warn('`StandardParameter` is deprecated, '
                        'use `Parameter` instead. {}'.format(self))


class ManualParameter(Parameter):
    def __init__(self, name, instrument=None, initial_value=None, **kwargs):
        """
        A simple alias for a parameter that does not have a set or
        a get function. Useful for parameters that do not have a direct
        instrument mapping.
        """
        super().__init__(name=name, instrument=instrument,
                         get_cmd=None, set_cmd=None,
                         initial_value=initial_value, **kwargs)


class ScaledParameter(Parameter):
    """
    Parameter Scaler

    To be used when you use a physical voltage divider or an amplifier to set
    or get a quantity.

    Initialize the parameter by passing the parameter to be measured/set
    and the value of the division OR the gain.

    The scaling value can be either a scalar value or a Qcodes Parameter.

    The parameter scaler acts a your original parameter, but will set the right
    value, and store the gain/division in the metadata.

    Examples:
        Resistive voltage divider
        >>> vd = ScaledParameter(dac.chan0, division = 10)

        Voltage multiplier
        >>> vb = ScaledParameter(dac.chan0, gain = 30, name = 'Vb')

        Transimpedance amplifier
        >>> Id = ScaledParameter(multimeter.amplitude, division = 1e6, name = 'Id', unit = 'A')

    Args:
        output: Physical Parameter that need conversion
        division: the division value
        gain: the gain value
        label: label of this parameter, by default uses 'output' label
            but attaches _amplified or _attenuated depending if gain
            or division has been specified
        name: name of this parameter, by default uses 'output' name
            but attaches _amplified or _attenuated depending if gain
            or division has been specified
        unit: resulting unit. It uses the one of 'output' by default
    """

    class Role(enum.Enum):
        GAIN = enum.auto()
        DIVISION = enum.auto()


    def __init__(self,
                 output: Parameter,
                 division: Union[int, float, Parameter] = None,
                 gain: Union[int, float, Parameter] = None,
                 name: str=None,
                 label: str=None,
                 unit: str=None) -> None:
        # Set the name
        if name:
            self.name = name
        else:
            self.name = "{}_scaled".format(output.name)

        # Set label
        if label:
            self.label = label
        elif name:
            self.label = name
        else:
            self.label = "{}_scaled".format(output.label)

        # Set the unit
        if unit:
            self.unit = unit
        else:
            self.unit = output.unit

        super().__init__(
            name=self.name,
            label=self.label,
            unit=self.unit
            )

        self._wrapped_parameter = output
        self._wrapped_instrument = getattr(output, "_instrument", None)

        # Set the role, either as divider or amplifier
        # Raise an error if nothing is specified
        is_divider = division is not None
        is_amplifier = gain is not None

        if not xor(is_divider, is_amplifier):
            raise ValueError('Provide only division OR gain')

        if is_divider:
            self.role = ScaledParameter.Role.DIVISION
            self._multiplier = division
        elif is_amplifier:
            self.role = ScaledParameter.Role.GAIN
            self._multiplier = gain

        # extend metadata
        self._meta_attrs.extend(["division"])
        self._meta_attrs.extend(["gain"])
        self._meta_attrs.extend(["role"])
        self.metadata['wrapped_parameter'] = self._wrapped_parameter.name
        if self._wrapped_instrument:
            self.metadata['wrapped_instrument'] = getattr(self._wrapped_instrument, "name", None)

    # Internal handling of the multiplier
    # can be either a Parameter or a scalar
    @property
    def _multiplier(self):
        return self._multiplier_parameter

    @_multiplier.setter
    def _multiplier(self, multiplier: Union[int, float, Parameter]):
        if isinstance(multiplier, Parameter):
            self._multiplier_parameter = multiplier
            self.metadata['variable_multiplier'] = self._multiplier_parameter.name
        else:
            self._multiplier_parameter = ManualParameter(
                'multiplier', initial_value=multiplier)
            self.metadata['variable_multiplier'] = False

    # Division of the scaler
    @property
    def division(self):
        if self.role == ScaledParameter.Role.DIVISION:
            return self._multiplier()
        elif self.role == ScaledParameter.Role.GAIN:
            return 1 / self._multiplier()

    @division.setter
    def division(self, division: Union[int, float, Parameter]):
        self.role = ScaledParameter.Role.DIVISION
        self._multiplier = division

    # Gain of the scaler
    @property
    def gain(self):
        if self.role == ScaledParameter.Role.GAIN:
            return self._multiplier()
        elif self.role == ScaledParameter.Role.DIVISION:
            return 1 / self._multiplier()

    @gain.setter
    def gain(self, gain: Union[int, float, Parameter]):
        self.role = ScaledParameter.Role.GAIN
        self._multiplier = gain

    # Getter and setter for the real value
    def get_raw(self) -> Union[int, float]:
        """
        Returns:
            number: value at which was set at the sample
        """
        if self.role == ScaledParameter.Role.GAIN:
            value = self._wrapped_parameter() * self._multiplier()
        elif self.role == ScaledParameter.Role.DIVISION:
            value = self._wrapped_parameter() / self._multiplier()

        self._save_val(value)
        return value

    @property
    def wrapped_parameter(self) -> Parameter:
        """
        Returns:
            the attached unscaled parameter
        """
        return self._wrapped_parameter

    def get_wrapped_parameter_value(self) -> Union[int, float]:
        """
        Returns:
            number: value at which the attached parameter is (i.e. does
            not account for the scaling)
        """
        return self._wrapped_parameter.get()

    def set_raw(self, value: Union[int, float]) -> None:
        """
        Set the value on the wrapped parameter, accounting for the scaling
        """
        if self.role == ScaledParameter.Role.GAIN:
            instrument_value = value / self._multiplier()
        elif self.role == ScaledParameter.Role.DIVISION:
            instrument_value = value * self._multiplier()

        # don't leak unknow type
        instrument_value = cast(Union[int, float], instrument_value)

        self._save_val(value)
        self._wrapped_parameter.set(instrument_value)
