"""
The Parameter module implements Parameter interface
that are the basis of measurements and control within QCoDeS.

Anything that you want to either measure or control within QCoDeS should
satisfy the Parameter interface. Most of the time that is easiest to do
by either using or subclassing one of the classes defined here, but you can
also use any class with the right attributes.

All parameter classes are subclassed from ``._BaseParameter`` (except
CombinedParameter). The _BaseParameter provides functionality that is common
to all parameter types, such as ramping and scaling of values, adding delays
(see documentation for details).

This module defines the following basic classes of parameters as well as some
more specialized ones:

- :class:`.Parameter` is the base class for scalar-valued parameters.
    Two primary ways in which it can be used:

    1. As an :class:`.Instrument` parameter that sends/receives commands.
       Provides a standardized interface to construct strings to pass to the
       :meth:`.Instrument.write` and :meth:`.Instrument.ask` methods
    2. As a variable that stores and returns a value. For instance, for storing
       of values you want to keep track of but cannot set or get electronically.

- :class:`.ParameterWithSetpoints` is intended for array-values parameters.
    This Parameter class is intended for anything where a call to the instrument
    returns an array of values. `This notebook
    <../../examples/Parameters/Simple-Example-of-ParameterWithSetpoints
    .ipynb>`_ gives more detailed examples of how this parameter
    can be used `and this notebook
    <../../examples/writing_drivers/A-ParameterWithSetpoints
    -Example-with-Dual-Setpoints.ipynb>`_ explains writing driver
    using :class:`.ParameterWithSetpoints`.

    :class:`.ParameterWithSetpoints` is supported in a
    :class:`qcodes.dataset.measurements.Measurement` but is not supported by
    the legacy :class:`qcodes.loops.Loop` and :class:`qcodes.measure.Measure`
    measurement types.

- :class:`.DelegateParameter` is intended for proxy-ing other parameters.
    It forwards its ``get`` and ``set`` to the underlying source parameter,
    while allowing to specify label/unit/etc that is different from the
    source parameter.

- :class:`.ArrayParameter` is an older base class for array-valued parameters.
    For any new driver we strongly recommend using
    :class:`.ParameterWithSetpoints` which is both more flexible and
    significantly easier to use. This Parameter is intended for anything for
    which each ``get`` call returns an array of values that all have the same
    type and meaning. Currently not settable, only gettable. Can be used in a
    :class:`qcodes.dataset.measurements.Measurement`
    as well as in the legacy :class:`qcodes.loops.Loop`
    and :class:`qcodes.measure.Measure` measurements - in which case
    these arrays are nested inside the loop's setpoint array. To use, provide a
    ``get`` method that returns an array or regularly-shaped sequence, and
    describe that array in ``super().__init__``.

- :class:`.MultiParameter` is the base class for multi-valued parameters.
    Currently not settable, only gettable, but can return an arbitrary
    collection of scalar and array values and can be used in
    :class:`qcodes.dataset.measurements.Measurement` as well as the
    legacy :class:`qcodes.loops.Loop` and :class:`qcodes.measure.Measure`
    measurements. To use, provide a ``get`` method
    that returns a sequence of values, and describe those values in
    ``super().__init__``.

"""

# TODO (alexcjohnson) update this with the real duck-typing requirements or
# create an ABC for Parameter and MultiParameter - or just remove this statement
# if everyone is happy to use these classes.

import collections
import enum
import logging
import os
import time
import warnings
from copy import copy
from datetime import datetime, timedelta
from functools import wraps
from operator import xor
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Sized,
    Tuple,
    Type,
    Union,
    cast,
)

import numpy
from typing_extensions import Protocol

from qcodes.data.data_array import DataArray
from qcodes.instrument.sweep_values import SweepFixedValues
from qcodes.utils.command import Command
from qcodes.utils.deprecate import deprecate, issue_deprecation_warning
from qcodes.utils.helpers import (
    DelegateAttributes,
    abstractmethod,
    full_class,
    is_sequence_of,
    named_repr,
    permissive_range,
    warn_units,
)
from qcodes.utils.metadata import Metadatable
from qcodes.utils.validators import Arrays, Enum, Ints, Strings, Validator

if TYPE_CHECKING:
    from .base import Instrument, InstrumentBase


# for now the type the parameter may contain is not restricted at all
ParamDataType = Any
ParamRawDataType = Any


log = logging.getLogger(__name__)


class _SetParamContext:
    """
    This class is returned by the ``set_to`` method of parameter

    Example usage:

    >>> v = dac.voltage()
    >>> with dac.voltage.set_to(-1):
        ...     # Do stuff with the DAC output set to -1 V.
        ...
    >>> assert abs(dac.voltage() - v) <= tolerance

    """
    def __init__(self, parameter: "_BaseParameter", value: ParamDataType,
                 allow_changes: bool = False):
        self._parameter = parameter
        self._value = value
        self._allow_changes = allow_changes
        self._original_value = None
        self._original_settable: Optional[bool] = None

    def __enter__(self) -> None:
        self._original_value = self._parameter.cache()

        if self._original_value != self._value:
            self._parameter.set(self._value)

        if not self._allow_changes:
            self._original_settable = self._parameter.settable
            self._parameter._settable = False  # type: ignore[has-type]

    def __exit__(self,
                 typ: Optional[Type[BaseException]],
                 value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        if not self._allow_changes:
            self._parameter._settable = (  # type: ignore[has-type]
                self._original_settable
            )

        if self._parameter.cache() != self._original_value:
            self._parameter.set(self._original_value)


def invert_val_mapping(val_mapping: Mapping[Any, Any]) -> Dict[Any, Any]:
    """Inverts the value mapping dictionary for allowed parameter values"""
    return {v: k for k, v in val_mapping.items()}


class _BaseParameter(Metadatable):
    """
    Shared behavior for all parameters. Not intended to be used
    directly, normally you should use ``Parameter``, ``ArrayParameter``,
    ``MultiParameter``, or ``CombinedParameter``.
    Note that ``CombinedParameter`` is not yet a subclass of ``_BaseParameter``

    Args:
        name: the local name of the parameter. Must be a valid
            identifier, ie no spaces or special characters or starting with a
            number. If this parameter is part of an Instrument or Station,
            this should match how it will be referenced from that parent,
            ie ``instrument.name`` or ``instrument.parameters[name]``

        instrument: the instrument this parameter
            belongs to, if any

        snapshot_get: False prevents any update to the
            parameter during a snapshot, even if the snapshot was called with
            ``update=True``, for example if it takes too long to update.
            Default True.

        snapshot_value: False prevents parameter value to be
            stored in the snapshot. Useful if the value is large.

        snapshot_exclude: True prevents parameter to be
            included in the snapshot. Useful if there are many of the same
            parameter which are clogging up the snapshot.
            Default False

        step: max increment of parameter value.
            Larger changes are broken into multiple steps this size.
            When combined with delays, this acts as a ramp.

        scale: Scale to multiply value with before
            performing set. the internally multiplied value is stored in
            ``cache.raw_value``. Can account for a voltage divider.

        offset: Compensate for a parameter specific offset. (just as scale)
            get value = raw value - offset.
            set value = argument + offset.
            If offset and scale are used in combination, when getting a value,
            first an offset is added, then the scale is applied.

        inter_delay: Minimum time (in seconds) between successive sets.
            If the previous set was less than this, it will wait until the
            condition is met. Can be set to 0 to go maximum speed with
            no errors.

        post_delay: time (in seconds) to wait after the *start* of each set,
            whether part of a sweep or not. Can be set to 0 to go maximum
            speed with no errors.

        val_mapping: A bidirectional map data/readable values to instrument
            codes, expressed as a dict: ``{data_val: instrument_code}``
            For example, if the instrument uses '0' to mean 1V and '1' to mean
            10V, set val_mapping={1: '0', 10: '1'} and on the user side you
            only see 1 and 10, never the coded '0' and '1'
            If vals is omitted, will also construct a matching Enum validator.
            NOTE: only applies to get if get_cmd is a string, and to set if
            set_cmd is a string.
            You can use ``val_mapping`` with ``get_parser``, in which case
            ``get_parser`` acts on the return value from the instrument first,
            then ``val_mapping`` is applied (in reverse).

        get_parser: Function to transform the response from get to the final
            output value. See also val_mapping

        set_parser: Function to transform the input set value to an encoded
            value sent to the instrument. See also val_mapping.

        vals: a Validator object for this parameter

        max_val_age: The max time (in seconds) to trust a saved value obtained
            from ``cache.get`` (or ``get_latest``). If this parameter has not
            been set or measured more recently than this, perform an
            additional measurement.

        metadata: extra information to include with the
            JSON snapshot of the parameter

        abstract: Specifies if this parameter is abstract or not. Default
            is False. If the parameter is 'abstract', it *must* be overridden
            by a non-abstract parameter before the instrument containing
            this parameter can be instantiated. We override a parameter by
            adding one with the same name and unit. An abstract parameter
            can be added in a base class and overridden in a subclass.

        bind_to_instrument: Should the parameter be registered as a delegate attribute
            on the instrument passed via the instrument argument.
    """

    def __init__(
        self,
        name: str,
        instrument: Optional["InstrumentBase"],
        snapshot_get: bool = True,
        metadata: Optional[Mapping[Any, Any]] = None,
        step: Optional[float] = None,
        scale: Optional[Union[float, Iterable[float]]] = None,
        offset: Optional[Union[float, Iterable[float]]] = None,
        inter_delay: float = 0,
        post_delay: float = 0,
        val_mapping: Optional[Mapping[Any, Any]] = None,
        get_parser: Optional[Callable[..., Any]] = None,
        set_parser: Optional[Callable[..., Any]] = None,
        snapshot_value: bool = True,
        snapshot_exclude: bool = False,
        max_val_age: Optional[float] = None,
        vals: Optional[Validator[Any]] = None,
        abstract: Optional[bool] = False,
        bind_to_instrument: bool = True,
    ) -> None:
        super().__init__(metadata)
        if not str(name).isidentifier():
            raise ValueError(f"Parameter name must be a valid identifier "
                             f"got {name} which is not. Parameter names "
                             f"cannot start with a number and "
                             f"must not contain spaces or special characters")
        self._short_name = str(name)
        self._instrument = instrument
        self._snapshot_get = snapshot_get
        self._snapshot_value = snapshot_value
        self.snapshot_exclude = snapshot_exclude

        if not isinstance(vals, (Validator, type(None))):
            raise TypeError('vals must be None or a Validator')
        elif val_mapping is not None:
            vals = Enum(*val_mapping.keys())
        self.vals = vals

        self.step = step
        self.scale = scale
        self.offset = offset

        self.inter_delay = inter_delay
        self.post_delay = post_delay

        self.val_mapping = val_mapping
        if val_mapping is None:
            self.inverse_val_mapping = None
        else:
            self.inverse_val_mapping = invert_val_mapping(val_mapping)

        self.get_parser = get_parser
        self.set_parser = set_parser

        # ``_Cache`` stores "latest" value (and raw value) and timestamp
        # when it was set or measured
        self.cache: _CacheProtocol = _Cache(self, max_val_age=max_val_age)
        # ``GetLatest`` is left from previous versions where it would
        # implement a subset of features which ``_Cache`` has.
        # It is left for now for backwards compatibility reasons and shall
        # be deprecated and removed in the future versions.
        self.get_latest: GetLatest
        self.get_latest = GetLatest(self)

        self.get: Callable[..., ParamDataType]
        implements_get_raw = (
            hasattr(self, 'get_raw')
            and not getattr(self.get_raw,
                            '__qcodes_is_abstract_method__', False)
        )
        self._gettable = False
        if implements_get_raw:
            self.get = self._wrap_get(self.get_raw)
            self._gettable = True
        elif hasattr(self, 'get'):
            raise RuntimeError(f'Overwriting get in a subclass of '
                               f'_BaseParameter: '
                               f'{self.full_name} is not allowed.')

        self.set: Callable[..., None]
        implements_set_raw = (
            hasattr(self, 'set_raw')
            and not getattr(self.set_raw,
                            '__qcodes_is_abstract_method__', False)
        )
        self._settable = False
        if implements_set_raw:
            self.set = self._wrap_set(self.set_raw)
            self._settable = True
        elif hasattr(self, 'set'):
            raise RuntimeError(f'Overwriting set in a subclass of '
                               f'_BaseParameter: '
                               f'{self.full_name} is not allowed.')

        # subclasses should extend this list with extra attributes they
        # want automatically included in the snapshot
        self._meta_attrs = ['name', 'instrument', 'step', 'scale', 'offset',
                            'inter_delay', 'post_delay', 'val_mapping', 'vals']

        # Specify time of last set operation, used when comparing to delay to
        # check if additional waiting time is needed before next set
        self._t_last_set = time.perf_counter()
        # should we call validate when getting data. default to False
        # intended to be changed in a subclass if you want the subclass
        # to perform a validation on get
        self._validate_on_get = False
        self._abstract = abstract

        if instrument is not None and bind_to_instrument:
            existing_parameter = instrument.parameters.get(name, None)

            if existing_parameter:

                if not existing_parameter.abstract:
                    raise KeyError(
                        f"Duplicate parameter name {name} on instrument {instrument}"
                    )

            instrument.parameters[name] = self

    @property
    def raw_value(self) -> ParamRawDataType:
        """
        Note that this property will be deprecated soon. Use
        ``cache.raw_value`` instead.

        Represents the cached raw value of the parameter.

        :getter: Returns the cached raw value of the parameter.
        """
        return self.cache.raw_value

    @abstractmethod
    def get_raw(self) -> ParamRawDataType:
        """
        ``get_raw`` is called to perform the actual data acquisition from the
        instrument. This method should either be overwritten to perform the
        desired operation or alternatively for :class:`.Parameter` a
        suitable method is automatically generated if ``get_cmd`` is supplied
        to the parameter constructor. The method is automatically wrapped to
        provide a ``get`` method on the parameter instance.
        """
        raise NotImplementedError

    @abstractmethod
    def set_raw(self, value: ParamRawDataType) -> None:
        """
        ``set_raw`` is called to perform the actual setting of a parameter on
        the instrument. This method should either be overwritten to perform the
        desired operation or alternatively for :class:`.Parameter` a
        suitable method is automatically generated if ``set_cmd`` is supplied
        to the parameter constructor. The method is automatically wrapped to
        provide a ``set`` method on the parameter instance.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """Include the instrument name with the Parameter name if possible."""
        inst_name = getattr(self._instrument, 'name', '')
        if inst_name:
            return f'{inst_name}_{self.name}'
        else:
            return self.name

    def __repr__(self) -> str:
        return named_repr(self)

    def __call__(self, *args: Any, **kwargs: Any) -> Optional[ParamDataType]:
        if len(args) == 0 and len(kwargs) == 0:
            if self.gettable:
                return self.get()
            else:
                raise NotImplementedError('no get cmd found in' +
                                          f' Parameter {self.name}')
        else:
            if self.settable:
                self.set(*args, **kwargs)
                return None
            else:
                raise NotImplementedError('no set cmd found in' +
                                          f' Parameter {self.name}')

    def snapshot_base(self, update: Optional[bool] = True,
                      params_to_skip_update: Optional[Sequence[str]] = None
                      ) -> Dict[Any, Any]:
        """
        State of the parameter as a JSON-compatible dict (everything that
        the custom JSON encoder class
        :class:`qcodes.utils.helpers.NumpyJSONEncoder` supports).

        If the parameter has been initiated with ``snapshot_value=False``,
        the snapshot will NOT include the ``value`` and ``raw_value`` of the
        parameter.

        Args:
            update: If True, update the state by calling ``parameter.get()``
                unless ``snapshot_get`` of the parameter is ``False``.
                If ``update`` is ``None``, use the current value from the
                ``cache`` unless the cache is invalid. If ``False``, never call
                ``parameter.get()``.
            params_to_skip_update: No effect but may be passed from superclass

        Returns:
            base snapshot
        """
        if self.snapshot_exclude:
            warnings.warn(
                f"Parameter ({self.name}) is used in the snapshot while it "
                f"should be excluded from the snapshot")

        state: Dict[str, Any] = {'__class__': full_class(self),
                                 'full_name': str(self)}

        if self._snapshot_value:
            has_get = self.gettable
            allowed_to_call_get_when_snapshotting = (self._snapshot_get
                                                     and update is not False)
            can_call_get_when_snapshotting = (
                    allowed_to_call_get_when_snapshotting and has_get)

            if can_call_get_when_snapshotting and update:
                state['value'] = self.get()
            else:
                state['value'] = self.cache.get(
                    get_if_invalid=can_call_get_when_snapshotting)

            state['raw_value'] = self.cache.raw_value

        state['ts'] = self.cache.timestamp

        if isinstance(state['ts'], datetime):
            dttime: datetime = state['ts']
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

    @property
    def snapshot_value(self) -> bool:
        """
        If True the value of the parameter will be included in the snapshot.
        """
        return self._snapshot_value

    def _from_value_to_raw_value(self, value: ParamDataType
                                 ) -> ParamRawDataType:
        raw_value: ParamRawDataType

        if self.val_mapping is not None:
            # Convert set values using val_mapping dictionary
            raw_value = self.val_mapping[value]
        else:
            raw_value = value

        # transverse transformation in reverse order as compared to
        # getter: apply scale first
        if self.scale is not None:
            if isinstance(self.scale, collections.abc.Iterable):
                # Scale contains multiple elements, one for each value
                raw_value = tuple(val * scale for val, scale
                                  in zip(raw_value, self.scale))
            else:
                # Use single scale for all values
                raw_value = raw_value * self.scale

        # apply offset next
        if self.offset is not None:
            if isinstance(self.offset, collections.abc.Iterable):
                # offset contains multiple elements, one for each value
                raw_value = tuple(val + offset for val, offset
                                  in zip(raw_value, self.offset))
            else:
                # Use single offset for all values
                raw_value = raw_value + self.offset

        # parser last
        if self.set_parser is not None:
            raw_value = self.set_parser(raw_value)

        return raw_value

    def _from_raw_value_to_value(self, raw_value: ParamRawDataType
                                 ) -> ParamDataType:
        value: ParamDataType

        if self.get_parser is not None:
            value = self.get_parser(raw_value)
        else:
            value = raw_value

        # apply offset first (native scale)
        if self.offset is not None and value is not None:
            # offset values
            try:
                value = value - self.offset
            except TypeError:
                if isinstance(self.offset, collections.abc.Iterable):
                    # offset contains multiple elements, one for each value
                    value = tuple(val - offset for val, offset
                                  in zip(value, self.offset))
                elif isinstance(value, collections.abc.Iterable):
                    # Use single offset for all values
                    value = tuple(val - self.offset for val in value)
                else:
                    raise

        # scale second
        if self.scale is not None and value is not None:
            # Scale values
            try:
                value = value / self.scale
            except TypeError:
                if isinstance(self.scale, collections.abc.Iterable):
                    # Scale contains multiple elements, one for each value
                    value = tuple(val / scale for val, scale in zip(value,
                                                                    self.scale))
                elif isinstance(value, collections.abc.Iterable):
                    # Use single scale for all values
                    value = tuple(val / self.scale for val in value)
                else:
                    raise

        if self.inverse_val_mapping is not None:
            if value in self.inverse_val_mapping:
                value = self.inverse_val_mapping[value]
            else:
                try:
                    value = self.inverse_val_mapping[int(value)]
                except (ValueError, KeyError):
                    raise KeyError(f"'{value}' not in val_mapping")

        return value

    def _wrap_get(self, get_function: Callable[..., ParamDataType]) ->\
            Callable[..., ParamDataType]:
        @wraps(get_function)
        def get_wrapper(*args: Any, **kwargs: Any) -> ParamDataType:
            if not self.gettable:
                raise TypeError("Trying to get a parameter"
                                " that is not gettable.")
            if self.abstract:
                raise NotImplementedError(
                    f"Trying to get an abstract parameter: {self.full_name}"
                )
            try:
                # There might be cases where a .get also has args/kwargs
                raw_value = get_function(*args, **kwargs)

                value = self._from_raw_value_to_value(raw_value)

                if self._validate_on_get:
                    self.validate(value)

                self.cache._update_with(value=value, raw_value=raw_value)

                return value

            except Exception as e:
                e.args = e.args + (f'getting {self}',)
                raise e

        return get_wrapper

    def _wrap_set(self, set_function: Callable[..., None]) -> \
            Callable[..., None]:
        @wraps(set_function)
        def set_wrapper(value: ParamDataType, **kwargs: Any) -> None:
            try:
                if not self.settable:
                    raise TypeError("Trying to set a parameter"
                                    " that is not settable.")
                if self.abstract:
                    raise NotImplementedError(
                        f"Trying to set an abstract parameter: {self.full_name}"
                    )
                self.validate(value)

                # In some cases intermediate sweep values must be used.
                # Unless `self.step` is defined, get_sweep_values will return
                # a list containing only `value`.
                steps = self.get_ramp_values(value, step=self.step)

                for step_index, val_step in enumerate(steps):
                    # even if the final value is valid we may be generating
                    # steps that are not so validate them too
                    self.validate(val_step)

                    raw_val_step = self._from_value_to_raw_value(val_step)

                    # Check if delay between set operations is required
                    t_elapsed = time.perf_counter() - self._t_last_set
                    if t_elapsed < self.inter_delay:
                        # Sleep until time since last set is larger than
                        # self.inter_delay
                        time.sleep(self.inter_delay - t_elapsed)

                    # Start timer to measure execution time of set_function
                    t0 = time.perf_counter()

                    set_function(raw_val_step, **kwargs)

                    # Update last set time (used for calculating delays)
                    self._t_last_set = time.perf_counter()

                    # Check if any delay after setting is required
                    t_elapsed = self._t_last_set - t0
                    if t_elapsed < self.post_delay:
                        # Sleep until total time is larger than self.post_delay
                        time.sleep(self.post_delay - t_elapsed)

                    self.cache._update_with(value=val_step,
                                            raw_value=raw_val_step)

            except Exception as e:
                e.args = e.args + (f'setting {self} to {value}',)
                raise e

        return set_wrapper

    def get_ramp_values(self, value: Union[float, Sized],
                        step: Optional[float] = None
                        ) -> Sequence[Union[float, Sized]]:
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
                raise RuntimeError("Don't know how to step a parameter"
                                   " with more than one value")
            if self.get_latest() is None:
                self.get()
            start_value = self.get_latest()
            if not (isinstance(start_value, (int, float)) and
                    isinstance(value, (int, float))):
                # parameter is numeric but either one of the endpoints
                # is not or the starting point is unknown. The later
                # can happen for a non gettable parameter in the initial set
                # operation.
                log.warning(f'cannot sweep {self.name} from {start_value!r} '
                            f'to {value!r} - jumping.')
                return [value]

            # drop the initial value, we're already there
            return permissive_range(start_value, value, step)[1:] + [value]

    def validate(self, value: ParamDataType) -> None:
        """
        Validate the value supplied.

        Args:
            value: value to validate

        Raises:
            TypeError: If the value is of the wrong type.
            ValueError: If the value is outside the bounds specified by the
               validator.
        """
        if self._instrument:
            context = (getattr(self._instrument, 'name', '') or
                       str(self._instrument.__class__)) + '.' + self.name
        else:
            context = self.name
        if self.vals is not None:
            self.vals.validate(value, 'Parameter: ' + context)

    @property
    def step(self) -> Optional[float]:
        """
        Stepsize that this Parameter uses during set operation.
        Stepsize must be a positive number or None.
        If step is a positive number, this is the maximum value change
        allowed in one hardware call, so a single set can result in many
        calls to the hardware if the starting value is far from the target.
        All but the final change will attempt to change by +/- step exactly.
        If step is None stepping will not be used.

        :getter: Returns the current stepsize.
        :setter: Sets the value of the step.

        Raises:
            TypeError: if step is set to not numeric or None
            ValueError: if step is set to negative
            TypeError:  if step is set to not integer or None for an
                integer parameter
            TypeError: if step is set to not a number on None
        """
        return self._step

    @step.setter
    def step(self, step: Optional[float]) -> None:
        if step is None:
            self._step: Optional[float] = step
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
    def post_delay(self) -> float:
        """
        Delay time after *start* of set operation, for each set.
        The actual time will not be shorter than this, but may be longer
        if the underlying set call takes longer.

        Typically used in conjunction with `step` to create an effective
        ramp rate, but can also be used without a `step` to enforce a delay
        *after* every set. One might think of post_delay as how long a set
        operation is supposed to take. For example, there might be an
        instrument that needs extra time after setting a parameter although
        the command for setting the parameter returns quickly.

        :getter: Returns the current post_delay.
        :setter: Sets the value of the post_delay.

        Raises:
            TypeError: If delay is not int nor float
            ValueError: If delay is negative
        """
        return self._post_delay

    @post_delay.setter
    def post_delay(self, post_delay: float) -> None:
        if not isinstance(post_delay, (int, float)):
            raise TypeError(
                f'post_delay ({post_delay}) must be a number')
        if post_delay < 0:
            raise ValueError(
                f'post_delay ({post_delay}) must not be negative')
        self._post_delay = post_delay

    @property
    def inter_delay(self) -> float:
        """
        Delay time between consecutive set operations.
        The actual time will not be shorter than this, but may be longer
        if the underlying set call takes longer.

        Typically used in conjunction with `step` to create an effective
        ramp rate, but can also be used without a `step` to enforce a delay
        *between* sets.

        :getter: Returns the current inter_delay.
        :setter: Sets the value of the inter_delay.

        Raises:
            TypeError: If delay is not int nor float
            ValueError: If delay is negative
        """
        return self._inter_delay

    @inter_delay.setter
    def inter_delay(self, inter_delay: float) -> None:
        if not isinstance(inter_delay, (int, float)):
            raise TypeError(
                f'inter_delay ({inter_delay}) must be a number')
        if inter_delay < 0:
            raise ValueError(
                f'inter_delay ({inter_delay}) must not be negative')
        self._inter_delay = inter_delay

    @property
    def name(self) -> str:
        """Name of the parameter. This is identical to :meth:`short_name`."""
        return self._short_name

    @property
    def short_name(self) -> str:
        """Short name of the parameter. This is without the name of the
        instrument or submodule that the parameter may be bound to. For
        full name refer to :meth:`full_name`."""
        return self._short_name

    @property
    def full_name(self) -> str:
        """
        Name of the parameter including the name of the instrument and
        submodule that the parameter may be bound to. The names are separated
        by underscores, like this: ``instrument_submodule_parameter``.
        """
        return "_".join(self.name_parts)

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

    def set_to(self, value: ParamDataType,
               allow_changes: bool = False) -> _SetParamContext:
        """
        Use a context manager to temporarily set a parameter to a value. By
        default, the parameter value cannot be changed inside the context.
        This may be overridden with ``allow_changes=True``.

        Examples:

            >>> from qcodes import Parameter
            >>> p = Parameter("p", set_cmd=None, get_cmd=None)
            >>> p.set(2)
            >>> with p.set_to(3):
            ...     print(f"p value in with block {p.get()}")  # prints 3
            ...     p.set(5)  # raises an exception
            >>> print(f"p value outside with block {p.get()}")  # prints 2
            >>> with p.set_to(3, allow_changes=True):
            ...     p.set(5)  # now this works
            >>> print(f"value after second block: {p.get()}")  # still prints 2
        """
        context_manager = _SetParamContext(self, value,
                                           allow_changes=allow_changes)
        return context_manager

    def restore_at_exit(self, allow_changes: bool = True) -> _SetParamContext:
        """
        Use a context manager to restore the value of a parameter after a
        ``with`` block.

        By default, the parameter value may be changed inside the block, but
        this can be prevented with ``allow_changes=False``. This can be
        useful, for example, for debugging a complex measurement that
        unintentionally modifies a parameter.

        Example:

            >>> p = Parameter("p", set_cmd=None, get_cmd=None)
            >>> p.set(2)
            >>> with p.restore_at_exit():
            ...     p.set(3)
            ...     print(f"value inside with block: {p.get()}")  # prints 3
            >>> print(f"value after with block: {p.get()}")  # prints 2
            >>> with p.restore_at_exit(allow_changes=False):
            ...     p.set(5)  # raises an exception
        """
        return self.set_to(self.cache(), allow_changes=allow_changes)

    @property
    def name_parts(self) -> List[str]:
        """
        List of the parts that make up the full name of this parameter
        """
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

    @property
    def gettable(self) -> bool:
        """
        Is it allowed to call get on this parameter?
        """
        return self._gettable

    @property
    def settable(self) -> bool:
        """
        Is it allowed to call set on this parameter?
        """
        return self._settable

    @property
    def underlying_instrument(self) -> Optional['InstrumentBase']:
        """
        Returns an instance of the underlying hardware instrument that this
        parameter communicates with, per this parameter's implementation.

        This is useful in the case where a parameter does not belongs to
        an instrument instance that represents a real hardware instrument
        but actually uses a real hardware instrument in its implementation
        (e.g. via calls to one or more parameters of that real hardware
        instrument). This is also useful when a parameter does belong to
        an instrument instance but that instance does not represent the
        real hardware instrument that the parameter interacts with: hence
        ``root_instrument`` of the parameter cannot be the
        ``hardware_instrument``, however ``underlying_instrument`` can be
        implemented to return the ``hardware_instrument``.

        By default it returns the ``root_instrument`` of the parameter.
        """
        return self.root_instrument

    @property
    def abstract(self) -> Optional[bool]:
        return self._abstract


class Parameter(_BaseParameter):
    """
    A parameter represents a single degree of freedom. Most often,
    this is the standard parameter for Instruments, though it can also be
    used as a variable, i.e. storing/retrieving a value, or be subclassed for
    more complex uses.

    By default only gettable, returning its last value.
    This behaviour can be modified in two ways:

    1. Providing a ``get_cmd``/``set_cmd``, which can do the following:

       a. callable, with zero args for get_cmd, one arg for set_cmd
       b. VISA command string
       c. None, in which case it retrieves its last value for ``get_cmd``,
          and stores a value for ``set_cmd``
       d. False, in which case trying to get/set will raise an error.

    2. Creating a subclass with an explicit :meth:`get_raw`/:meth:`set_raw`
       method.

       This enables more advanced functionality. The :meth:`get_raw` and
       :meth:`set_raw` methods are automatically wrapped to provide ``get`` and
       ``set``.

    It is an error to do both 1 and 2. E.g supply a ``get_cmd``/``set_cmd``
    and implement ``get_raw``/``set_raw``


    To detect if a parameter is gettable or settable check the attributes
    :py:attr:`~gettable` and :py:attr:`~settable` on the parameter.

    Parameters have a ``cache`` object that stores internally the current
    ``value`` and ``raw_value`` of the parameter. Calling ``cache.get()``
    (or ``cache()``) simply returns the most recent set or measured value of
    the parameter.

    Parameter also has a ``.get_latest`` method that duplicates the behavior
    of ``cache()`` call, as in, it also simply returns the most recent set
    or measured value.

    Args:
        name: The local name of the parameter. Should be a valid
            identifier, ie no spaces or special characters. If this parameter
            is part of an Instrument or Station, this is how it will be
            referenced from that parent, ie ``instrument.name`` or
            ``instrument.parameters[name]``.

        instrument: The instrument this parameter
            belongs to, if any.

        label: Normally used as the axis label when this
            parameter is graphed, along with ``unit``.

        unit: The unit of measure. Use ``''`` for unitless.

        snapshot_get: ``False`` prevents any update to the
            parameter during a snapshot, even if the snapshot was called with
            ``update=True``, for example, if it takes too long to update,
            or if the parameter is only meant for measurements hence calling
            get on it during snapshot may be an error. Default True.

        snapshot_value: ``False`` prevents parameter value to be
            stored in the snapshot. Useful if the value is large.

        snapshot_exclude: ``True`` prevents parameter to be
            included in the snapshot. Useful if there are many of the same
            parameter which are clogging up the snapshot.
            Default ``False``.

        step: Max increment of parameter value.
            Larger changes are broken into multiple steps this size.
            When combined with delays, this acts as a ramp.

        scale: Scale to multiply value with before
            performing set. the internally multiplied value is stored in
            ``cache.raw_value``. Can account for a voltage divider.

        inter_delay: Minimum time (in seconds)
            between successive sets. If the previous set was less than this,
            it will wait until the condition is met.
            Can be set to 0 to go maximum speed with no errors.

        post_delay: Time (in seconds) to wait
            after the *start* of each set, whether part of a sweep or not.
            Can be set to 0 to go maximum speed with no errors.

        val_mapping: A bi-directional map data/readable values
            to instrument codes, expressed as a dict:
            ``{data_val: instrument_code}``
            For example, if the instrument uses '0' to mean 1V and '1' to mean
            10V, set val_mapping={1: '0', 10: '1'} and on the user side you
            only see 1 and 10, never the coded '0' and '1'
            If vals is omitted, will also construct a matching Enum validator.
            **NOTE** only applies to get if get_cmd is a string, and to set if
            set_cmd is a string.
            You can use ``val_mapping`` with ``get_parser``, in which case
            ``get_parser`` acts on the return value from the instrument first,
            then ``val_mapping`` is applied (in reverse).

        get_parser: Function to transform the response
            from get to the final output value. See also `val_mapping`.

        set_parser: Function to transform the input set
            value to an encoded value sent to the instrument.
            See also `val_mapping`.

        vals: Allowed values for setting this parameter.
            Only relevant if settable. Defaults to ``Numbers()``.

        max_val_age: The max time (in seconds) to trust a
            saved value obtained from ``cache()`` (or ``cache.get()``, or
            ``get_latest()``. If this parameter has not been set or measured
            more recently than this, perform an additional measurement.

        initial_value: Value to set the parameter to at the end of its
            initialization (this is equivalent to calling
            ``parameter.set(initial_value)`` after parameter initialization).
            Cannot be passed together with ``initial_cache_value`` argument.

        initial_cache_value: Value to set the cache of the parameter to
            at the end of its initialization (this is equivalent to calling
            ``parameter.cache.set(initial_cache_value)`` after parameter
            initialization). Cannot be passed together with ``initial_value``
            argument.

        docstring: Documentation string for the ``__doc__``
            field of the object. The ``__doc__``  field of the instance is
            used by some help systems, but not all.

        metadata: Extra information to include with the
            JSON snapshot of the parameter.

        abstract: Specifies if this parameter is abstract or not. Default
            is False. If the parameter is 'abstract', it *must* be overridden
            by a non-abstract parameter before the instrument containing
            this parameter can be instantiated. We override a parameter by
            adding one with the same name and unit. An abstract parameter
            can be added in a base class and overridden in a subclass.

        bind_to_instrument: Should the parameter be registered as a delegate attribute
            on the instrument passed via the instrument argument.
    """

    def __init__(
        self,
        name: str,
        instrument: Optional["InstrumentBase"] = None,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        get_cmd: Optional[Union[str, Callable[..., Any], bool]] = None,
        set_cmd: Optional[Union[str, Callable[..., Any], bool]] = False,
        initial_value: Optional[Union[float, str]] = None,
        max_val_age: Optional[float] = None,
        vals: Optional[Validator[Any]] = None,
        docstring: Optional[str] = None,
        initial_cache_value: Optional[Union[float, str]] = None,
        bind_to_instrument: bool = True,
        **kwargs: Any,
    ) -> None:
        if instrument is not None and bind_to_instrument:
            existing_parameter = instrument.parameters.get(name, None)

            if existing_parameter:

                # this check is redundant since its also in the baseclass
                # but if we do not put it here it would be an api break
                # as parameter duplication check won't be done first,
                # hence for parameters that are duplicates and have
                # wrong units, users will be getting ValueError where
                # they used to have KeyError before.
                if not existing_parameter.abstract:
                    raise KeyError(
                        f"Duplicate parameter name {name} on instrument {instrument}"
                    )

                existing_unit = getattr(existing_parameter, "unit", None)
                if existing_unit != unit:
                    raise ValueError(
                        f"The unit of the parameter '{name}' is '{unit}'. "
                        f"This is inconsistent with the unit defined in the "
                        f"base class"
                    )

        super().__init__(
            name=name,
            instrument=instrument,
            vals=vals,
            max_val_age=max_val_age,
            bind_to_instrument=bind_to_instrument,
            **kwargs,
        )

        no_instrument_get = not self.gettable and \
            (get_cmd is None or get_cmd is False)
        # TODO: a matching check should be in _BaseParameter but
        #   due to the current limited design the _BaseParameter cannot
        #   know if this subclass will supply a get_cmd
        #   To work around this a RunTime check is put into get of GetLatest
        #   and into get of _Cache
        if max_val_age is not None and no_instrument_get:
            raise SyntaxError('Must have get method or specify get_cmd '
                              'when max_val_age is set')

        # Enable set/get methods from get_cmd/set_cmd if given and
        # no `get`/`set` or `get_raw`/`set_raw` methods have been defined
        # in the scope of this class.
        # (previous call to `super().__init__` wraps existing
        # get_raw/set_raw into get/set methods)
        if self.gettable and get_cmd not in (None, False):
            raise TypeError("Supplying a not None or False `get_cmd` to a Parameter"
                            " that already implements"
                            " get_raw is an error.")
        elif not self.gettable and get_cmd is not False:
            if get_cmd is None:
                self.get_raw = (  # type: ignore[assignment]
                    lambda: self.cache.raw_value)
            else:
                exec_str_ask = getattr(instrument, "ask", None) \
                    if instrument else None
                self.get_raw = Command(arg_count=0,  # type: ignore[assignment]
                                       cmd=get_cmd,
                                       exec_str=exec_str_ask)
            self._gettable = True
            self.get = self._wrap_get(self.get_raw)

        if self.settable and set_cmd not in (None, False):
            raise TypeError("Supplying a not None or False `set_cmd` to a Parameter"
                            " that already implements"
                            " set_raw is an error.")
        elif not self.settable and set_cmd is not False:
            if set_cmd is None:
                self.set_raw: Callable[..., Any] = lambda x: x
            else:
                exec_str_write = getattr(instrument, "write", None) \
                    if instrument else None
                self.set_raw = Command(arg_count=1, cmd=set_cmd,
                                       exec_str=exec_str_write)
            self._settable = True
            self.set = self._wrap_set(self.set_raw)

        self._meta_attrs.extend(['label', 'unit', 'vals'])

        #: Label of the data used for plots etc.
        self.label: str = name if label is None else label
        #: The unit of measure. Use ``''`` for unitless.
        self.unit = unit if unit is not None else ''

        if initial_value is not None and initial_cache_value is not None:
            raise SyntaxError('It is not possible to specify both of the '
                              '`initial_value` and `initial_cache_value` '
                              'keyword arguments.')

        if initial_value is not None:
            self.set(initial_value)

        if initial_cache_value is not None:
            self.cache.set(initial_cache_value)

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

    def __getitem__(self, keys: Any) -> 'SweepFixedValues':
        """
        Slice a Parameter to get a SweepValues object
        to iterate over during a sweep
        """
        return SweepFixedValues(self, keys)

    def increment(self, value: ParamDataType) -> None:
        """ Increment the parameter with a value

        Args:
            value: Value to be added to the parameter.
        """
        self.set(self.get() + value)

    def sweep(self, start: float, stop: float,
              step: Optional[float] = None,
              num: Optional[int] = None) -> SweepFixedValues:
        """
        Create a collection of parameter values to be iterated over.
        Requires `start` and `stop` and (`step` or `num`)
        The sign of `step` is not relevant.

        Args:
            start: The starting value of the sequence.
            stop: The end value of the sequence.
            step:  Spacing between values.
            num: Number of values to generate.

        Returns:
            SweepFixedValues: Collection of parameter values to be
            iterated over.

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


class ParameterWithSetpoints(Parameter):
    """
    A parameter that has associated setpoints. The setpoints is nothing
    more than a list of other parameters that describe the values, names
    and units of the setpoint axis for this parameter.

    In most cases this will probably be a parameter that returns an array.
    It is expected that the setpoint arrays are 1D arrays such that the
    combined shape of the parameter e.g. if parameter is of shape (m,n)
    `setpoints` is a list of parameters of shape (m,) and (n,)

    In all other ways this is identical to  :class:`Parameter`. See the
    documentation of :class:`Parameter` for more details.
    """

    def __init__(self, name: str, *,
                 vals: Optional[Validator[Any]] = None,
                 setpoints: Optional[Sequence[_BaseParameter]] = None,
                 snapshot_get: bool = False,
                 snapshot_value: bool = False,
                 **kwargs: Any) -> None:

        if not isinstance(vals, Arrays):
            raise ValueError(f"A ParameterWithSetpoints must have an Arrays "
                             f"validator got {type(vals)}")
        if vals.shape_unevaluated is None:
            raise RuntimeError("A ParameterWithSetpoints must have a shape "
                               "defined for its validator.")

        super().__init__(name=name, vals=vals, snapshot_get=snapshot_get,
                         snapshot_value=snapshot_value, **kwargs)
        if setpoints is None:
            self.setpoints: Sequence[_BaseParameter] = []
        else:
            self.setpoints = setpoints

        self._validate_on_get = True

    @property
    def setpoints(self) -> Sequence[_BaseParameter]:
        """
        Sequence of parameters to use as setpoints for this parameter.

        :getter: Returns a list of parameters currently used for setpoints.
        :setter: Sets the parameters to be used as setpoints from a sequence.
            The combined shape of the parameters supplied must be consistent
            with the data shape of the data returned from get on the parameter.
        """
        return self._setpoints

    @setpoints.setter
    def setpoints(self, setpoints: Sequence[_BaseParameter]) -> None:
        for setpointarray in setpoints:
            if not isinstance(setpointarray, Parameter):
                raise TypeError(f"Setpoints is of type {type(setpointarray)}"
                                f" expcected a QCoDeS parameter")
        self._setpoints = setpoints

    def validate_consistent_shape(self) -> None:
        """
        Verifies that the shape of the Array Validator of the parameter
        is consistent with the Validator of the Setpoints. This requires that
        both the setpoints and the actual parameters have validators
        of type Arrays with a defined shape.
        """

        if not isinstance(self.vals, Arrays):
            raise ValueError(f"Can only validate shapes for parameters "
                             f"with Arrays validator. {self.name} does "
                             f"not have an Arrays validator.")
        output_shape = self.vals.shape_unevaluated
        setpoints_shape_list: List[Optional[Union[int, Callable[[], int]]]] = []
        for sp in self.setpoints:
            if not isinstance(sp.vals, Arrays):
                raise ValueError(f"Can only validate shapes for parameters "
                                 f"with Arrays validator. {sp.name} is "
                                 f"a setpoint vector but does not have an "
                                 f"Arrays validator")
            if sp.vals.shape_unevaluated is not None:
                setpoints_shape_list.extend(sp.vals.shape_unevaluated)
            else:
                setpoints_shape_list.append(sp.vals.shape_unevaluated)
        setpoints_shape = tuple(setpoints_shape_list)

        if output_shape is None:
            raise ValueError(f"Trying to validate shape but parameter "
                             f"{self.name} does not define a shape")
        if None in output_shape or None in setpoints_shape:
            raise ValueError(f"One or more dimensions have unknown shape "
                             f"when comparing output: {output_shape} to "
                             f"setpoints: {setpoints_shape}")

        if output_shape != setpoints_shape:
            raise ValueError(f"Shape of output is not consistent with "
                             f"setpoints. Output is shape {output_shape} and "
                             f"setpoints are shape {setpoints_shape}")
        log.info(f"For parameter {self.full_name} verified "
                 f"that {output_shape} matches {setpoints_shape}")

    def validate(self, value: ParamDataType) -> None:
        """
        Overwrites the standard ``validate`` method to also check the the
        parameter has consistent shape with its setpoints. This only makes
        sense if the parameter has an Arrays
        validator

        Arguments are passed to the super method
        """
        if isinstance(self.vals, Arrays):
            self.validate_consistent_shape()
        super().validate(value)


class DelegateParameter(Parameter):
    """
    The :class:`.DelegateParameter` wraps a given `source` :class:`Parameter`.
    Setting/getting it results in a set/get of the source parameter with
    the provided arguments.

    The reason for using a :class:`DelegateParameter` instead of the
    source parameter is to provide all the functionality of the Parameter
    base class without overwriting properties of the source: for example to
    set a different scaling factor and unit on the :class:`.DelegateParameter`
    without changing those in the source parameter.

    The :class:`DelegateParameter` supports changing the `source`
    :class:`Parameter`. :py:attr:`~gettable`, :py:attr:`~settable` and
    :py:attr:`snapshot_value` properties automatically follow the source
    parameter. If source is set to ``None`` :py:attr:`~gettable` and
    :py:attr:`~settable` will always be ``False``. It is therefore an error
    to call get and set on a :class:`DelegateParameter` without a `source`.
    Note that a parameter without a source can be snapshotted correctly.

    :py:attr:`.unit` and :py:attr:`.label` can either be set when constructing
    a :class:`DelegateParameter` or inherited from the source
    :class:`Parameter`. If inherited they will automatically change when
    changing the source. Otherwise they will remain fixed.

    Note:
        DelegateParameter only supports mappings between the
        :class:`.DelegateParameter` and :class:`.Parameter` that are invertible
        (e.g. a bijection). It is therefor not allowed to create a
        :class:`.DelegateParameter` that performs non invertible
        transforms in its ``get_raw`` method.

        A DelegateParameter is not registered on the instrument by default.
        You should pass ``bind_to_instrument=True`` if you want this to
        be the case.
    """

    class _DelegateCache:
        def __init__(self,
                     parameter: 'DelegateParameter'):
            self._parameter = parameter
            self._marked_valid: bool = False

        @property
        def raw_value(self) -> ParamRawDataType:
            """
            raw_value is an attribute that surfaces the raw value from the
            cache. In the case of a :class:`DelegateParameter` it reflects
            the value of the cache of the source.

            Strictly speaking it should represent that value independent of
            its validity according to the `max_val_age` but in fact it does
            lose its validity when the maximum value age has been reached.
            This bug will not be fixed since the `raw_value` property will be
            removed soon.
            """
            if self._parameter.source is None:
                raise TypeError("Cannot get the raw value of a "
                                "DelegateParameter that delegates to None")
            return self._parameter.source.cache.get(get_if_invalid=False)

        @property
        def max_val_age(self) -> Optional[float]:
            if self._parameter.source is None:
                return None
            return self._parameter.source.cache.max_val_age

        @property
        def timestamp(self) -> Optional[datetime]:
            if self._parameter.source is None:
                return None
            return self._parameter.source.cache.timestamp

        @property
        def valid(self) -> bool:
            if self._parameter.source is None:
                return False
            source_cache = self._parameter.source.cache
            return source_cache.valid

        def invalidate(self) -> None:
            if self._parameter.source is not None:
                self._parameter.source.cache.invalidate()

        def get(self, get_if_invalid: bool = True) -> ParamDataType:
            if self._parameter.source is None:
                raise TypeError("Cannot get the cache of a "
                                "DelegateParameter that delegates to None")
            return self._parameter._from_raw_value_to_value(
                self._parameter.source.cache.get(get_if_invalid=get_if_invalid))

        def set(self, value: ParamDataType) -> None:
            if self._parameter.source is None:
                raise TypeError("Cannot set the cache of a DelegateParameter "
                                "that delegates to None")
            self._parameter.validate(value)
            self._parameter.source.cache.set(
                self._parameter._from_value_to_raw_value(value))

        def _set_from_raw_value(self, value: ParamRawDataType) -> None:
            if self._parameter.source is None:
                raise TypeError("Cannot set the cache of a DelegateParameter "
                                "that delegates to None")
            self._parameter.source.cache.set(value)

        def _update_with(self, *,
                         value: ParamDataType,
                         raw_value: ParamRawDataType,
                         timestamp: Optional[datetime] = None
                         ) -> None:
            """
            This method is needed for interface consistency with ``._Cache``
            because it is used by ``_BaseParameter`` in
            ``_wrap_get``/``_wrap_set``. Due to the fact that the source
            parameter already maintains it's own cache and the cache of the
            delegate parameter mirrors the cache of the source parameter by
            design, this method is just a noop.
            """
            pass

        def __call__(self) -> ParamDataType:
            return self.get(get_if_invalid=True)

    def __init__(
        self,
        name: str,
        source: Optional[Parameter],
        *args: Any,
        **kwargs: Any,
    ):
        if "bind_to_instrument" not in kwargs.keys():
            kwargs["bind_to_instrument"] = False

        self._attr_inherit = {"label": {"fixed": False,
                                        "value_when_without_source": name},
                              "unit": {"fixed": False,
                                       "value_when_without_source": ""}}

        for attr, attr_props in self._attr_inherit.items():
            if attr in kwargs:
                attr_props["fixed"] = True
            else:
                attr_props["fixed"] = False
            source_attr = getattr(source, attr,
                                  attr_props["value_when_without_source"])
            kwargs[attr] = kwargs.get(attr, source_attr)

        for cmd in ('set_cmd', 'get_cmd'):
            if cmd in kwargs:
                raise KeyError(f'It is not allowed to set "{cmd}" of a '
                               f'DelegateParameter because the one of the '
                               f'source parameter is supposed to be used.')
        if source is None and ("initial_cache_value" in kwargs
                               or "initial_value" in kwargs):
            raise KeyError("It is not allowed to supply 'initial_value'"
                           " or 'initial_cache_value' "
                           "without a source.")

        initial_cache_value = kwargs.pop("initial_cache_value", None)
        self.source = source
        super().__init__(name, *args, **kwargs)
        # explicitly set the source properties as
        # init will overwrite the ones set when assigning source
        self._set_properties_from_source(source)

        self.cache = self._DelegateCache(self)
        if initial_cache_value is not None:
            self.cache.set(initial_cache_value)

    @property
    def source(self) -> Optional[Parameter]:
        """
        The source parameter that this :class:`DelegateParameter` is bound to
        or ``None`` if this  :class:`DelegateParameter` is unbound.

        :getter: Returns the current source.
        :setter: Sets the source.
        """
        return self._source

    @source.setter
    def source(self, source: Optional[Parameter]) -> None:
        self._set_properties_from_source(source)
        self._source: Optional[Parameter] = source

    def _set_properties_from_source(self, source: Optional[Parameter]) -> None:
        if source is None:
            self._gettable = False
            self._settable = False
            self._snapshot_value = False
        else:
            self._gettable = source.gettable
            self._settable = source.settable
            self._snapshot_value = source._snapshot_value

        for attr, attr_props in self._attr_inherit.items():
            if not attr_props["fixed"]:
                attr_val = getattr(source,
                                   attr,
                                   attr_props["value_when_without_source"])
                setattr(self, attr, attr_val)

    # pylint: disable=method-hidden
    def get_raw(self) -> Any:
        if self.source is None:
            raise TypeError("Cannot get the value of a DelegateParameter "
                            "that delegates to a None source.")
        return self.source.get()

    # pylint: disable=method-hidden
    def set_raw(self, value: Any) -> None:
        if self.source is None:
            raise TypeError("Cannot set the value of a DelegateParameter "
                            "that delegates to a None source.")
        self.source(value)

    def snapshot_base(self, update: Optional[bool] = True,
                      params_to_skip_update: Optional[Sequence[str]] = None
                      ) -> Dict[Any, Any]:
        snapshot = super().snapshot_base(
            update=update,
            params_to_skip_update=params_to_skip_update
        )
        source_parameter_snapshot = None if self.source is None \
            else self.source.snapshot(update=update)
        snapshot.update(
            {'source_parameter': source_parameter_snapshot}
        )
        return snapshot


class ArrayParameter(_BaseParameter):
    """
    A gettable parameter that returns an array of values.
    Not necessarily part of an instrument.

    For new driver we strongly recommend using
    :class:`.ParameterWithSetpoints` which is both more flexible and
    significantly easier to use

    Subclasses should define a ``.get_raw`` method, which returns an array.
    This method is automatically wrapped to provide a ``.get`` method.

    :class:`.ArrayParameter` can be used in both a
    :class:`qcodes.dataset.measurements.Measurement`
    as well as in the legacy :class:`qcodes.loops.Loop`
    and :class:`qcodes.measure.Measure` measurements

    When used in a ``Loop`` or ``Measure`` operation, this will be entered
    into a single ``DataArray``, with extra dimensions added by the ``Loop``.
    The constructor args describe the array we expect from each ``.get`` call
    and how it should be handled.

    For now you must specify upfront the array shape, and this cannot change
    from one call to the next. Later we intend to require only that you specify
    the dimension, and the size of each dimension can vary from call to call.

    Args:
        name: The local name of the parameter. Should be a valid
            identifier, i.e. no spaces or special characters. If this parameter
            is part of an ``Instrument`` or ``Station``, this is how it will be
            referenced from that parent, i.e. ``instrument.name`` or
            ``instrument.parameters[name]``

        shape: The shape (as used in numpy arrays) of the array
            to expect. Scalars should be denoted by (), 1D arrays as (n,),
            2D arrays as (n, m), etc.

        instrument: The instrument this parameter
            belongs to, if any.

        label: Normally used as the axis label when this
            parameter is graphed, along with ``unit``.

        unit: The unit of measure. Use ``''`` for unitless.

        setpoints: ``array`` can be a DataArray, numpy.ndarray, or sequence.
            The setpoints for each dimension of the returned array. An
            N-dimension item should have N setpoint arrays, where the first is
            1D, the second 2D, etc.
            If omitted for any or all items, defaults to integers from zero in
            each respective direction.
            **Note**: if the setpoints will be different each measurement,
            leave this out and return the setpoints (with extra names) in
            ``.get``.

        setpoint_names: One identifier (like ``name``) per setpoint array.
            Ignored if a setpoint is a DataArray, which already has a name.

        setpoint_labels: One label (like ``labels``) per setpoint array.
            Ignored if a setpoint is a DataArray, which already has a label.

        setpoint_units: One unit (like ``v``) per setpoint array. Ignored
            if a setpoint is a DataArray, which already has a unit.

        docstring: documentation string for the ``__doc__``
            field of the object. The ``__doc__`` field of the instance
            is used by some help systems, but not all.

        snapshot_get: Prevent any update to the parameter, for example
            if it takes too long to update. Default ``True``.

        snapshot_value: Should the value of the parameter be stored in the
            snapshot. Unlike Parameter this defaults to False as
            ArrayParameters are potentially huge.

        snapshot_exclude: ``True`` prevents parameter to be
            included in the snapshot. Useful if there are many of the same
            parameter which are clogging up the snapshot.

            Default ``False``.

        metadata: Extra information to include with the
            JSON snapshot of the parameter.
    """

    def __init__(
        self,
        name: str,
        shape: Sequence[int],
        instrument: Optional["InstrumentBase"] = None,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        setpoints: Optional[Sequence[Any]] = None,
        setpoint_names: Optional[Sequence[str]] = None,
        setpoint_labels: Optional[Sequence[str]] = None,
        setpoint_units: Optional[Sequence[str]] = None,
        docstring: Optional[str] = None,
        snapshot_get: bool = True,
        snapshot_value: bool = False,
        snapshot_exclude: bool = False,
        metadata: Optional[Mapping[Any, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name,
            instrument,
            snapshot_get,
            metadata,
            snapshot_value=snapshot_value,
            snapshot_exclude=snapshot_exclude,
            **kwargs,
        )

        if self.settable:
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

        if not self.gettable and not self.settable:
            raise AttributeError('ArrayParameter must have a get, set or both')

    @property
    def setpoint_full_names(self) -> Optional[Sequence[str]]:
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


def _is_nested_sequence_or_none(obj: Any,
                                types: Optional[Union[
                                    Type[object],
                                    Tuple[Type[object], ...]]],
                                shapes: Sequence[Sequence[Optional[int]]]
                                ) -> bool:
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
    each of arbitrary shape. Not necessarily part of an instrument.

    Subclasses should define a ``.get_raw`` method, which returns a sequence of
    values. This method is automatically wrapped to provide a ``.get`` method.
    When used in a legacy  method``Loop`` or ``Measure`` operation, each of
    these values will be entered into a different ``DataArray``. The
    constructor args describe what data we expect from each ``.get`` call
    and how it should be handled. ``.get`` should always return the same
    number of items, and most of the constructor arguments should be tuples
    of that same length.

    For now you must specify upfront the array shape of each item returned by
    ``.get_raw``, and this cannot change from one call to the next. Later, we
    intend to require only that you specify the dimension of each item
    returned, and the size of each dimension can vary from call to call.

    Args:
        name: The local name of the whole parameter. Should be a valid
            identifier, ie no spaces or special characters. If this parameter
            is part of an Instrument or Station, this is how it will be
            referenced from that parent, i.e. ``instrument.name`` or
            ``instrument.parameters[name]``.

        names: A name for each item returned by a ``.get``
            call. Will be used as the basis of the ``DataArray`` names
            when this parameter is used to create a ``DataSet``.

        shapes: The shape (as used in numpy arrays) of
            each item. Scalars should be denoted by (), 1D arrays as (n,),
            2D arrays as (n, m), etc.

        instrument: The instrument this parameter
            belongs to, if any.

        labels: A label for each item. Normally used
            as the axis label when a component is graphed, along with the
            matching entry from ``units``.

        units: A unit of measure for each item.
            Use ``''`` or ``None`` for unitless values.

        setpoints: ``array`` can be a DataArray, numpy.ndarray, or sequence.
            The setpoints for each returned array. An N-dimension item should
            have N setpoint arrays, where the first is 1D, the second 2D, etc.
            If omitted for any or all items, defaults to integers from zero in
            each respective direction.
            **Note**: if the setpoints will be different each measurement,
            leave this out and return the setpoints (with extra names) in
            ``.get``.

        setpoint_names: One identifier (like
            ``name``) per setpoint array. Ignored if a setpoint is a
            DataArray, which already has a name.

        setpoint_labels: One label (like
            ``labels``) per setpoint array. Ignored if a setpoint is a
            DataArray, which already has a label.

        setpoint_units: One unit (like
            ``V``) per setpoint array. Ignored if a setpoint is a
            DataArray, which already has a unit.

        docstring: Documentation string for the ``__doc__``
            field of the object. The ``__doc__`` field of the  instance is
            used by some help systems, but not all

        snapshot_get: Prevent any update to the parameter, for example
            if it takes too long to update. Default ``True``.

        snapshot_value: Should the value of the parameter be stored in the
            snapshot. Unlike Parameter this defaults to False as
            MultiParameters are potentially huge.

        snapshot_exclude: True prevents parameter to be
            included in the snapshot. Useful if there are many of the same
            parameter which are clogging up the snapshot.
            Default ``False``.

        metadata: Extra information to include with the
            JSON snapshot of the parameter.
    """

    def __init__(
        self,
        name: str,
        names: Sequence[str],
        shapes: Sequence[Sequence[int]],
        instrument: Optional["InstrumentBase"] = None,
        labels: Optional[Sequence[str]] = None,
        units: Optional[Sequence[str]] = None,
        setpoints: Optional[Sequence[Sequence[Any]]] = None,
        setpoint_names: Optional[Sequence[Sequence[str]]] = None,
        setpoint_labels: Optional[Sequence[Sequence[str]]] = None,
        setpoint_units: Optional[Sequence[Sequence[str]]] = None,
        docstring: Optional[str] = None,
        snapshot_get: bool = True,
        snapshot_value: bool = False,
        snapshot_exclude: bool = False,
        metadata: Optional[Mapping[Any, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name,
            instrument,
            snapshot_get,
            metadata,
            snapshot_value=snapshot_value,
            snapshot_exclude=snapshot_exclude,
            **kwargs,
        )

        self._meta_attrs.extend(['setpoint_names', 'setpoint_labels',
                                 'setpoint_units', 'names', 'labels', 'units'])

        if not is_sequence_of(names, str):
            raise ValueError('names must be a tuple of strings, not ' +
                             repr(names))

        self.names = tuple(names)
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

        if not self.gettable and not self.settable:
            raise AttributeError('MultiParameter must have a get, set or both')

    @property
    def short_names(self) -> Tuple[str, ...]:
        """
        short_names is identical to names i.e. the names of the parameter
        parts but does not add the instrument name.

        It exists for consistency with instruments and other parameters.
        """

        return self.names

    @property
    def full_names(self) -> Tuple[str, ...]:
        """
        Names of the parameter components including the name of the instrument
        and submodule that the parameter may be bound to. The name parts are
        separated by underscores, like this: ``instrument_submodule_parameter``
        """
        inst_name = "_".join(self.name_parts[:-1])
        if inst_name != "":
            return tuple(inst_name + "_" + name for name in self.names)
        else:
            return self.names

    @property
    def setpoint_full_names(self) -> Optional[Sequence[Sequence[str]]]:
        """
        Full names of setpoints including instrument names, if available
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
                        full_sp_names_subgroupd.append(
                            inst_name + '_' + spname)
                    else:
                        full_sp_names_subgroupd.append(None)
                full_sp_names.append(tuple(full_sp_names_subgroupd))

            return tuple(full_sp_names)
        else:
            return self.setpoint_names


class _CacheProtocol(Protocol):
    """
    This protocol defines the interface that a Parameter Cache implementation
    must implement. This is currently used for 2 implementations, one in
    _BaseParameter and a specialized one in DelegateParameter.
    """
    @property
    def raw_value(self) -> ParamRawDataType:
        ...

    @property
    def timestamp(self) -> Optional[datetime]:
        ...

    @property
    def max_val_age(self) -> Optional[float]:
        ...

    @property
    def valid(self) -> bool:
        ...

    def invalidate(self) -> None:
        ...

    def set(self, value: ParamDataType) -> None:
        ...

    def _set_from_raw_value(self, raw_value: ParamRawDataType) -> None:
        ...

    def get(self, get_if_invalid: bool = True) -> ParamDataType:
        ...

    def _update_with(self, *,
                     value: ParamDataType,
                     raw_value: ParamRawDataType,
                     timestamp: Optional[datetime] = None
                     ) -> None:
        ...

    def __call__(self) -> ParamDataType:
        ...


class _Cache:
    """
    Cache object for parameter to hold its value and raw value

    It also implements ``set`` method for setting parameter's value without
    invoking its ``set_cmd``, and ``get`` method that allows to retrieve the
    cached value of the parameter without calling ``get_cmd`` might be called
    unless the cache is invalid.

    Args:
         parameter: instance of the parameter that this cache belongs to.
         max_val_age: Max time (in seconds) to trust a value stored in cache.
            If the parameter has not been set or measured more recently than
            this, an additional measurement will be performed in order to
            update the cached value. If it is ``None``, this behavior is
            disabled. ``max_val_age`` should not be used for a parameter
            that does not have a get function.
    """
    def __init__(self,
                 parameter: '_BaseParameter',
                 max_val_age: Optional[float] = None):
        self._parameter = parameter
        self._value: ParamDataType = None
        self._raw_value: ParamRawDataType = None
        self._timestamp: Optional[datetime] = None
        self._max_val_age = max_val_age
        self._marked_valid: bool = False

    @property
    def raw_value(self) -> ParamRawDataType:
        """Raw value of the parameter"""
        return self._raw_value

    @property
    def timestamp(self) -> Optional[datetime]:
        """
        Timestamp of the moment when cache was last updated

        If ``None``, the cache hasn't been updated yet and shall be seen as
        "invalid".
        """
        return self._timestamp

    @property
    def max_val_age(self) -> Optional[float]:
        """
        Max time (in seconds) to trust a value stored in cache. If the
        parameter has not been set or measured more recently than this,
        perform an additional measurement.

        If it is ``None``, this behavior is disabled.
        """
        return self._max_val_age

    @property
    def valid(self) -> bool:
        """
        Returns True if the cache is expected be be valid.
        """
        return not self._timestamp_expired() and self._marked_valid

    def invalidate(self) -> None:
        """
        Call this method to mark the cache invalid.
        If the cache is invalid the next call to `cache.get()` attempt
        to get the value from the instrument.
        """
        self._marked_valid = False

    def set(self, value: ParamDataType) -> None:
        """
        Set the cached value of the parameter without invoking the
        ``set_cmd`` of the parameter (if it has one). For example, in case of
        an instrument parameter, calling :meth:`cache.set` as opposed to
        calling ``set`` will only change the internally-stored value of
        the parameter (that is available when calling ``cache.get()`` or
        ``get_latest()``), and will NOT pass that value to the instrument.

        Note that this method also respects all the validation, parsing,
        offsetting, etc that the parameter's ``set`` method respects. However,
        if the parameter has :attr:`step` defined, unlike the ``set`` method,
        this method does not perform setting the parameter step-by-step.

        Args:
            value: new value for the parameter
        """
        self._parameter.validate(value)
        raw_value = self._parameter._from_value_to_raw_value(value)
        self._update_with(value=value, raw_value=raw_value)

    def _set_from_raw_value(self, raw_value: ParamRawDataType) -> None:
        value = self._parameter._from_raw_value_to_value(raw_value)
        if self._parameter._validate_on_get:
            self._parameter.validate(value)
        self._update_with(value=value, raw_value=raw_value)

    def _update_with(self, *,
                     value: ParamDataType,
                     raw_value: ParamRawDataType,
                     timestamp: Optional[datetime] = None
                     ) -> None:
        """
        Simply overwrites the value, raw value, and timestamp in this cache
        with new ones.

        Args:
            value: new value of the parameter
            raw_value: new raw value of the parameter
            timestamp: new timestamp of the parameter; if ``None``,
                then timestamp of "now" is used
        """
        self._value = value
        self._raw_value = raw_value
        if timestamp is None:
            self._timestamp = datetime.now()
        else:
            self._timestamp = timestamp
        self._marked_valid = True

    def _timestamp_expired(self) -> bool:
        if self._timestamp is None:
            # parameter has never been captured
            return True
        if self._max_val_age is None:
            # parameter cannot expire
            return False
        oldest_accepted_timestamp = (
                datetime.now() - timedelta(seconds=self._max_val_age))
        if self._timestamp < oldest_accepted_timestamp:
            # Time of last get exceeds max_val_age seconds, need to
            # perform new .get()
            return True
        else:
            # parameter is still valid
            return False

    def get(self, get_if_invalid: bool = True) -> ParamDataType:
        """
        Return cached value if time since get was less than ``max_val_age``,
        or the parameter was explicitly marked invalid.
        Otherwise perform ``get()`` on the parameter and return result. A
        ``get()`` will also be performed if the parameter has never been
        captured but only if ``get_if_invalid`` argument is ``True``.

        Args:
            get_if_invalid: if set to ``True``, ``get()`` on a parameter
                will be performed in case the cached value is invalid (for
                example, due to ``max_val_age``, because the parameter has
                never been captured, or because the parameter was marked
                invalid)
        """

        gettable = self._parameter.gettable
        cache_valid = self.valid

        if cache_valid:
            return self._value
        else:
            if get_if_invalid:
                if gettable:
                    return self._parameter.get()
                else:
                    error_msg = self._construct_error_msg()
                    raise RuntimeError(error_msg)
            else:
                return self._value

    def _construct_error_msg(self) -> str:
        if self._timestamp is None:
            error_msg = (f"Value of parameter "
                         f"{self._parameter.full_name} "
                         f"is unknown and the Parameter "
                         f"does not have a get command. "
                         f"Please set the value before "
                         f"attempting to get it.")
        elif self._max_val_age is not None:
            # TODO: this check should really be at the time
            #  of setting max_val_age unfortunately this
            #  happens in init before get wrapping is performed.
            error_msg = ("`max_val_age` is not supported "
                         "for a parameter without get "
                         "command.")
        else:
            # max_val_age is None and TS is not None but cache is
            # invalid with the current logic that should never
            # happen
            error_msg = ("Cannot return cache of a parameter "
                         "that does not have a get command "
                         "and has an invalid cache")
        return error_msg

    def __call__(self) -> ParamDataType:
        """
        Same as :meth:`get` but always call ``get`` on parameter if the
        cache is not valid
        """
        return self.get(get_if_invalid=True)


class GetLatest(DelegateAttributes):
    """
    Wrapper for a class:`.Parameter` that just returns the last set or measured
    value stored in the class:`.Parameter` itself. If get has never been called
    on the parameter or the time since get was called is larger than
    ``max_val_age``, get will be called on the parameter. If the parameter
    does not implement get, set should be called (or the initial_value set)
    before calling get on this wrapper. It is an error to set
    ``max_val_age`` for a parameter that does not have a get function.

    The functionality of this class is subsumed and improved in
    parameter's cache that is accessible via ``.cache`` attribute of the
    :class:`.Parameter`. Use of ``parameter.cache`` is recommended over use of
    ``parameter.get_latest``.

    Examples:
        >>> # Can be called:
        >>> param.get_latest()
        >>> # Or used as if it were a gettable-only parameter itself:
        >>> Loop(...).each(param.get_latest)

    Args:
        parameter: Parameter to be wrapped.
    """
    def __init__(self, parameter: _BaseParameter):
        self.parameter = parameter

    delegate_attr_objects = ['parameter']
    omit_delegate_attrs = ['set']

    def get(self) -> ParamDataType:
        """
        Return latest value if time since get was less than
        `max_val_age`, otherwise perform `get()` and
        return result. A `get()` will also be performed if the
        parameter never has been captured.

        It is recommended to use ``parameter.cache.get()`` instead.
        """
        return self.parameter.cache.get()

    def get_timestamp(self) -> Optional[datetime]:
        """
        Return the age of the latest parameter value.

        It is recommended to use ``parameter.cache.timestamp`` instead.
        """
        return self.cache.timestamp

    def get_raw_value(self) -> Optional[ParamRawDataType]:
        """
        Return latest raw value of the parameter.

        It is recommended to use ``parameter.cache.raw_value`` instead.
        """
        return self.cache._raw_value

    def __call__(self) -> ParamDataType:
        """
        Same as ``get()``

        It is recommended to use ``parameter.cache()`` instead.
        """
        return self.cache()


def combine(*parameters: 'Parameter',
            name: str,
            label: Optional[str] = None,
            unit: Optional[str] = None,
            units: Optional[str] = None,
            aggregator: Optional[Callable[[Sequence[Any]], Any]] = None
            ) -> 'CombinedParameter':
    """
    Combine parameters into one sweepable parameter

    A combined parameter sets all the combined parameters at every point
    of the sweep. The sets are called in the same order the parameters are,
    and sequentially.

    Args:
        *parameters: The parameters to combine.
        name: The name of the paramter.
        label: The label of the combined parameter.
        unit: the unit of the combined parameter.
        aggregator: a function to aggregate
            the set values into one.
    """
    my_parameters = list(parameters)
    multi_par = CombinedParameter(my_parameters, name, label, unit, units,
                                  aggregator)
    return multi_par


class CombinedParameter(Metadatable):
    """
    A combined parameter. It sets all the combined parameters at every
    point of the sweep. The sets are called in the same order
    the parameters are, and sequentially.

    Args:
        *parameters: The parameters to combine.
        name: The name of the parameter
        label: The label of the combined parameter
        unit: The unit of the combined parameter
        aggregator: A function to aggregate the set values into one
    """

    def __init__(self, parameters: Sequence[Parameter],
                 name: str,
                 label: Optional[str] = None,
                 unit: Optional[str] = None,
                 units: Optional[str] = None,
                 aggregator: Optional[Callable[..., Any]] = None) -> None:
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
        self.parameter.full_name = name  # type: ignore[attr-defined]
        self.parameter.name = name  # type: ignore[attr-defined]
        self.parameter.label = label  # type: ignore[attr-defined]

        if units is not None:
            warn_units('CombinedParameter', self)
            if unit is None:
                unit = units
        self.parameter.unit = unit  # type: ignore[attr-defined]
        self.setpoints: List[Any] = []
        # endhack
        self.parameters = parameters
        self.sets = [parameter.set for parameter in self.parameters]
        self.dimensionality = len(self.sets)

        if aggregator:
            self.f = aggregator
            setattr(self, 'aggregate', self._aggregate)

    def set(self, index: int) -> List[Any]:
        """
        Set multiple parameters.

        Args:
            index: the index of the setpoints one wants to set

        Returns:
            list of values that where actually set
        """
        values = self.setpoints[index]
        for setFunction, value in zip(self.sets, values):
            setFunction(value)
        return values

    def sweep(self, *array: numpy.ndarray) -> 'CombinedParameter':
        """
        Creates a new combined parameter to be iterated over.
        One can sweep over either:

         - n array of length m
         - one nxm array

        where n is the number of combined parameters
        and m is the number of setpoints

        Args:
            *array: Array(s) of setpoints.

        Returns:
            combined parameter
        """
        # if it's a list of arrays, convert to one array
        if len(array) > 1:
            dim = {len(a) for a in array}
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

    def _aggregate(self, *vals: Any) -> Any:
        # check f args
        return self.f(*vals)

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.setpoints)))

    def __len__(self) -> int:
        # dimension of the sweep_values
        # i.e. how many setpoint
        return numpy.shape(self.setpoints)[0]

    def snapshot_base(self, update: Optional[bool] = False,
                      params_to_skip_update: Optional[Sequence[str]] = None
                      ) -> Dict[Any, Any]:
        """
        State of the combined parameter as a JSON-compatible dict (everything
        that the custom JSON encoder class
        :class:`qcodes.utils.helpers.NumpyJSONEncoder` supports).

        Args:
            update: ``True`` or ``False``.
            params_to_skip_update: Unused in this subclass.

        Returns:
            dict: Base snapshot.
        """
        meta_data: Dict[str, Any] = collections.OrderedDict()
        meta_data['__class__'] = full_class(self)
        param = self.parameter
        meta_data['unit'] = param.unit  # type: ignore[attr-defined]
        meta_data['label'] = param.label  # type: ignore[attr-defined]
        meta_data['full_name'] = param.full_name  # type: ignore[attr-defined]
        meta_data['aggregator'] = repr(getattr(self, 'f', None))
        for parameter in self.parameters:
            meta_data[str(parameter)] = parameter.snapshot()

        return meta_data


class InstrumentRefParameter(Parameter):
    """
    An instrument reference parameter.

    This parameter is useful when one needs a reference to another instrument
    from within an instrument, e.g., when creating a meta instrument that
    sets parameters on instruments it contains.

    Args:
        name: The name of the parameter that one wants to add.

        instrument: The "parent" instrument this
            parameter is attached to, if any.

        initial_value: Starting value, may be None even if None does not
            pass the validator. None is only allowed as an initial value
            and cannot be set after initiation.

        **kwargs: Passed to InstrumentRefParameter parent class
    """

    def __init__(self, name: str,
                 instrument: Optional['InstrumentBase'] = None,
                 label: Optional[str] = None,
                 unit: Optional[str] = None,
                 get_cmd: Optional[Union[str, Callable[..., Any], bool]] = None,
                 set_cmd: Optional[Union[str, Callable[..., Any], bool]] = None,
                 initial_value: Optional[Union[float, str]] = None,
                 max_val_age: Optional[float] = None,
                 vals: Optional[Validator[Any]] = None,
                 docstring: Optional[str] = None,
                 **kwargs: Any) -> None:
        if vals is None:
            vals = Strings()
        if set_cmd is not None:
            raise RuntimeError("InstrumentRefParameter does not support "
                               "set_cmd.")
        super().__init__(name, instrument, label, unit, get_cmd, set_cmd,
                         initial_value, max_val_age, vals, docstring,
                         **kwargs)

    # TODO(nulinspiratie) check class works now it's subclassed from Parameter
    def get_instr(self) -> 'InstrumentBase':
        """
        Returns the instance of the instrument with the name equal to the
        value of this parameter.
        """
        ref_instrument_name = self.get()
        # note that _instrument refers to the instrument this parameter belongs
        # to, while the ref_instrument_name is the instrument that is the value
        # of this parameter.
        if self._instrument is None:
            raise RuntimeError("InstrumentRefParameter is not bound to "
                               "an instrument.")
        return self._instrument.find_instrument(ref_instrument_name)


class ManualParameter(Parameter):
    def __init__(self, name: str,
                 instrument: Optional['InstrumentBase'] = None,
                 initial_value: Any = None,
                 **kwargs: Any):
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
    :class:`.Parameter` Scaler

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
        >>> Id = ScaledParameter(multimeter.amplitude,
        ...                      division = 1e6, name = 'Id', unit = 'A')

    Args:
        output: Physical Parameter that need conversion.
        division: The division value.
        gain: The gain value.
        label: Label of this parameter, by default uses 'output' label
            but attaches _amplified or _attenuated depending if gain
            or division has been specified.
        name: Name of this parameter, by default uses 'output' name
            but attaches _amplified or _attenuated depending if gain
            or division has been specified.
        unit: Resulting unit. It uses the one of 'output' by default.
    """

    class Role(enum.Enum):
        GAIN = enum.auto()
        DIVISION = enum.auto()

    def __init__(self,
                 output: Parameter,
                 division: Optional[Union[float, Parameter]] = None,
                 gain: Optional[Union[float, Parameter]] = None,
                 name: Optional[str] = None,
                 label: Optional[str] = None,
                 unit: Optional[str] = None) -> None:

        # Set label
        if label:
            self.label = label
        elif name:
            self.label = name
        else:
            self.label = f"{output.label}_scaled"

        # Set the name
        if not name:
            name = f"{output.name}_scaled"

        # Set the unit
        if unit:
            self.unit = unit
        else:
            self.unit = output.unit

        super().__init__(
            name=name,
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

        if division is not None:
            self.role = ScaledParameter.Role.DIVISION
            # Unfortunately mypy does not support
            # properties where the setter has different types than
            # the actual property. We use that here to cast different inputs
            # to the same type.
            # https://github.com/python/mypy/issues/3004
            self._multiplier = division  # type: ignore[assignment]
        elif gain is not None:
            self.role = ScaledParameter.Role.GAIN
            self._multiplier = gain  # type: ignore[assignment]

        # extend metadata
        self._meta_attrs.extend(["division"])
        self._meta_attrs.extend(["gain"])
        self._meta_attrs.extend(["role"])
        self.metadata['wrapped_parameter'] = self._wrapped_parameter.name
        if self._wrapped_instrument:
            wrapped_instr_name = getattr(self._wrapped_instrument, "name", None)
            self.metadata['wrapped_instrument'] = wrapped_instr_name

    # Internal handling of the multiplier
    # can be either a Parameter or a scalar
    @property
    def _multiplier(self) -> Parameter:
        if self._multiplier_parameter is None:
            raise RuntimeError("Cannot get multiplier when multiplier "
                               "parameter in unknown.")
        return self._multiplier_parameter

    @_multiplier.setter
    def _multiplier(self, multiplier: Union[float, Parameter]) -> None:
        if isinstance(multiplier, Parameter):
            self._multiplier_parameter = multiplier
            multiplier_name = self._multiplier_parameter.name
            self.metadata['variable_multiplier'] = multiplier_name
        else:
            self._multiplier_parameter = ManualParameter(
                'multiplier', initial_value=multiplier)
            self.metadata['variable_multiplier'] = False

    # Division of the scaler
    @property
    def division(self) -> float:  # type: ignore[return]
        value = cast(float, self._multiplier())
        if self.role == ScaledParameter.Role.DIVISION:
            return value
        elif self.role == ScaledParameter.Role.GAIN:
            return 1 / value

    @division.setter
    def division(self, division: Union[float, Parameter]) -> None:
        self.role = ScaledParameter.Role.DIVISION
        self._multiplier = division  # type: ignore[assignment]

    # Gain of the scaler
    @property
    def gain(self) -> float:   # type: ignore[return]
        value = cast(float, self._multiplier())
        if self.role == ScaledParameter.Role.GAIN:
            return value
        elif self.role == ScaledParameter.Role.DIVISION:
            return 1 / value

    @gain.setter
    def gain(self, gain: Union[float, Parameter]) -> None:
        self.role = ScaledParameter.Role.GAIN
        self._multiplier = gain  # type: ignore[assignment]

    # Getter and setter for the real value
    def get_raw(self) -> float:
        """
        Returns:
            value at which was set at the sample
        """
        wrapped_value = cast(float, self._wrapped_parameter())
        multiplier = cast(float, self._multiplier())

        if self.role == ScaledParameter.Role.GAIN:
            value = wrapped_value * multiplier
        elif self.role == ScaledParameter.Role.DIVISION:
            value = wrapped_value / multiplier
        else:
            raise RuntimeError(f"ScaledParameter must be either a"
                               f"Multiplier or Divisor; got {self.role}")

        return value

    @property
    def wrapped_parameter(self) -> Parameter:
        """
        The attached unscaled parameter
        """
        return self._wrapped_parameter

    def get_wrapped_parameter_value(self) -> float:
        """
        Returns:
            value at which the attached parameter is (i.e. does
            not account for the scaling)
        """
        return self._wrapped_parameter.get()

    def set_raw(self, value: float) -> None:
        """
        Set the value on the wrapped parameter, accounting for the scaling
        """
        multiplier_value = cast(float, self._multiplier())
        if self.role == ScaledParameter.Role.GAIN:
            instrument_value = value / multiplier_value
        elif self.role == ScaledParameter.Role.DIVISION:
            instrument_value = value * multiplier_value
        else:
            raise RuntimeError(f"ScaledParameter must be either a"
                               f"Multiplier or Divisor; got {self.role}")

        self._wrapped_parameter.set(instrument_value)


def expand_setpoints_helper(parameter: ParameterWithSetpoints,
                            results: Optional[ParamDataType] = None) -> List[
        Tuple[_BaseParameter, ParamDataType]]:
    """
    A helper function that takes a :class:`.ParameterWithSetpoints` and
    acquires the parameter along with it's setpoints. The data is returned
    in a format prepared to insert into the dataset.

    Args:
        parameter: A :class:`.ParameterWithSetpoints` to be acquired and
            expanded
        results: The data for the given parameter. Typically the output of
            `parameter.get()`. If None this function will call `parameter.get`

    Returns:
        A list of tuples of parameters and values for the specified parameter
        and its setpoints.
    """
    if not isinstance(parameter, ParameterWithSetpoints):
        raise TypeError(
            f"Expanding setpoints only works for ParameterWithSetpoints. "
            f"Supplied a {type(parameter)}")
    res = []
    setpoint_params = []
    setpoint_data = []
    for setpointparam in parameter.setpoints:
        these_setpoints = setpointparam.get()
        setpoint_params.append(setpointparam)
        setpoint_data.append(these_setpoints)
    output_grids = numpy.meshgrid(*setpoint_data, indexing='ij')
    for param, grid in zip(setpoint_params, output_grids):
        res.append((param, grid))
    if results is None:
        data = parameter.get()
    else:
        data = results
    res.append((parameter, data))
    return res
