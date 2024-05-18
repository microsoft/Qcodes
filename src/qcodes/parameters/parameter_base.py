from __future__ import annotations

import collections.abc
import logging
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from functools import cached_property, wraps
from typing import TYPE_CHECKING, Any, ClassVar, overload

from qcodes.metadatable import Metadatable, MetadatableWithName
from qcodes.utils import DelegateAttributes, full_class, qcodes_abstractmethod
from qcodes.validators import Enum, Ints, Validator

from .cache import _Cache, _CacheProtocol
from .named_repr import named_repr
from .permissive_range import permissive_range

# for now the type the parameter may contain is not restricted at all
ParamDataType = Any
ParamRawDataType = Any

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping, Sequence, Sized
    from types import TracebackType

    from qcodes.instrument.base import InstrumentBase

LOG = logging.getLogger(__name__)


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

    def __init__(
        self,
        parameter: ParameterBase,
        value: ParamDataType,
        allow_changes: bool = False,
    ):
        self._parameter: ParameterBase = parameter
        self._value = value
        self._allow_changes = allow_changes
        self._original_value = None
        self._original_settable: bool | None = None

    def __enter__(self) -> None:
        self._original_value = self._parameter.cache()

        if self._original_value != self._value:
            self._parameter.set(self._value)

        if not self._allow_changes:
            self._original_settable = self._parameter.settable
            self._parameter._settable = False

    def __exit__(
        self,
        typ: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self._allow_changes:
            assert self._original_settable is not None
            self._parameter._settable = self._original_settable

        if self._parameter.cache() != self._original_value:
            try:
                self._parameter.set(self._original_value)
            except Exception:
                # Likely an uninitialized Parameter
                LOG.info(
                    "Encountered an exception setting the original value "
                    "when exiting set_to context of "
                    f"{self._parameter.full_name}",
                    exc_info=True,
                )


def invert_val_mapping(val_mapping: Mapping[Any, Any]) -> dict[Any, Any]:
    """Inverts the value mapping dictionary for allowed parameter values"""
    return {v: k for k, v in val_mapping.items()}


class ParameterBase(MetadatableWithName):
    """
    Shared behavior for all parameters. Not intended to be used
    directly, normally you should use ``Parameter``, ``ArrayParameter``,
    ``MultiParameter``, or ``CombinedParameter``.
    Note that ``CombinedParameter`` is not yet a subclass of ``ParameterBase``

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

        register_name: Specifies if the parameter should be registered in datasets
            using a different name than the parameter's full_name
    """

    def __init__(
        self,
        name: str,
        instrument: InstrumentBase | None,
        snapshot_get: bool = True,
        metadata: Mapping[Any, Any] | None = None,
        step: float | None = None,
        scale: float | Iterable[float] | None = None,
        offset: float | Iterable[float] | None = None,
        inter_delay: float = 0,
        post_delay: float = 0,
        val_mapping: Mapping[Any, Any] | None = None,
        get_parser: Callable[..., Any] | None = None,
        set_parser: Callable[..., Any] | None = None,
        snapshot_value: bool = True,
        snapshot_exclude: bool = False,
        max_val_age: float | None = None,
        vals: Validator[Any] | None = None,
        abstract: bool | None = False,
        bind_to_instrument: bool = True,
        register_name: str | None = None,
    ) -> None:
        super().__init__(metadata)
        if not str(name).isidentifier():
            raise ValueError(
                f"Parameter name must be a valid identifier "
                f"got {name} which is not. Parameter names "
                f"cannot start with a number and "
                f"must not contain spaces or special characters"
            )
        self._short_name = str(name)
        self._register_name = register_name
        self._instrument = instrument
        self._snapshot_get = snapshot_get
        self._snapshot_value = snapshot_value
        self.snapshot_exclude = snapshot_exclude

        if not isinstance(vals, (Validator, type(None))):
            raise TypeError("vals must be None or a Validator")
        elif val_mapping is not None:
            vals = Enum(*val_mapping.keys())
        if vals is not None:
            self._vals: list[Validator[Any]] = [vals]
        else:
            self._vals = []

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

        self.get_parser: Callable[..., Any] | None = get_parser
        self.set_parser: Callable[..., Any] | None = set_parser

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
        implements_get_raw = hasattr(self, "get_raw") and not getattr(
            self.get_raw, "__qcodes_is_abstract_method__", False
        )
        self._gettable = False
        if implements_get_raw:
            self.get = self._wrap_get(self.get_raw)
            self._gettable = True
        elif hasattr(self, "get"):
            raise RuntimeError(
                f"Overwriting get in a subclass of "
                f"ParameterBase: "
                f"{self.full_name} is not allowed."
            )

        self.set: Callable[..., None]
        implements_set_raw = hasattr(self, "set_raw") and not getattr(
            self.set_raw, "__qcodes_is_abstract_method__", False
        )
        self._settable: bool = False
        if implements_set_raw:
            self.set = self._wrap_set(self.set_raw)
            self._settable = True
        elif hasattr(self, "set"):
            raise RuntimeError(
                f"Overwriting set in a subclass of "
                f"ParameterBase: "
                f"{self.full_name} is not allowed."
            )

        # subclasses should extend this list with extra attributes they
        # want automatically included in the snapshot
        self._meta_attrs = [
            "name",
            "instrument",
            "step",
            "scale",
            "offset",
            "inter_delay",
            "post_delay",
            "val_mapping",
            "vals",
            "validators",
        ]

        # Specify time of last set operation, used when comparing to delay to
        # check if additional waiting time is needed before next set
        self._t_last_set = time.perf_counter()
        # should we call validate when getting data. default to False
        # intended to be changed in a subclass if you want the subclass
        # to perform a validation on get
        self._validate_on_get: bool = False
        self._abstract = abstract

        if instrument is not None and bind_to_instrument:
            existing_parameter = instrument.parameters.get(name, None)

            if existing_parameter:
                if not existing_parameter.abstract:
                    raise KeyError(
                        f"Duplicate parameter name {name} on instrument {instrument}"
                    )

            instrument.parameters[name] = self

    def _build__doc__(self) -> str | None:
        return self.__doc__

    @property
    def vals(self) -> Validator | None:
        """
        The first validator of the parameter. None
        if no validators are set for this parameter.

        :getter: Returns the first validator or None if no validators.
        :setter: Sets the first validator. Set to None to remove the first validator.

        Raises:
            RuntimeError: If removing the first validator when more than one validator is set.
        """

        if len(self._vals):
            return self._vals[0]
        else:
            return None

    @vals.setter
    def vals(self, vals: Validator | None) -> None:
        if vals is not None and len(self._vals) > 0:
            self._vals[0] = vals
        elif vals is not None:
            self._vals = [vals]
        elif len(self._vals) == 1:
            self._vals = []
        elif len(self._vals) > 1:
            raise RuntimeError(
                "Cannot remove default validator from parameter with additional validators."
            )
        else:
            # setting the validator to None but the parameter already doesn't have a validator
            pass
        self.__doc__ = self._build__doc__()

    def add_validator(self, vals: Validator) -> None:
        """Add a validator for the parameter. The parameter is validated against
        all validators in reverse order of how they are added.

        Args:
            vals: Validator to add to the parameter.
        """
        self._vals.append(vals)
        self.__doc__ = self._build__doc__()

    def remove_validator(self) -> Validator | None:
        """
        Remove the last validator added to the parameter and return it.
        Returns None if there are no validators associated with the parameter.

        Returns:
            The last validator added to the parameter or None if there are no
            validators associated with the parameter.
        """
        if len(self._vals) > 0:
            removed = self._vals.pop()
            self.__doc__ = self._build__doc__()
            return removed
        else:
            return None

    @property
    def validators(self) -> tuple[Validator, ...]:
        """
        Tuple of all validators associated with the parameter.

        :getter: All validators associated with the parameter.
        """

        return tuple(self._vals)

    @contextmanager
    def extra_validator(self, vals: Validator) -> Generator[None, None, None]:
        """
        Contextmanager to to temporarily add a validator to the parameter within the
        given context. The validator is removed from the parameter when the context
        ends.
        """
        self.add_validator(vals)
        yield
        self.remove_validator()

    @property
    def raw_value(self) -> ParamRawDataType:
        """
        Note that this property will be deprecated soon. Use
        ``cache.raw_value`` instead.

        Represents the cached raw value of the parameter.

        :getter: Returns the cached raw value of the parameter.
        """
        return self.cache.raw_value

    @qcodes_abstractmethod
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

    @qcodes_abstractmethod
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
        inst_name = getattr(self._instrument, "name", "")
        if inst_name:
            return f"{inst_name}_{self.name}"
        else:
            return self.name

    def __repr__(self) -> str:
        return named_repr(self)

    @overload
    def __call__(self) -> ParamDataType:
        pass

    @overload
    def __call__(self, value: ParamDataType, **kwargs: Any) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> ParamDataType | None:
        if len(args) == 0 and len(kwargs) == 0:
            if self.gettable:
                return self.get()
            else:
                raise NotImplementedError(f"no get cmd found in Parameter {self.name}")
        elif self.settable:
            self.set(*args, **kwargs)
            return None
        else:
            raise NotImplementedError(f"no set cmd found in Parameter {self.name}")

    def snapshot_base(
        self,
        update: bool | None = True,
        params_to_skip_update: Sequence[str] | None = None,
    ) -> dict[Any, Any]:
        """
        State of the parameter as a JSON-compatible dict (everything that
        the custom JSON encoder class
        :class:`.NumpyJSONEncoder` supports).

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
                f"Parameter ({self.full_name}) is used in the snapshot while it "
                f"should be excluded from the snapshot",
                stacklevel=2,
            )

        state: dict[str, Any] = {"__class__": full_class(self), "full_name": str(self)}

        if self._snapshot_value:
            has_get = self.gettable
            allowed_to_call_get_when_snapshotting = (
                self._snapshot_get and update is not False
            )
            can_call_get_when_snapshotting = (
                allowed_to_call_get_when_snapshotting and has_get
            )

            if can_call_get_when_snapshotting and update:
                state["value"] = self.get()
            else:
                state["value"] = self.cache.get(
                    get_if_invalid=can_call_get_when_snapshotting
                )

            state["raw_value"] = self.cache.raw_value

        state["ts"] = self.cache.timestamp

        if isinstance(state["ts"], datetime):
            dttime: datetime = state["ts"]
            state["ts"] = dttime.strftime("%Y-%m-%d %H:%M:%S")

        for attr in set(self._meta_attrs):
            if attr == "instrument" and self._instrument:
                state.update(
                    {
                        "instrument": full_class(self._instrument),
                        "instrument_name": self._instrument.name,
                    }
                )
            elif attr == "validators":
                state["validators"] = [repr(validator) for validator in self.validators]
            else:
                val = getattr(self, attr, None)
                if val is not None:
                    attr_strip = attr.lstrip("_")  # strip leading underscores
                    if isinstance(val, Validator):
                        state[attr_strip] = repr(val)
                    elif isinstance(val, Metadatable):
                        state[attr_strip] = val.snapshot(update=update)
                    else:
                        state[attr_strip] = val

        return state

    @property
    def snapshot_value(self) -> bool:
        """
        If True the value of the parameter will be included in the snapshot.
        """
        return self._snapshot_value

    def _from_value_to_raw_value(self, value: ParamDataType) -> ParamRawDataType:
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
                raw_value = tuple(
                    val * scale for val, scale in zip(raw_value, self.scale)
                )
            else:
                # Use single scale for all values
                raw_value = raw_value * self.scale

        # apply offset next
        if self.offset is not None:
            if isinstance(self.offset, collections.abc.Iterable):
                # offset contains multiple elements, one for each value
                raw_value = tuple(
                    val + offset for val, offset in zip(raw_value, self.offset)
                )
            else:
                # Use single offset for all values
                raw_value = raw_value + self.offset

        # parser last
        if self.set_parser is not None:
            raw_value = self.set_parser(raw_value)

        return raw_value

    def _from_raw_value_to_value(self, raw_value: ParamRawDataType) -> ParamDataType:
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
                    value = tuple(
                        val - offset for val, offset in zip(value, self.offset)
                    )
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
                    value = tuple(val / scale for val, scale in zip(value, self.scale))
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

    def _wrap_get(
        self, get_function: Callable[..., ParamRawDataType]
    ) -> Callable[..., ParamDataType]:
        @wraps(get_function)
        def get_wrapper(*args: Any, **kwargs: Any) -> ParamDataType:
            if not self.gettable:
                raise TypeError("Trying to get a parameter that is not gettable.")
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
                e.args = e.args + (f"getting {self}",)
                raise e

        return get_wrapper

    def _wrap_set(self, set_function: Callable[..., None]) -> Callable[..., None]:
        @wraps(set_function)
        def set_wrapper(value: ParamDataType, **kwargs: Any) -> None:
            try:
                if not self.settable:
                    raise TypeError("Trying to set a parameter that is not settable.")
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

                    self.cache._update_with(value=val_step, raw_value=raw_val_step)

            except Exception as e:
                e.args = e.args + (f"setting {self} to {value}",)
                raise e

        return set_wrapper

    def get_ramp_values(
        self, value: float | Sized, step: float | None = None
    ) -> Sequence[float | Sized]:
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
                raise RuntimeError(
                    "Don't know how to step a parameter with more than one value"
                )
            if self.get_latest() is None:
                self.get()
            start_value = self.get_latest()
            if not (
                isinstance(start_value, (int, float))
                and isinstance(value, (int, float))
            ):
                # parameter is numeric but either one of the endpoints
                # is not or the starting point is unknown. The later
                # can happen for a non gettable parameter in the initial set
                # operation.
                LOG.warning(
                    f"cannot sweep {self.name} from {start_value!r} "
                    f"to {value!r} - jumping."
                )
                return [value]

            # drop the initial value, we're already there
            return permissive_range(start_value, value, step)[1:] + [value]

    @cached_property
    def _validate_context(self) -> str:
        # return string describing the context for a validator
        if self._instrument:
            context = (
                (
                    getattr(self._instrument, "name", "")
                    or str(self._instrument.__class__)
                )
                + "."
                + self.name
            )
        else:
            context = self.name
        return "Parameter: " + context

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
        for validator in reversed(self._vals):
            if validator is not None:
                validator.validate(value, self._validate_context)

    @property
    def step(self) -> float | None:
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
    def step(self, step: float | None) -> None:
        if step is None:
            self._step: float | None = step
        elif not all(getattr(vals, "is_numeric", True) for vals in self._vals):
            raise TypeError("you can only step numeric parameters")
        elif not isinstance(step, (int, float)):
            raise TypeError("step must be a number")
        elif step == 0:
            self._step = None
        elif step <= 0:
            raise ValueError("step must be positive")
        elif any(isinstance(vals, Ints) for vals in self._vals) and not isinstance(
            step, int
        ):
            raise TypeError("step must be a positive int for an Ints parameter")

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
            raise TypeError(f"post_delay ({post_delay}) must be a number")
        if post_delay < 0:
            raise ValueError(f"post_delay ({post_delay}) must not be negative")
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
            raise TypeError(f"inter_delay ({inter_delay}) must be a number")
        if inter_delay < 0:
            raise ValueError(f"inter_delay ({inter_delay}) must not be negative")
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
    def register_name(self) -> str:
        """
        Name that will be used to register this parameter in a dataset
        By default, this returns ``full_name`` or the value of the
        ``register_name`` argument if it was passed at initialization.
        """
        return self._register_name or self.full_name

    @property
    def instrument(self) -> InstrumentBase | None:
        """
        Return the first instrument that this parameter is bound to.
        E.g if this is bound to a channel it will return the channel
        and not the instrument that the channel is bound too. Use
        :meth:`root_instrument` to get the real instrument.
        """
        return self._instrument

    @property
    def root_instrument(self) -> InstrumentBase | None:
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

    def set_to(
        self, value: ParamDataType, allow_changes: bool = False
    ) -> _SetParamContext:
        """
        Use a context manager to temporarily set a parameter to a value. By
        default, the parameter value cannot be changed inside the context.
        This may be overridden with ``allow_changes=True``.

        Examples:
            >>> from qcodes.parameters import Parameter
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
        context_manager = _SetParamContext(self, value, allow_changes=allow_changes)
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
    def name_parts(self) -> list[str]:
        """
        List of the parts that make up the full name of this parameter
        """
        if self.instrument is not None:
            name_parts = getattr(self.instrument, "name_parts", [])
            if name_parts == []:
                # add fallback for the case where someone has bound
                # the parameter to something that is not an instrument
                # but perhaps it has a name anyway?
                name = getattr(self.instrument, "name", None)
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
    def underlying_instrument(self) -> InstrumentBase | None:
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
    def abstract(self) -> bool | None:
        return self._abstract


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

    def __init__(self, parameter: ParameterBase):
        self.parameter = parameter

    delegate_attr_objects: ClassVar[list[str]] = ["parameter"]
    omit_delegate_attrs: ClassVar[list[str]] = ["set"]

    def get(self) -> ParamDataType:
        """
        Return latest value if time since get was less than
        `max_val_age`, otherwise perform `get()` and
        return result. A `get()` will also be performed if the
        parameter never has been captured.

        It is recommended to use ``parameter.cache.get()`` instead.
        """
        return self.parameter.cache.get()

    def get_timestamp(self) -> datetime | None:
        """
        Return the age of the latest parameter value.

        It is recommended to use ``parameter.cache.timestamp`` instead.
        """
        return self.cache.timestamp

    def get_raw_value(self) -> ParamRawDataType | None:
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
