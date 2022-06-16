# TODO (alexcjohnson) update this with the real duck-typing requirements or
# create an ABC for Parameter and MultiParameter - or just remove this statement
# if everyone is happy to use these classes.

import collections
import collections.abc
import enum
import logging
import os
from copy import copy
from datetime import datetime
from operator import xor
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

import numpy
from typing_extensions import Literal

from qcodes.data.data_array import DataArray
from qcodes.utils.command import Command
from qcodes.utils.helpers import full_class, is_sequence_of, warn_units
from qcodes.utils.metadata import Metadatable
from qcodes.utils.validators import Strings, Validator

from .parameter_base import ParamDataType, ParameterBase, ParamRawDataType
from .sweep_values import SweepFixedValues

if TYPE_CHECKING:
    from qcodes.instrument.base import InstrumentBase


log = logging.getLogger(__name__)


class Parameter(ParameterBase):
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
        get_cmd: Optional[Union[str, Callable[..., Any], Literal[False]]] = None,
        set_cmd: Optional[Union[str, Callable[..., Any], Literal[False]]] = False,
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

        no_instrument_get = not self.gettable and (get_cmd is None or get_cmd is False)
        # TODO: a matching check should be in ParameterBase but
        #   due to the current limited design the ParameterBase cannot
        #   know if this subclass will supply a get_cmd
        #   To work around this a RunTime check is put into get of GetLatest
        #   and into get of _Cache
        if max_val_age is not None and no_instrument_get:
            raise SyntaxError(
                "Must have get method or specify get_cmd when max_val_age is set"
            )

        # Enable set/get methods from get_cmd/set_cmd if given and
        # no `get`/`set` or `get_raw`/`set_raw` methods have been defined
        # in the scope of this class.
        # (previous call to `super().__init__` wraps existing
        # get_raw/set_raw into get/set methods)
        if self.gettable and get_cmd not in (None, False):
            raise TypeError(
                "Supplying a not None or False `get_cmd` to a Parameter"
                " that already implements"
                " get_raw is an error."
            )
        elif not self.gettable and get_cmd is not False:
            if get_cmd is None:
                self.get_raw = lambda: self.cache.raw_value  # type: ignore[assignment]
            else:
                if isinstance(get_cmd, str) and instrument is None:
                    raise TypeError(
                        f"Cannot use a str get_cmd without "
                        f"binding to an instrument. "
                        f"Got: get_cmd {get_cmd} for parameter {name}"
                    )

                exec_str_ask = getattr(instrument, "ask", None) if instrument else None

                self.get_raw = Command(  # type: ignore[assignment]
                    arg_count=0,
                    cmd=get_cmd,
                    exec_str=exec_str_ask,
                )
            self._gettable = True
            self.get = self._wrap_get(self.get_raw)

        if self.settable and set_cmd not in (None, False):
            raise TypeError(
                "Supplying a not None or False `set_cmd` to a Parameter"
                " that already implements"
                " set_raw is an error."
            )
        elif not self.settable and set_cmd is not False:
            if set_cmd is None:
                self.set_raw: Callable[..., Any] = lambda x: x
            else:
                if isinstance(set_cmd, str) and instrument is None:
                    raise TypeError(
                        f"Cannot use a str set_cmd without "
                        f"binding to an instrument. "
                        f"Got: set_cmd {set_cmd} for parameter {name}"
                    )

                exec_str_write = (
                    getattr(instrument, "write", None) if instrument else None
                )
                self.set_raw = Command(
                    arg_count=1, cmd=set_cmd, exec_str=exec_str_write
                )
            self._settable = True
            self.set = self._wrap_set(self.set_raw)

        self._meta_attrs.extend(["label", "unit", "vals"])

        self.label = name if label is None else label
        self._label: str

        self.unit = unit if unit is not None else ""
        self._unitval: str

        if initial_value is not None and initial_cache_value is not None:
            raise SyntaxError(
                "It is not possible to specify both of the "
                "`initial_value` and `initial_cache_value` "
                "keyword arguments."
            )

        if initial_value is not None:
            self.set(initial_value)

        if initial_cache_value is not None:
            self.cache.set(initial_cache_value)

        # generate default docstring
        self.__doc__ = os.linesep.join(
            (
                "Parameter class:",
                "",
                "* `name` %s" % self.name,
                "* `label` %s" % self.label,
                "* `unit` %s" % self.unit,
                "* `vals` %s" % repr(self.vals),
            )
        )

        if docstring is not None:
            self.__doc__ = os.linesep.join((docstring, "", self.__doc__))

    @property
    def unit(self) -> str:
        """
        The unit of measure. Use ``''`` (the empty string)
        for unitless.
        """
        return self._unitval

    @unit.setter
    def unit(self, unit: str) -> None:
        self._unitval = unit

    @property
    def label(self) -> str:
        """
        Label of the data used for plots etc.
        """
        return self._label

    @label.setter
    def label(self, label: str) -> None:
        self._label = label

    def __getitem__(self, keys: Any) -> "SweepFixedValues":
        """
        Slice a Parameter to get a SweepValues object
        to iterate over during a sweep
        """
        return SweepFixedValues(self, keys)

    def increment(self, value: ParamDataType) -> None:
        """Increment the parameter with a value

        Args:
            value: Value to be added to the parameter.
        """
        self.set(self.get() + value)

    def sweep(
        self,
        start: float,
        stop: float,
        step: Optional[float] = None,
        num: Optional[int] = None,
    ) -> SweepFixedValues:
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
        return SweepFixedValues(self, start=start, stop=stop, step=step, num=num)


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
        def __init__(self, parameter: "DelegateParameter"):
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
                raise TypeError(
                    "Cannot get the raw value of a "
                    "DelegateParameter that delegates to None"
                )
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
                raise TypeError(
                    "Cannot get the cache of a "
                    "DelegateParameter that delegates to None"
                )
            return self._parameter._from_raw_value_to_value(
                self._parameter.source.cache.get(get_if_invalid=get_if_invalid)
            )

        def set(self, value: ParamDataType) -> None:
            if self._parameter.source is None:
                raise TypeError(
                    "Cannot set the cache of a DelegateParameter "
                    "that delegates to None"
                )
            self._parameter.validate(value)
            self._parameter.source.cache.set(
                self._parameter._from_value_to_raw_value(value)
            )

        def _set_from_raw_value(self, value: ParamRawDataType) -> None:
            if self._parameter.source is None:
                raise TypeError(
                    "Cannot set the cache of a DelegateParameter "
                    "that delegates to None"
                )
            self._parameter.source.cache.set(value)

        def _update_with(
            self,
            *,
            value: ParamDataType,
            raw_value: ParamRawDataType,
            timestamp: Optional[datetime] = None,
        ) -> None:
            """
            This method is needed for interface consistency with ``._Cache``
            because it is used by ``ParameterBase`` in
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

        self._attr_inherit = {
            "label": {"fixed": False, "value_when_without_source": name},
            "unit": {"fixed": False, "value_when_without_source": ""},
        }

        for attr, attr_props in self._attr_inherit.items():
            if attr in kwargs:
                attr_props["fixed"] = True
            else:
                attr_props["fixed"] = False
            source_attr = getattr(source, attr, attr_props["value_when_without_source"])
            kwargs[attr] = kwargs.get(attr, source_attr)

        for cmd in ("set_cmd", "get_cmd"):
            if cmd in kwargs:
                raise KeyError(
                    f'It is not allowed to set "{cmd}" of a '
                    f"DelegateParameter because the one of the "
                    f"source parameter is supposed to be used."
                )
        if source is None and (
            "initial_cache_value" in kwargs or "initial_value" in kwargs
        ):
            raise KeyError(
                "It is not allowed to supply 'initial_value'"
                " or 'initial_cache_value' "
                "without a source."
            )

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
                attr_val = getattr(
                    source, attr, attr_props["value_when_without_source"]
                )
                setattr(self, attr, attr_val)

    # pylint: disable=method-hidden
    def get_raw(self) -> Any:
        if self.source is None:
            raise TypeError(
                "Cannot get the value of a DelegateParameter "
                "that delegates to a None source."
            )
        return self.source.get()

    # pylint: disable=method-hidden
    def set_raw(self, value: Any) -> None:
        if self.source is None:
            raise TypeError(
                "Cannot set the value of a DelegateParameter "
                "that delegates to a None source."
            )
        self.source(value)

    def snapshot_base(
        self,
        update: Optional[bool] = True,
        params_to_skip_update: Optional[Sequence[str]] = None,
    ) -> Dict[Any, Any]:
        snapshot = super().snapshot_base(
            update=update, params_to_skip_update=params_to_skip_update
        )
        source_parameter_snapshot = (
            None if self.source is None else self.source.snapshot(update=update)
        )
        snapshot.update({"source_parameter": source_parameter_snapshot})
        return snapshot


class ArrayParameter(ParameterBase):
    """
    A gettable parameter that returns an array of values.
    Not necessarily part of an instrument.

    For new driver we strongly recommend using
    :class:`.ParameterWithSetpoints` which is both more flexible and
    significantly easier to use

    Subclasses should define a ``.get_raw`` method, which returns an array.
    This method is automatically wrapped to provide a ``.get`` method.

    :class:`.ArrayParameter` can be used in both a
    :class:`qcodes.dataset.Measurement`
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
            raise AttributeError("ArrayParameters do not support set " "at this time.")

        self._meta_attrs.extend(
            ["setpoint_names", "setpoint_labels", "setpoint_units", "label", "unit"]
        )

        self.label = name if label is None else label
        self.unit = unit if unit is not None else ""

        nt: Type[None] = type(None)

        if not is_sequence_of(shape, int):
            raise ValueError("shapes must be a tuple of ints, not " + repr(shape))
        self.shape = shape

        # require one setpoint per dimension of shape
        sp_shape = (len(shape),)

        sp_types = (
            nt,
            DataArray,
            collections.abc.Sequence,
            collections.abc.Iterator,
            numpy.ndarray,
        )
        if setpoints is not None and not is_sequence_of(
            setpoints, sp_types, shape=sp_shape
        ):
            raise ValueError("setpoints must be a tuple of arrays")
        if setpoint_names is not None and not is_sequence_of(
            setpoint_names, (nt, str), shape=sp_shape
        ):
            raise ValueError("setpoint_names must be a tuple of strings")
        if setpoint_labels is not None and not is_sequence_of(
            setpoint_labels, (nt, str), shape=sp_shape
        ):
            raise ValueError("setpoint_labels must be a tuple of strings")
        if setpoint_units is not None and not is_sequence_of(
            setpoint_units, (nt, str), shape=sp_shape
        ):
            raise ValueError("setpoint_units must be a tuple of strings")

        self.setpoints = setpoints
        self.setpoint_names = setpoint_names
        self.setpoint_labels = setpoint_labels
        self.setpoint_units = setpoint_units

        self.__doc__ = os.linesep.join(
            (
                "Parameter class:",
                "",
                "* `name` %s" % self.name,
                "* `label` %s" % self.label,
                "* `unit` %s" % self.unit,
                "* `shape` %s" % repr(self.shape),
            )
        )

        if docstring is not None:
            self.__doc__ = os.linesep.join((docstring, "", self.__doc__))

        if not self.gettable and not self.settable:
            raise AttributeError("ArrayParameter must have a get, set or both")

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
        if inst_name != "":
            spnames = []
            for spname in self.setpoint_names:
                if spname is not None:
                    spnames.append(inst_name + "_" + spname)
                else:
                    spnames.append(None)
            return tuple(spnames)
        else:
            return self.setpoint_names


def _is_nested_sequence_or_none(
    obj: Any,
    types: Optional[Union[Type[object], Tuple[Type[object], ...]]],
    shapes: Sequence[Sequence[Optional[int]]],
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


class MultiParameter(ParameterBase):
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

        self._meta_attrs.extend(
            [
                "setpoint_names",
                "setpoint_labels",
                "setpoint_units",
                "names",
                "labels",
                "units",
            ]
        )

        if not is_sequence_of(names, str):
            raise ValueError("names must be a tuple of strings, not " + repr(names))

        self.names = tuple(names)
        self.labels = labels if labels is not None else names
        self.units = units if units is not None else [""] * len(names)

        nt: Type[None] = type(None)

        if not is_sequence_of(shapes, int, depth=2) or len(shapes) != len(names):
            raise ValueError(
                "shapes must be a tuple of tuples " "of ints, not " + repr(shapes)
            )
        self.shapes = shapes

        sp_types = (
            nt,
            DataArray,
            collections.abc.Sequence,
            collections.abc.Iterator,
            numpy.ndarray,
        )
        if not _is_nested_sequence_or_none(setpoints, sp_types, shapes):
            raise ValueError("setpoints must be a tuple of tuples of arrays")

        if not _is_nested_sequence_or_none(setpoint_names, (nt, str), shapes):
            raise ValueError("setpoint_names must be a tuple of tuples of strings")

        if not _is_nested_sequence_or_none(setpoint_labels, (nt, str), shapes):
            raise ValueError("setpoint_labels must be a tuple of tuples of strings")

        if not _is_nested_sequence_or_none(setpoint_units, (nt, str), shapes):
            raise ValueError("setpoint_units must be a tuple of tuples of strings")

        self.setpoints = setpoints
        self.setpoint_names = setpoint_names
        self.setpoint_labels = setpoint_labels
        self.setpoint_units = setpoint_units

        self.__doc__ = os.linesep.join(
            (
                "MultiParameter class:",
                "",
                "* `name` %s" % self.name,
                "* `names` %s" % ", ".join(self.names),
                "* `labels` %s" % ", ".join(self.labels),
                "* `units` %s" % ", ".join(self.units),
            )
        )

        if docstring is not None:
            self.__doc__ = os.linesep.join((docstring, "", self.__doc__))

        if not self.gettable and not self.settable:
            raise AttributeError("MultiParameter must have a get, set or both")

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
        if inst_name != "":
            full_sp_names = []
            for sp_group in self.setpoint_names:
                full_sp_names_subgroupd = []
                for spname in sp_group:
                    if spname is not None:
                        full_sp_names_subgroupd.append(inst_name + "_" + spname)
                    else:
                        full_sp_names_subgroupd.append(None)
                full_sp_names.append(tuple(full_sp_names_subgroupd))

            return tuple(full_sp_names)
        else:
            return self.setpoint_names


def combine(
    *parameters: "Parameter",
    name: str,
    label: Optional[str] = None,
    unit: Optional[str] = None,
    units: Optional[str] = None,
    aggregator: Optional[Callable[[Sequence[Any]], Any]] = None,
) -> "CombinedParameter":
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
    multi_par = CombinedParameter(my_parameters, name, label, unit, units, aggregator)
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

    def __init__(
        self,
        parameters: Sequence[Parameter],
        name: str,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        units: Optional[str] = None,
        aggregator: Optional[Callable[..., Any]] = None,
    ) -> None:
        super().__init__()
        # TODO(giulioungaretti)temporary hack
        # starthack
        # this is a dummy parameter
        # that mimicks the api that a normal parameter has
        if not name.isidentifier():
            raise ValueError(
                f"Parameter name must be a valid identifier "
                f"got {name} which is not. Parameter names "
                f"cannot start with a number and "
                f"must not contain spaces or special characters"
            )

        self.parameter = lambda: None
        # mypy will complain that a callable does not have these attributes
        # but you can still create them here.
        self.parameter.full_name = name  # type: ignore[attr-defined]
        self.parameter.name = name  # type: ignore[attr-defined]
        self.parameter.label = label  # type: ignore[attr-defined]

        if units is not None:
            warn_units("CombinedParameter", self)
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
            setattr(self, "aggregate", self._aggregate)

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

    def sweep(self, *array: numpy.ndarray) -> "CombinedParameter":
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
                raise ValueError("Arrays have different number of setpoints")
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
                raise ValueError(
                    _error_msg.format(self.dimensionality, nparray.shape[1])
                )
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

    def snapshot_base(
        self,
        update: Optional[bool] = False,
        params_to_skip_update: Optional[Sequence[str]] = None,
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
        meta_data["__class__"] = full_class(self)
        param = self.parameter
        meta_data["unit"] = param.unit  # type: ignore[attr-defined]
        meta_data["label"] = param.label  # type: ignore[attr-defined]
        meta_data["full_name"] = param.full_name  # type: ignore[attr-defined]
        meta_data["aggregator"] = repr(getattr(self, "f", None))
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

    def __init__(
        self,
        name: str,
        instrument: Optional["InstrumentBase"] = None,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        get_cmd: Optional[Union[str, Callable[..., Any], Literal[False]]] = None,
        set_cmd: Optional[Union[str, Callable[..., Any], Literal[False]]] = None,
        initial_value: Optional[Union[float, str]] = None,
        max_val_age: Optional[float] = None,
        vals: Optional[Validator[Any]] = None,
        docstring: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if vals is None:
            vals = Strings()
        if set_cmd is not None:
            raise RuntimeError("InstrumentRefParameter does not support " "set_cmd.")
        super().__init__(
            name,
            instrument,
            label,
            unit,
            get_cmd,
            set_cmd,
            initial_value,
            max_val_age,
            vals,
            docstring,
            **kwargs,
        )

    # TODO(nulinspiratie) check class works now it's subclassed from Parameter
    def get_instr(self) -> "InstrumentBase":
        """
        Returns the instance of the instrument with the name equal to the
        value of this parameter.
        """
        ref_instrument_name = self.get()
        # note that _instrument refers to the instrument this parameter belongs
        # to, while the ref_instrument_name is the instrument that is the value
        # of this parameter.
        if self._instrument is None:
            raise RuntimeError(
                "InstrumentRefParameter is not bound to " "an instrument."
            )
        return self._instrument.find_instrument(ref_instrument_name)


class ManualParameter(Parameter):
    def __init__(
        self,
        name: str,
        instrument: Optional["InstrumentBase"] = None,
        initial_value: Any = None,
        **kwargs: Any,
    ):
        """
        A simple alias for a parameter that does not have a set or
        a get function. Useful for parameters that do not have a direct
        instrument mapping.
        """
        super().__init__(
            name=name,
            instrument=instrument,
            get_cmd=None,
            set_cmd=None,
            initial_value=initial_value,
            **kwargs,
        )


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

    def __init__(
        self,
        output: Parameter,
        division: Optional[Union[float, Parameter]] = None,
        gain: Optional[Union[float, Parameter]] = None,
        name: Optional[str] = None,
        label: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> None:

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

        super().__init__(name=name, label=self.label, unit=self.unit)

        self._wrapped_parameter = output
        self._wrapped_instrument = getattr(output, "_instrument", None)

        # Set the role, either as divider or amplifier
        # Raise an error if nothing is specified
        is_divider = division is not None
        is_amplifier = gain is not None

        if not xor(is_divider, is_amplifier):
            raise ValueError("Provide only division OR gain")

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
        self.metadata["wrapped_parameter"] = self._wrapped_parameter.name
        if self._wrapped_instrument:
            wrapped_instr_name = getattr(self._wrapped_instrument, "name", None)
            self.metadata["wrapped_instrument"] = wrapped_instr_name

    # Internal handling of the multiplier
    # can be either a Parameter or a scalar
    @property
    def _multiplier(self) -> Parameter:
        if self._multiplier_parameter is None:
            raise RuntimeError(
                "Cannot get multiplier when multiplier " "parameter in unknown."
            )
        return self._multiplier_parameter

    @_multiplier.setter
    def _multiplier(self, multiplier: Union[float, Parameter]) -> None:
        if isinstance(multiplier, Parameter):
            self._multiplier_parameter = multiplier
            multiplier_name = self._multiplier_parameter.name
            self.metadata["variable_multiplier"] = multiplier_name
        else:
            self._multiplier_parameter = ManualParameter(
                "multiplier", initial_value=multiplier
            )
            self.metadata["variable_multiplier"] = False

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
    def gain(self) -> float:  # type: ignore[return]
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
            raise RuntimeError(
                f"ScaledParameter must be either a"
                f"Multiplier or Divisor; got {self.role}"
            )

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
            raise RuntimeError(
                f"ScaledParameter must be either a"
                f"Multiplier or Divisor; got {self.role}"
            )

        self._wrapped_parameter.set(instrument_value)
