# TODO (alexcjohnson) update this with the real duck-typing requirements or
# create an ABC for Parameter and MultiParameter - or just remove this statement
# if everyone is happy to use these classes.
from __future__ import annotations

import logging
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Literal

from .command import Command
from .parameter_base import ParamDataType, ParameterBase, ParamRawDataType
from .sweep_values import SweepFixedValues

if TYPE_CHECKING:
    from collections.abc import Callable

    from qcodes.instrument.base import InstrumentBase
    from qcodes.logger.instrument_logger import InstrumentLoggerAdapter
    from qcodes.validators import Validator


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
        instrument: InstrumentBase | None = None,
        label: str | None = None,
        unit: str | None = None,
        get_cmd: str | Callable[..., Any] | Literal[False] | None = None,
        set_cmd: str | Callable[..., Any] | Literal[False] | None = False,
        initial_value: float | str | None = None,
        max_val_age: float | None = None,
        vals: Validator[Any] | None = None,
        docstring: str | None = None,
        initial_cache_value: float | str | None = None,
        bind_to_instrument: bool = True,
        **kwargs: Any,
    ) -> None:
        def _get_manual_parameter(self: Parameter) -> ParamRawDataType:
            if self.root_instrument is not None:
                mylogger: InstrumentLoggerAdapter | logging.Logger = (
                    self.root_instrument.log
                )
            else:
                mylogger = log
            mylogger.debug(
                "Getting raw value of parameter: %s as %s",
                self.full_name,
                self.cache.raw_value,
            )
            return self.cache.raw_value

        def _set_manual_parameter(
            self: Parameter, x: ParamRawDataType
        ) -> ParamRawDataType:
            if self.root_instrument is not None:
                mylogger: InstrumentLoggerAdapter | logging.Logger = (
                    self.root_instrument.log
                )
            else:
                mylogger = log
            mylogger.debug(
                "Setting raw value of parameter: %s to %s", self.full_name, x
            )
            self.cache._set_from_raw_value(x)
            return x

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
                # ignore typeerror since mypy does not allow setting a method dynamically
                self.get_raw = MethodType(_get_manual_parameter, self)  # type: ignore[method-assign]
            else:
                if isinstance(get_cmd, str) and instrument is None:
                    raise TypeError(
                        f"Cannot use a str get_cmd without "
                        f"binding to an instrument. "
                        f"Got: get_cmd {get_cmd} for parameter {name}"
                    )

                exec_str_ask = getattr(instrument, "ask", None) if instrument else None
                # TODO get_raw should also be a method here. This should probably be done by wrapping
                # it with MethodType like above
                # ignore typeerror since mypy does not allow setting a method dynamically
                self.get_raw = Command(  # type: ignore[method-assign]
                    arg_count=0,
                    cmd=get_cmd,
                    exec_str=exec_str_ask,
                )
            self._gettable = True
            # mypy resolves the type of self.get_raw to object here.
            # this may be resolvable if Command above is correctly wrapped in MethodType
            self.get = self._wrap_get(self.get_raw)  # type: ignore[arg-type]

        if self.settable and set_cmd not in (None, False):
            raise TypeError(
                "Supplying a not None or False `set_cmd` to a Parameter"
                " that already implements"
                " set_raw is an error."
            )
        elif not self.settable and set_cmd is not False:
            if set_cmd is None:
                # ignore typeerror since mypy does not allow setting a method dynamically
                self.set_raw = MethodType(_set_manual_parameter, self)  # type: ignore[method-assign]
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
                # TODO get_raw should also be a method here. This should probably be done by wrapping
                # it with MethodType like above
                # ignore typeerror since mypy does not allow setting a method dynamically
                self.set_raw = Command(  # type: ignore[assignment]
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

        self._docstring = docstring
        self.__doc__ = self._build__doc__()

    def _build__doc__(self) -> str:
        if len(self.validators) == 0:
            validator_docstrings = ["* `vals` None"]
        else:
            validator_docstrings = [
                f"* `vals` {validator!r}" for validator in self.validators
            ]
        # generate default docstring
        doc = os.linesep.join(
            (
                "Parameter class:",
                "",
                f"* `name` {self.name}",
                f"* `label` {self.label}",
                f"* `unit` {self.unit}",
                *validator_docstrings,
            )
        )
        if self._docstring is not None:
            doc = os.linesep.join((self._docstring, "", doc))

        return doc

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

    def __getitem__(self, keys: Any) -> SweepFixedValues:
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
        step: float | None = None,
        num: int | None = None,
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


class ManualParameter(Parameter):
    def __init__(
        self,
        name: str,
        instrument: InstrumentBase | None = None,
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
