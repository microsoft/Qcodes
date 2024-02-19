from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .parameter_base import ParamDataType, ParameterBase, ParamRawDataType


class _CacheProtocol(Protocol):
    """
    This protocol defines the interface that a Parameter Cache implementation
    must implement. This is currently used for 2 implementations, one in
    ParameterBase and a specialized one in DelegateParameter.
    """

    @property
    def raw_value(self) -> ParamRawDataType:
        ...

    @property
    def timestamp(self) -> datetime | None:
        ...

    @property
    def max_val_age(self) -> float | None:
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

    def _update_with(
        self,
        *,
        value: ParamDataType,
        raw_value: ParamRawDataType,
        timestamp: datetime | None = None,
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

    def __init__(self, parameter: ParameterBase, max_val_age: float | None = None):
        self._parameter = parameter
        self._value: ParamDataType = None
        self._raw_value: ParamRawDataType = None
        self._timestamp: datetime | None = None
        self._max_val_age = max_val_age
        self._marked_valid: bool = False

    @property
    def raw_value(self) -> ParamRawDataType:
        """Raw value of the parameter"""
        return self._raw_value

    @property
    def timestamp(self) -> datetime | None:
        """
        Timestamp of the moment when cache was last updated

        If ``None``, the cache hasn't been updated yet and shall be seen as
        "invalid".
        """
        return self._timestamp

    @property
    def max_val_age(self) -> float | None:
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

    def _update_with(
        self,
        *,
        value: ParamDataType,
        raw_value: ParamRawDataType,
        timestamp: datetime | None = None,
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
        oldest_accepted_timestamp = datetime.now() - timedelta(
            seconds=self._max_val_age
        )
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
        elif get_if_invalid:
            if gettable:
                return self._parameter.get()
            else:
                error_msg = self._construct_error_msg()
                raise RuntimeError(error_msg)
        else:
            return self._value

    def _construct_error_msg(self) -> str:
        if self._timestamp is None:
            error_msg = (
                f"Value of parameter "
                f"{self._parameter.full_name} "
                f"is unknown and the Parameter "
                f"does not have a get command. "
                f"Please set the value before "
                f"attempting to get it."
            )
        elif self._max_val_age is not None:
            # TODO: this check should really be at the time
            #  of setting max_val_age unfortunately this
            #  happens in init before get wrapping is performed.
            error_msg = (
                "`max_val_age` is not supported "
                "for a parameter without get "
                "command."
            )
        else:
            # max_val_age is None and TS is not None but cache is
            # invalid with the current logic that should never
            # happen
            error_msg = (
                "Cannot return cache of a parameter "
                "that does not have a get command "
                "and has an invalid cache"
            )
        return error_msg

    def __call__(self) -> ParamDataType:
        """
        Same as :meth:`get` but always call ``get`` on parameter if the
        cache is not valid
        """
        return self.get(get_if_invalid=True)
