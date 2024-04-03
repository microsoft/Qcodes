from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .parameter import Parameter

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from .parameter_base import ParamDataType, ParamRawDataType


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
        def __init__(self, parameter: DelegateParameter):
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
        def max_val_age(self) -> float | None:
            if self._parameter.source is None:
                return None
            return self._parameter.source.cache.max_val_age

        @property
        def timestamp(self) -> datetime | None:
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

        def _set_from_raw_value(self, raw_value: ParamRawDataType) -> None:
            if self._parameter.source is None:
                raise TypeError(
                    "Cannot set the cache of a DelegateParameter "
                    "that delegates to None"
                )
            self._parameter.source.cache.set(raw_value)

        def _update_with(
            self,
            *,
            value: ParamDataType,
            raw_value: ParamRawDataType,
            timestamp: datetime | None = None,
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
        source: Parameter | None,
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
    def source(self) -> Parameter | None:
        """
        The source parameter that this :class:`DelegateParameter` is bound to
        or ``None`` if this  :class:`DelegateParameter` is unbound.

        :getter: Returns the current source.
        :setter: Sets the source.
        """
        return self._source

    @source.setter
    def source(self, source: Parameter | None) -> None:
        self._set_properties_from_source(source)
        self._source: Parameter | None = source

    def _set_properties_from_source(self, source: Parameter | None) -> None:
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

    def get_raw(self) -> Any:
        if self.source is None:
            raise TypeError(
                "Cannot get the value of a DelegateParameter "
                "that delegates to a None source."
            )
        return self.source.get()

    def set_raw(self, value: Any) -> None:
        if self.source is None:
            raise TypeError(
                "Cannot set the value of a DelegateParameter "
                "that delegates to a None source."
            )
        self.source(value)

    def snapshot_base(
        self,
        update: bool | None = True,
        params_to_skip_update: Sequence[str] | None = None,
    ) -> dict[Any, Any]:
        snapshot = super().snapshot_base(
            update=update, params_to_skip_update=params_to_skip_update
        )
        source_parameter_snapshot = (
            None if self.source is None else self.source.snapshot(update=update)
        )
        snapshot.update({"source_parameter": source_parameter_snapshot})
        return snapshot

    def validate(self, value: ParamDataType) -> None:
        """
        Validate the supplied value.
        If it has a source parameter, validate the value as well with the source validator.

        Args:
            value: value to validate

        Raises:
            TypeError: If the value is of the wrong type.
            ValueError: If the value is outside the bounds specified by the
               validator.
        """
        super().validate(value)
        if self.source is not None:
            self.source.validate(self._from_value_to_raw_value(value))
