"""
Provides `OnCacheChangeParameterMixin`, a mixin that adds support for reacting
to parameter cache updates in QCoDeS.

When the cached value of a parameter changes, a user-defined callback
(`on_cache_change`) can be triggered to execute custom logic..

This mechanism is implemented by wrapping the parameter's internal cache
update logic and detecting changes.
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

from qcodes.parameters import Parameter, ParameterBase

from .parameter_mixin import ParameterMixin

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from datetime import datetime


class OnCacheChangeCallback(Protocol):
    """
    Protocol defining the signature for `on_cache_change` callbacks.

    The callback receives both raw and transformed old/new values.
    This interface allows users to react to changes in cached state.

    Parameters
    ----------
    value_old : Any
        Previous transformed value.
    value_new : Any
        New transformed value.
    raw_value_old : Any
        Previous raw value.
    raw_value_new : Any
        New raw value.
    """

    def __call__(
        self, *, value_old: Any, value_new: Any, raw_value_old: Any, raw_value_new: Any
    ) -> None:
        pass


class OnCacheChangeParameterMixin(ParameterMixin):
    """
    ParameterMixin to react to parameter cache changes via a user-defined callback.

    This mixin monitors updates to the cached value of a parameter.
    If a change is detected (in the transformed or raw value), a user-defined
    `on_cache_change` callback is invoked.

    This is useful when parameters need to trigger external actions or
    propagate changes to dependent systems.

    Attributes:
    -----------
    on_cache_change : Optional[OnCacheChangeCallback]
        A callable that is invoked when the cached value changes.
        It receives old and new values (raw and transformed) as keyword arguments.

    Notes:
    ------
    - The callback is triggered only if either the raw or transformed value changes.
    - This mixin modifies the internal `_update_with` method of the parameter's cache.

    """

    _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [
        Parameter,
    ]
    _INCOMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = []

    def __init__(
        self,
        *args: Any,
        on_cache_change: OnCacheChangeCallback | None = None,
        **kwargs: Any,
    ) -> None:
        self.on_cache_change = kwargs.pop("on_cache_change", on_cache_change)
        super().__init__(*args, **kwargs)
        self._wrap_cache_update_method()

    @property
    def on_cache_change(self) -> OnCacheChangeCallback | None:
        """
        Get the on_cache_change callback.

        Returns:
            The currently set callback or None.

        """
        return self._on_cache_change_callback

    @on_cache_change.setter
    def on_cache_change(self, callback: OnCacheChangeCallback | None) -> None:
        """
        Set the on_cache_change callback.

        Args:
            callback: The callback or None.

        Raises:
            TypeError: If callback is not callable or None.

        """
        if callback is not None and not callable(callback):
            raise TypeError("on_cache_change must be a callable or None")
        self._on_cache_change_callback = callback

    def _wrap_cache_update_method(self) -> None:
        """
        Wrap the parameter's cache update method to include callback logic.
        """
        parameter = cast("ParameterBase", self)
        original_update_cache = parameter.cache._update_with

        @wraps(original_update_cache)
        def wrapped_cache_update(
            *, value: Any, raw_value: Any, timestamp: datetime | None = None
        ) -> None:
            raw_value_old = parameter.cache.raw_value
            value_old = parameter.cache.get(get_if_invalid=False)

            original_update_cache(value=value, raw_value=raw_value, timestamp=timestamp)

            raw_value_new = parameter.cache.raw_value
            value_new = parameter.cache.get(get_if_invalid=False)

            if (value_old != value_new) or (raw_value_old != raw_value_new):
                self._handle_on_cache_change(
                    value_old=value_old,
                    value_new=value_new,
                    raw_value_old=raw_value_old,
                    raw_value_new=raw_value_new,
                )

        parameter.cache._update_with = wrapped_cache_update  # type: ignore[method-assign]

    def _handle_on_cache_change(
        self, *, value_old: Any, value_new: Any, raw_value_old: Any, raw_value_new: Any
    ) -> None:
        """
        Handle cache changes by invoking the on_cache_change callback.

        Args:
            value_old: Previous value.
            value_new: New value.
            raw_value_old: Previous raw value.
            raw_value_new: New raw value.

        """
        if self.on_cache_change:
            self.on_cache_change(
                value_old=value_old,
                value_new=value_new,
                raw_value_old=raw_value_old,
                raw_value_new=raw_value_new,
            )
