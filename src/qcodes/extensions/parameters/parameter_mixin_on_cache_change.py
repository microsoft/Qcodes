import logging
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Optional, cast
from qcodes.parameters import ParameterBase, Parameter
from .parameter_mixin import ParameterMixin

log = logging.getLogger(__name__)


class OnCacheChangeCallback(Protocol):
    """
    Protocol defining the callback signature for cache changes.

    The callback is invoked when the parameter's cache value changes.
    """

    def __call__(
        self,
        *,
        value_old: Any,
        value_new: Any,
        raw_value_old: Any,
        raw_value_new: Any
    ) -> None:
        pass


class OnCacheChangeParameterMixin(ParameterMixin):
    """
    A mixin that adds on_cache_change functionality to QCoDeS Parameters.

    Attributes:
        on_cache_change: Optional callback invoked when the cached value changes.
    """

    _COMPATIBLE_BASES: List[Type[ParameterBase]] = [
        Parameter,
    ]
    _INCOMPATIBLE_BASES: List[Type[ParameterBase]] = []

    def __init__(
        self,
        *args: Any,
        on_cache_change: Optional[OnCacheChangeCallback] = None,
        **kwargs: Any
    ) -> None:
        self.on_cache_change = kwargs.pop('on_cache_change', on_cache_change)
        super().__init__(*args, **kwargs)
        self._wrap_cache_update_method()

    @property
    def on_cache_change(self) -> Optional[OnCacheChangeCallback]:
        """
        Get the on_cache_change callback.

        Returns:
            The currently set callback or None.
        """
        return self._on_cache_change_callback

    @on_cache_change.setter
    def on_cache_change(self, callback: Optional[OnCacheChangeCallback]) -> None:
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
        parameter = cast(ParameterBase, self)
        original_update_cache = parameter.cache._update_with

        @wraps(original_update_cache)
        def wrapped_cache_update(
            *,
            value: Any,
            raw_value: Any,
            timestamp: Optional[datetime] = None
        ) -> None:
            raw_value_old = parameter.cache.raw_value
            value_old = parameter.cache.get(get_if_invalid=False)

            original_update_cache(value=value, raw_value=raw_value,
                                  timestamp=timestamp)

            raw_value_new = parameter.cache.raw_value
            value_new = parameter.cache.get(get_if_invalid=False)

            if (value_old != value_new) or (raw_value_old != raw_value_new):
                self._handle_on_cache_change(
                    value_old=value_old,
                    value_new=value_new,
                    raw_value_old=raw_value_old,
                    raw_value_new=raw_value_new,
                )

        parameter.cache._update_with = wrapped_cache_update # type: ignore[method-assign]

    def _handle_on_cache_change(
        self,
        *,
        value_old: Any,
        value_new: Any,
        raw_value_old: Any,
        raw_value_new: Any
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

