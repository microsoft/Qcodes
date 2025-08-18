"""
Provides `SetCacheValueOnResetParameterMixin`, a mixin that ensures QCoDeS parameters
maintain a correct cache value after a group-triggered event such as an instrument reset.

This is useful for write-only parameters whose values change as a side effect of a reset
operation and cannot be queried via `get_cmd`. Instead of polling, the value is
resynchronized based on a predefined fallback and coordinated group events.

Internally, this mixin uses `GroupRegistryParameterMixin` to register callbacks
for one or more group names. When a group is triggered, it updates the cache value
accordingly.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, ClassVar, cast

from qcodes.parameters import Parameter, ParameterBase

from .parameter_mixin_group_registry import GroupRegistryParameterMixin

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable


class SetCacheValueOnResetParameterMixin(GroupRegistryParameterMixin):
    """
    Mixin to update a parameter's cache after a coordinated group-triggered event (e.g., instrument reset).

    Intended for write-only parameters (i.e., those without a `get_cmd`), this mixin registers
    a callback that resets the parameter's cache to a known fallback value when a group is triggered.

    This is especially useful for maintaining software consistency in cases where an instrument
    may reset its internal state, and the parameter cannot be re-read from hardware.

    Parser Handling
    ---------------
    - `get_cmd` must be omitted or set to False.
    - Supplying `get_parser` is not allowed and will raise a `TypeError`.
    - If a `set_parser` is provided, a synthetic `get_parser` is automatically created
      to return the current cached value. This ensures that `get()` still returns the expected value.

    Parameters:
    -----------
    cache_value_after_reset : Any, optional
        Value that the parameter's cache will be reset to when the group is triggered.

    Attributes:
    -----------
    cache_value_after_reset : Optional[Any]
        Value used to update the cache after reset.
    group_names : list[str]
        Names of the reset groups this parameter participates in.

    Raises:
    -------
    TypeError
        If `get_cmd` is provided.
        If `get_parser` is supplied.

    Notes:
    ------
    - Relies on `GroupRegistryParameterMixin` to register `reset_cache_value` with one or more groups.
    - Triggering a group will invoke **all** registered callbacks; choose group names carefully.
    - A warning is issued if `cache_value_after_reset` is not provided.
    - If no `group_names` are specified, the callback is still registered but may not be triggered.

    See Also:
    ---------
    GroupRegistryParameterMixin
        Provides group registration and event triggering infrastructure.

    """

    _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [
        Parameter,
    ]
    _INCOMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = []

    _UNSET = object()

    def __init__(
        self,
        *args: Any,
        cache_value_after_reset: Any | None = _UNSET,
        **kwargs: Any,
    ) -> None:
        self.cache_value_after_reset: Any | None = kwargs.pop(
            "cache_value_after_reset", cache_value_after_reset
        )

        get_cmd = kwargs.get("get_cmd", None)
        if get_cmd not in (None, False):
            raise TypeError(
                f"{self.__class__.__name__} is intended for parameters, "
                f"without 'get_cmd'."
            )

        if "get_parser" in kwargs:
            raise TypeError("Supplying 'get_parser' is not allowed for this mixin.")

        set_parser = kwargs.get("set_parser", None)

        super().__init__(*args, callback=self.reset_cache_value, **kwargs)

        if self.cache_value_after_reset is SetCacheValueOnResetParameterMixin._UNSET:
            message = f"cache_value_after_reset for parameter '{cast('ParameterBase', self).name}' is not set."
            log.warning(message)
            warnings.warn(message, UserWarning)

        if set_parser is not None:
            self.get_parser: Callable[..., Any] | None = lambda _ignored_value: cast(
                "ParameterBase", self
            ).cache.get(get_if_invalid=False)

    def reset_cache_value(self) -> None:
        """
        Update the parameter's cache to `cache_value_after_reset`.

        This method is registered to group(s) during initialization.
        It is automatically called when the corresponding group is triggered
        via `trigger_group(...)`.
        """
        cast("ParameterBase", self).cache.set(self.cache_value_after_reset)
