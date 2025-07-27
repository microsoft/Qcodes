import logging
import warnings
from typing import TYPE_CHECKING, Any, ClassVar, Optional, cast

from qcodes.parameters import Parameter, ParameterBase

from .parameter_mixin_group_registry import GroupRegistryParameterMixin

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable


class SetCacheValueOnResetParameterMixin(GroupRegistryParameterMixin):
    """
    A mixin that synchronizes a parameter's cache value with the instrument state
    after a reset operation, by utilizing GroupRegistryParameterMixin.

    Intended for parameters that are only settable (no get_cmd) and whose values
    change after instrument resets or other operations, this mixin keeps the
    software cache in sync with the hardware state by updating the parameter cache
    when a group "reset" is triggered.

    How it works:
    - Registers the parameter's `reset_cache_value` method to one or more named groups,
        using `SetCacheValueOnResetParameterMixin.register_group_callback(group_name, ...)`.
    - To trigger a reset (or any coordinated action), call
        `SetCacheValueOnResetParameterMixin.trigger_group(group_name)` (or
        `GroupRegistryParameterMixin.trigger_group(...)`) to invoke **all callbacks registered
        for that group**.
    - This enables coordinated resetting of cache values for many parameters at once,
        **but may also trigger any other actions registered to the same group**.

    Usage pattern:
    1. Instantiate your parameter with `group_names` and `cache_value_after_reset`.
    2. When you want to trigger a "reset", call:
            SetCacheValueOnResetParameterMixin.trigger_group("my_group")
        or
            GroupRegistryParameterMixin.trigger_group("my_group")

    Rules:
    - Supplying a `get_parser` is not allowed and will raise a TypeError.
    - If a `set_parser` is supplied, a default `get_parser` is created to return
        the cached value, so that the parameter value always reflects the cache state.
    - If `cache_value_after_reset` was never set, a warning is issued.
    - If `group_names` is not provided, a warning is issued.

    Attributes:
    cache_value_after_reset: The value to assign to the cache after a group reset.
    group_names: List of group names in which this parameter participates.

    Raises:
    TypeError: If `get_cmd` or `get_parser` is supplied.

    See Also:
    GroupRegistryParameterMixin: Provides the underlying group/callback registration mechanism.

    Notes:
    - **Triggering a group will execute all callbacks registered for that group,**
        not just those for resetting cache values. Use unique or well-documented group
        names to avoid accidental cross-triggering of unrelated logic.

    """

    _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [
        Parameter,
    ]
    _INCOMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = []

    _UNSET = object()

    def __init__(
        self,
        *args: Any,
        cache_value_after_reset: Optional[Any] = _UNSET,
        **kwargs: Any,
    ) -> None:
        self.cache_value_after_reset: Optional[Any] = kwargs.pop(
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
            self.get_parser: Callable[..., Any] | None = lambda x: cast(
                "ParameterBase", self
            ).cache.get(get_if_invalid=False)

    def reset_cache_value(self) -> None:
        """
        The reset method is registered to group(s) and will be called whenever that group is triggered.
        """
        cast("ParameterBase", self).cache.set(self.cache_value_after_reset)
