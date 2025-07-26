import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, cast
from qcodes.parameters import ParameterBase, Parameter
from .parameter_mixin import ParameterMixin

log = logging.getLogger(__name__)


class SetCacheValueOnResetParameterMixin(ParameterMixin):
    """
    A mixin to synchronize parameter cache values with the instrument state
    after a reset operation.

    In instruments where certain parameters are only settable and their values
    change when a reset is performed, the software model (QCoDeS Parameter cache) 
    can become out of sync with the actual instrument state.
    This mixin ensures that the cached value of such parameters is updated to reflect
    the instrument's new state by registering their reset methods to specified 
    reset groups.

    Parameters using this mixin can register their `reset_cache_value` method to
    one or more reset groups by `group_name`. This allows handling multiple reset 
    scenarios in a flexible and maintainable manner.

    The instrument's reset method must manually invoke the: 
    `SetCacheValueOnResetParameterMixin.reset_group(group_name)` 
    class method to ensure all registered parameters are reset.

    The mixin will warn you if `cache_value_after_reset` has never been set.

    Additional rules:
    - Supplying a `get_parser` is not allowed and will raise a TypeError.
    - If a `set_parser` is supplied, a default `get_parser` will be assigned which
      returns the parameter's cached value via:
          get_parser = lambda x: self.cache.get(get_if_invalid=False)
      This ensures that the retrieved value always reflects the latest cache state.

    Attributes:
        cache_value_after_reset: The value to set the parameter cache to after
            an instrument reset.
        reset_group_names: A list of reset group names (strings) to which the parameter's
            `reset_cache_value` method should be appended. This allows the parameter 
            to be reset by multiple groups, facilitating different reset scenarios.

    Raises:
        TypeError: If `get_cmd` is supplied or if `get_parser` is supplied.
    """

    _COMPATIBLE_BASES: List[Type[ParameterBase]] = [
        Parameter,
    ]
    _INCOMPATIBLE_BASES: List[Type[ParameterBase]] = []

    _UNSET = object()

    _reset_group_registry: Dict[str, List[Callable]] = {}

    @classmethod
    def register_reset_callback(cls, group_name: str, callback: Callable) -> None:
        if group_name not in cls._reset_group_registry:
            cls._reset_group_registry[group_name] = []
        cls._reset_group_registry[group_name].append(callback)

    @classmethod
    def reset_group(cls, group_name: str) -> None:
        callbacks = cls._reset_group_registry.get(group_name, [])
        if not callbacks:
            message = f"No callbacks registered for reset group '{group_name}'."
            log.warning(message)
            warnings.warn(message, UserWarning)
            return

        for callback in callbacks:
            callback()

    def __init__(
        self,
        *args: Any,
        cache_value_after_reset: Optional[Any] = _UNSET,
        reset_group_names: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:

        self.cache_value_after_reset: Optional[Any] = kwargs.pop(
            'cache_value_after_reset', cache_value_after_reset
        )
        self.reset_group_names = kwargs.pop('reset_group_names', reset_group_names)

        get_cmd = kwargs.get('get_cmd', None)
        if get_cmd not in (None, False):
            raise TypeError(
                f"{self.__class__.__name__} is intended for parameters, "
                f"without 'get_cmd'."
            )

        if 'get_parser' in kwargs:
            raise TypeError("Supplying 'get_parser' is not allowed for this mixin.")

        if 'set_parser' in kwargs:
            set_parser = kwargs['set_parser']
        else:
            set_parser = None

        super().__init__(*args, **kwargs)

        if self.cache_value_after_reset is SetCacheValueOnResetParameterMixin._UNSET:
            message = (
                f"cache_value_after_reset for parameter '{cast(ParameterBase, self).name}' is not set."
            )
            log.warning(message)
            warnings.warn(message, UserWarning)

        if self.reset_group_names is None:
            message = (
                f"No reset_group_name(s) provided for parameter '{cast(ParameterBase, self).name}'."
            )
            log.warning(message)
            warnings.warn(message, UserWarning)

        if set_parser is not None:
            self.get_parser: Callable[..., Any] | None = (
                lambda x: cast(ParameterBase, self).cache.get(get_if_invalid=False)
            )

        if reset_group_names:
            self._register_reset_callbacks(reset_group_names)

    @property
    def reset_group_names(self) -> Optional[List[str]]:
        return self._reset_group_names

    @reset_group_names.setter
    def reset_group_names(self, value: Optional[List[str]]) -> None:
        if value is not None:
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise TypeError("reset_group_names must be a list of strings or None.")
        self._reset_group_names = value

    def reset_cache_value(self) -> None:
        cast(ParameterBase, self).cache.set(self.cache_value_after_reset)

    def _register_reset_callbacks(self, group_names: List[str]) -> None:
        for group_name in group_names:
            SetCacheValueOnResetParameterMixin.register_reset_callback(
                group_name, self.reset_cache_value
            )
