"""
Provides a mixin that allows QCoDeS parameters to register and respond
to group-based callbacks. This is useful for coordinated behavior across
parameters, such as resetting or recalibrating them together.

Classes:
    - GroupRegistryParameterMixin: Enables registration of parameter callbacks
      to named groups for coordinated triggering.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, ClassVar, cast

from qcodes.parameters import Parameter, ParameterBase

from .parameter_mixin import ParameterMixin

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable


class GroupRegistryParameterMixin(ParameterMixin):
    """
    A mixin that enables parameters to register callbacks to named groups
    (registries) for coordinated actions across multiple parameters.

    This is useful for triggering parameter-related logic, such as cache resets
    or recalibration, in a batch when a group event occurs (e.g., an instrument reset).

    Parameters can register their methods to one or more named groups.
    When a group is triggered, all registered callbacks for that group are executed.

    Usage Examples:
        Register a callback to a group:
            GroupRegistryParameterMixin.register_group_callback("my_group", callback)

        Trigger all callbacks for a group:
            GroupRegistryParameterMixin.trigger_group("my_group")

    Attributes:
        _group_registry (dict[str, list[Callable]]): Class-level registry mapping
            group names to lists of callback functions.

    Raises:
        TypeError: If group names are not a list of strings.

    Note:
        This mixin is intended as a reusable mechanism for grouping parameter actions,
        and can be subclassed to implement group-based logic in specific use cases.

    """

    _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [
        Parameter,
    ]
    _INCOMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = []

    _group_registry: ClassVar[dict[str, list[Callable[[], None]]]] = {}

    def __init__(
        self,
        *args: Any,
        group_names: list[str] | None = None,
        callback: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        self.group_names = kwargs.pop("group_names", group_names)
        callback = kwargs.pop("callback", callback)

        super().__init__(*args, **kwargs)

        if self.group_names is None:
            message = f"No group_name(s) provided for parameter '{cast('ParameterBase', self).name}'."
            log.warning(message)
            warnings.warn(message, UserWarning)

        if self.group_names and callback is not None:
            self._register_callback_to_groups(self.group_names, callback)

    @property
    def group_names(self) -> list[str] | None:
        return self._group_names

    @group_names.setter
    def group_names(self, value: list[str] | None) -> None:
        if value is not None:
            if not isinstance(value, list) or not all(
                isinstance(v, str) for v in value
            ):
                raise TypeError("group_names must be a list of strings or None.")
        self._group_names = value

    @classmethod
    def register_group_callback(
        cls, group_name: str, callback: Callable[[], None]
    ) -> None:
        cls._group_registry.setdefault(group_name, []).append(callback)

    @classmethod
    def trigger_group(cls, group_name: str) -> None:
        callbacks = cls._group_registry.get(group_name, [])
        if not callbacks:
            message = f"No callbacks registered for group '{group_name}'."
            log.warning(message)
            warnings.warn(message, UserWarning)
            return
        for callback in callbacks:
            callback()

    def _register_callback_to_groups(
        self, group_names: list[str], callback: Callable[[], None]
    ) -> None:
        for group_name in group_names:
            self.register_group_callback(group_name, callback)
