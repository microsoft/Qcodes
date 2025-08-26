"""
Provides `InterdependentParameterMixin`, a QCoDeS mixin that enables one parameter
to react to changes in other parameters it depends on.

This is useful when parameter metadata (e.g., unit, label, validator) must be kept
in sync with the state of other parameters. The dependency mechanism is triggered
via the cache change system provided by `OnCacheChangeParameterMixin`.

Typical usage involves defining dependencies by name and providing a callable
to update internal state when those dependencies change.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, cast

from qcodes.parameters import Parameter, ParameterBase

from .parameter_mixin_on_cache_change import OnCacheChangeParameterMixin

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)


class InterdependentParameterMixin(OnCacheChangeParameterMixin):
    """
    Mixin to define dependencies between QCoDeS parameters.

    Allows a parameter to respond dynamically when other parameters change,
    e.g., updating its label, unit, or validator when dependent values change.

    Dependencies are declared via ``dependent_on`` (a list of parameter names).
    These referenced parameters must also use this mixin. On cache changes,
    they notify dependents via `OnCacheChangeParameterMixin`.

    A user-defined ``dependency_update_method`` is called to apply custom logic
    when dependencies change. This callable gives full control over how the
    parameter reacts—whether adjusting metadata or triggering re-evaluation.

    Attributes:
    -----------
    dependency_update_method : Callable[[], None] | None
        User-provided function called when a dependency changes. Used to
        update metadata or internal state.

    dependent_on : list[str]
        Names of other parameters this one depends on. They must use this mixin.

    Notes:
    ------
    - ``get()`` is called automatically on dependents after updates.
    - Parameters listed in ``dependent_on`` must already be added to the instrument.

    """

    _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [
        Parameter,
    ]
    _INCOMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = []

    _update_method: Callable[[], None] | None

    def __init__(
        self,
        *args: Any,
        dependency_update_method: Callable[[], None] | None = None,
        dependent_on: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        dependency_update_method = kwargs.pop(
            "dependency_update_method", dependency_update_method
        )
        dependent_on = kwargs.pop("dependent_on", dependent_on)

        super().__init__(*args, **kwargs)

        self.dependency_update_method = dependency_update_method
        self.dependent_on = [] if dependent_on is None else list(dependent_on)

        self._dependent_params: list[InterdependentParameterMixin] = []

        self._register_dependencies()

    @property
    def dependency_update_method(self) -> Callable[[], None] | None:
        """
        Get the method used to update parameter attributes based on dependencies.

        Returns:
            The currently set update method or None.

        """
        return self._update_method

    @dependency_update_method.setter
    def dependency_update_method(self, method: Callable[[], None] | None) -> None:
        """
        Set the dependency update method.

        Args:
            method: The method or None.

        Raises:
            TypeError: If method is not callable or None.

        """
        if method is not None and not callable(method):
            raise TypeError("dependency_update_method must be callable or None.")
        self._update_method = method

    @property
    def dependent_on(self) -> list[str]:
        """
        Get the list of parameter names this parameter depends on.

        Returns:
            The list of dependent parameter names.

        """
        return self._dependent_on

    @dependent_on.setter
    def dependent_on(self, dependencies: list[str]) -> None:
        """
        Set the list of dependent parameter names.

        Args:
            dependencies: The new list of dependent parameter names.

        Raises:
            TypeError: If dependencies is not a list of strings.

        """
        if not isinstance(dependencies, list) or not all(
            isinstance(dep, str) for dep in dependencies
        ):
            raise TypeError("dependent_on must be a list of strings.")
        self._dependent_on = dependencies

    def _register_dependencies(self) -> None:
        """
        Register dependencies with other parameters in the same instrument.
        """
        for dep_name in self.dependent_on:
            instrument = cast("ParameterBase", self).instrument
            dep_param = getattr(instrument, dep_name, None)
            if not isinstance(dep_param, InterdependentParameterMixin):
                raise TypeError(
                    f"Dependent parameter '{dep_name}' must be an instance of "
                    f"InterdependentParameterMixin."
                )
            dep_param.add_dependent_parameter(self)

    def add_dependent_parameter(self, parameter: InterdependentParameterMixin) -> None:
        """
        Add a dependent parameter to this parameter's list.

        Args:
            parameter: The dependent parameter to add.

        """
        if parameter not in self._dependent_params:
            self._dependent_params.append(parameter)

    def _handle_on_cache_change(
        self, *, value_old: Any, value_new: Any, raw_value_old: Any, raw_value_new: Any
    ) -> None:
        """
        Extend cache change handling to manage dependencies.

        After updating this parameter's cache, update dependent parameters and
        optionally call the dependency_update_method if available.

        Args:
            value_old: Previous transformed value.
            value_new: New transformed value.
            raw_value_old: Previous raw value.
            raw_value_new: New raw value.

        """
        super()._handle_on_cache_change(
            value_old=value_old,
            value_new=value_new,
            raw_value_old=raw_value_old,
            raw_value_new=raw_value_new,
        )

        for dep_param in self._dependent_params:
            if dep_param.dependency_update_method is not None:
                dep_param.dependency_update_method()
                cast("ParameterBase", dep_param).get()

        if self.dependency_update_method is not None:
            self.dependency_update_method()
