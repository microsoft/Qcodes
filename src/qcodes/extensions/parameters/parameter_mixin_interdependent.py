import logging
from typing import TYPE_CHECKING, Any, ClassVar, Optional, cast

from qcodes.parameters import Parameter, ParameterBase

from .parameter_mixin_on_cache_change import OnCacheChangeParameterMixin

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable


class InterdependentParameterMixin(OnCacheChangeParameterMixin):
    """
    Mixin for handling interdependent parameters in instruments.

    Instruments often have parameters that depend on others, such as
    when a parameter's unit or valid values change based on another
    parameter's value. This mixin manages these dependencies, ensuring
    that the software model stays synchronized with the instrument by
    automatically updating dependent parameters when necessary.
    Dependent referenced in dependent_on must also use this mixin.

    Attributes:
        dependency_update_method (Optional[Callable[..., Any]]):
            Callable to update parameter attributes based on dependencies.
        dependent_on (List[str]):
            Names of parameters this parameter depends on.

    """

    _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = [
        Parameter,
    ]
    _INCOMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = []

    def __init__(
        self,
        *args: Any,
        dependency_update_method: Optional["Callable[..., Any]"] = None,
        dependent_on: Optional[list[str]] = [],
        **kwargs: Any,
    ) -> None:
        self.dependency_update_method = kwargs.pop(
            "dependency_update_method", dependency_update_method
        )
        self.dependent_on = kwargs.pop("dependent_on", dependent_on)

        super().__init__(*args, **kwargs)

        self._dependent_params: list[InterdependentParameterMixin] = []

        self._register_dependencies()

    @property
    def dependency_update_method(self) -> Optional["Callable[..., Any]"]:
        """
        Get the method used to update parameter attributes based on dependencies.

        Returns:
            The currently set update method or None.

        """
        return self._update_method

    @dependency_update_method.setter
    def dependency_update_method(self, method: Optional["Callable[..., Any]"]) -> None:
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
    def dependent_on(self, dependencies: Optional[list[str]]) -> None:
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

    def add_dependent_parameter(
        self, parameter: "InterdependentParameterMixin"
    ) -> None:
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
            if dep_param.dependency_update_method:
                dep_param.dependency_update_method()
                cast("ParameterBase", dep_param).get()

        if self.dependency_update_method:
            self.dependency_update_method()
