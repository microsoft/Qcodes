import logging
import warnings
from datetime import datetime
from functools import wraps
from datetime import datetime
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Final,
    List,
    Optional,
    Protocol,
    Type,
)

from .cache import _CacheProtocol
from qcodes.parameters import (
    ParameterBase,
    Parameter,
    ParamDataType, 
    ParamRawDataType,
)
from qcodes.instrument import InstrumentBase, Instrument

log = logging.getLogger(__name__)

class ParameterMixin():
    """
    A mixin for extending QCoDeS Parameters with additional functionalities.
    
    This mixin enforces naming conventions and verifies compatibility with
    `ParameterBase` subclasses. The class name must end with "ParameterMixin".
    If multiple mixins are combined, declare `_PARAMETER_MIXIN_CLASSES_COMPATIBLE`
    as True. Each mixin should define a `_COMPATIBLE_BASES` to specify which
    `ParameterBase` subclasses it can extend. If no explicit compatibility
    is declared, warnings are issued. Explicit incompatibilities raise errors.
    
    Attributes:
        _COMPATIBLE_BASES (List[Type[ParameterBase]]):
            List of supported `ParameterBase` subclasses.
        _INCOMPATIBLE_BASES (List[Type[ParameterBase]]):
            List of explicitly incompatible `ParameterBase` subclasses.
        _PARAMETER_MIXIN_CLASSES_COMPATIBLE (bool):
            Indicates if multiple ParameterMixin classes can safely combine.
    
    Examples:
        ```python
        class NewFeatureParameterMixin(ParameterMixin):
            \"""
            Adds NewFeature-related functionality to Parameters.
            \"""
            _COMPATIBLE_BASES: List[Type[ParameterBase]] = [Parameter]
            _INCOMPATIBLE_BASES: List[Type[ParameterBase]] = []
    
        class ABParameterMixin(AParameterMixin, BParameterMixin):
            \"""
            Combine A and B ParameterMixin.
            \"""
            _PARAMETER_MIXIN_CLASSES_COMPATIBLE: Final[bool] = True
            _COMPATIBLE_BASES: List[Type[ParameterBase]] = [Parameter]
            _INCOMPATIBLE_BASES: List[Type[ParameterBase]] = []
        ```
    """

    _COMPATIBLE_BASES: List[Type[ParameterBase]] = []
    _INCOMPATIBLE_BASES: List[Type[ParameterBase]] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if not (issubclass(cls, ParameterBase) or issubclass(cls, ParameterMixin)):
            raise TypeError(
                f"Class {cls.__name__} must inherit from ParameterBase or "
                f"ParameterMixin."
            )

        if '_COMPATIBLE_BASES' in cls.__dict__:
            cls._COMPATIBLE_BASES = list(cls._COMPATIBLE_BASES)
        else:
            cls._COMPATIBLE_BASES = []

        if '_INCOMPATIBLE_BASES' in cls.__dict__:
            cls._INCOMPATIBLE_BASES = list(cls._INCOMPATIBLE_BASES)
        else:
            cls._INCOMPATIBLE_BASES = []

        applied_mixin_leaf_list = cls._get_leaf_classes(
                                        base_type=ParameterMixin,
                                        exclude_base_type=ParameterBase)

        all_applied_mixins = cls._get_mixin_classes(
                                base_type=ParameterMixin,
                                exclude_base_type=ParameterBase)

        apply_to_parameter_base = False
        parameter_base_leaf: Optional[Type[ParameterBase]] = None
        if issubclass(cls, ParameterBase):
            apply_to_parameter_base = True
            parameter_base_leaves = cls._get_leaf_classes(
                                        base_type=ParameterBase,
                                        exclude_base_type=ParameterMixin)

            if len(parameter_base_leaves) != 1:
                raise TypeError(
                    f"Expected exactly one ParameterBase leaf subclass, found "
                    f"{len(parameter_base_leaves)}."
                )
            parameter_base_leaf = parameter_base_leaves[0]
            cls._check_compatibility(
                all_mixins=all_applied_mixins, 
                mixin_leaves=applied_mixin_leaf_list,
                parameter_base_leaf=parameter_base_leaf,
            )

        elif issubclass(cls, ParameterMixin):
            if not cls.__name__.endswith('ParameterMixin'):
                raise ValueError(
                    f"Class name '{cls.__name__}' must end with 'ParameterMixin'."
                )

            if hasattr(cls, '_COMPATIBLE_BASES') and not isinstance(cls._INCOMPATIBLE_BASES, list):
                raise TypeError(
                    f"{cls.__name__} must define _COMPATIBLE_BASES as a list."
                )

            if hasattr(cls, '_INCOMPATIBLE_BASES') and not isinstance(cls._INCOMPATIBLE_BASES, list):
                raise TypeError(
                    f"{cls.__name__} must define _INCOMPATIBLE_BASES as a list."
                )

        multiple_mixin_leaves = len(applied_mixin_leaf_list) > 1
        parameter_mixin_classes_compatible = getattr(cls, '_PARAMETER_MIXIN_CLASSES_COMPATIBLE', False)
        if multiple_mixin_leaves:
            if apply_to_parameter_base:
                raise TypeError(
                    f"Multiple ParameterMixin are applied together with "
                    f"{parameter_base_leaf.__name__ if parameter_base_leaf is not None else 'None'}."
                    f"Combine them into a single ParameterMixin "
                    f"class before combining with a ParameterBase class."
                )
            else:
                if not parameter_mixin_classes_compatible:
                    message = (
                        "Multiple ParameterMixin are combined without a being declared compatible."
                    )
                    log.warning(message)
                    warnings.warn(message, UserWarning)

                if cls._COMPATIBLE_BASES == []:
                    all_compatible_bases_sets = [
                        set(aml._COMPATIBLE_BASES) for aml in applied_mixin_leaf_list if hasattr(aml, '_COMPATIBLE_BASES')
                    ]

                    if all_compatible_bases_sets:
                        common_compatible_bases = list(set.intersection(*all_compatible_bases_sets))
                    else:
                        common_compatible_bases = []

                    if not common_compatible_bases:
                        raise TypeError(
                            f"{cls.__name__} does not have any common compatible ParameterBase classes "
                            f"(_COMPATIBLE_BASES) among its applied ParameterMixin classes."
                        )
                    else:
                        cls._COMPATIBLE_BASES = list(common_compatible_bases)


        if issubclass(cls, ParameterBase):
            cls._update_docstring(all_applied_mixins, parameter_base_leaf)


    @classmethod
    def _check_compatibility(
        cls,
        all_mixins: List[Type["ParameterMixin"]],
        mixin_leaves: List[Type["ParameterMixin"]],
        parameter_base_leaf: Type[ParameterBase],
        
    ) -> None:
        """
        Check compatibility between applied ParameterMixin classes and the ParameterBase subclass.
        
        Only ParameterMixin-to-ParameterBase compatibility is considered:
        - Raise TypeError if any applied mixin or the parameter base class is explicitly incompatible with the other.
        - Issue warnings if no explicit compatibility declaration exists between a mixin leaf and the ParameterBase leaf.
        """
        for mixin in all_mixins:
            mixin_incompatible = set(getattr(mixin, '_INCOMPATIBLE_BASES', []))
            if parameter_base_leaf in mixin_incompatible:
                message = (
                    f"{mixin.__name__} is incompatible with "
                    f"{parameter_base_leaf.__name__}."
                )
                raise TypeError(message)

        for mixin_leaf in mixin_leaves:
            mixin_leaf_compatible = set(getattr(mixin_leaf, '_COMPATIBLE_BASES', []))
            if parameter_base_leaf not in mixin_leaf_compatible:
                message = (
                    f"{mixin_leaf.__name__} is not explicitly compatible with "
                    f"{parameter_base_leaf.__name__}. Compatibility is untested."
                )
                log.warning(message)
                warnings.warn(message, UserWarning)


    @classmethod
    def _update_docstring(
        cls,
        all_applied_mixins: List[Type["ParameterMixin"]],
        parameter_base_leaf: Optional[Type[ParameterBase]],
    ) -> None:
        """
        Update the class docstring with information about applied mixins
        and the base parameter class.

        Args:
            all_applied_mixins: List of applied ParameterMixin classes.
            parameter_base_leaf: The ParameterBase subclass.
        """
        mixin_names = [m.__name__ for m in all_applied_mixins]
        mixin_docs = [m.__doc__ or "" for m in all_applied_mixins]
        base_doc = parameter_base_leaf.__doc__ if parameter_base_leaf else "No documentation available."

        additional_doc = (
            f"This Parameter has been extended by the following ParameterMixins: \n    "
            f"{', '.join(mixin_names) if mixin_names else 'None'}\n"
            f"Base Class: "
            f"{parameter_base_leaf.__name__ if parameter_base_leaf else 'None'}\n"
            f"Base Class Docstring:\n"
            f"{base_doc}\n"
            f"Mixin Docstrings:\n"
            f"{'\n\n'.join(mixin_docs) if mixin_docs else 'None'}"
        )
        original_doc = cls.__doc__ or ""
        cls.__doc__ = (original_doc.strip() + "\n\n" + additional_doc).strip()

    @classmethod
    def _get_leaf_classes(cls, base_type: Type, exclude_base_type: Optional[Type] = None) -> List[Type]:
        """
        Retrieve all leaf classes in the MRO of cls that are subclasses of base_type,
        excluding first MRO entry and any classes that are subclasses of exclude_base_type.

        A leaf class is subclass of base_type that does not have any subclasses within the MRO
        for the specified base type, and is not a subclass of exclude_base_type.

        Args:
            base_type: The base type to filter classes by (e.g., ParameterMixin).
            exclude_base_type: An optional base type to exclude from the results.

        Returns:
            A list of leaf classes that are subclasses of base_type,
            excluding those that are subclasses of exclude_base_type.
        """
        mro = cls.__mro__[1:]

        mixin_classes = [
            base for base in mro
            if issubclass(base, base_type) and base is not base_type
        ]

        if exclude_base_type:
            mixin_classes = [
                base for base in mixin_classes
                if not issubclass(base, exclude_base_type)
            ]

        leaf_classes = []
        for mixin in mixin_classes:
            if not any(
                issubclass(other, mixin) and other is not mixin
                for other in mixin_classes
            ):
                leaf_classes.append(mixin)
        return leaf_classes

    @classmethod
    def _get_mixin_classes(cls, base_type: Type, exclude_base_type: Optional[Type] = None) -> List[Type]:
        """
        Retrieve all classes in the MRO of cls that are subclasses of base_type,
        excluding any classes that are subclasses of exclude_base_type and the first MRO entry.

        Args:
            base_type: The base type to filter classes by (e.g., ParameterMixin).
            exclude_base_type: An optional base type to exclude from the results.

        Returns:
            A list of classes that are subclasses of base_type,
            excluding those that are subclasses of exclude_base_type.
        """
        mro = cls.__mro__[1:]

        mixin_classes = [
            base for base in mro
            if issubclass(base, base_type) and base is not base_type
        ]

        if exclude_base_type:
            mixin_classes = [
                base for base in mixin_classes
                if not issubclass(base, exclude_base_type)
            ]

        return mixin_classes


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

    _COMPATIBLE_BASES: List[Type[ParameterBase]] = [
        Parameter,
    ]
    _INCOMPATIBLE_BASES: List[Type[ParameterBase]] = []

    def __init__(
        self,
        *args: Any,
        dependency_update_method: Optional[Callable[..., Any]] = None,
        dependent_on: Optional[List[str]] = [],
        **kwargs: Any
    ) -> None:
        self.dependency_update_method = kwargs.pop(
                        'dependency_update_method', dependency_update_method)
        self.dependent_on = kwargs.pop('dependent_on', dependent_on)

        super().__init__(*args, **kwargs)
        
        self._dependent_params: List["InterdependentParameterMixin"] = []

        self._register_dependencies()

    @property
    def dependency_update_method(self) -> Optional[Callable[..., Any]]:
        """
        Get the method used to update parameter attributes based on dependencies.

        Returns:
            The currently set update method or None.
        """
        return self._update_method

    @dependency_update_method.setter
    def dependency_update_method(
        self,
        method: Optional[Callable[..., Any]]
    ) -> None:
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
    def dependent_on(self) -> List[str]:
        """
        Get the list of parameter names this parameter depends on.

        Returns:
            The list of dependent parameter names.
        """
        return self._dependent_on

    @dependent_on.setter
    def dependent_on(self, dependencies: Optional[List[str]]) -> None:
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
            instrument = cast(ParameterBase, self).instrument
            dep_param = getattr(instrument, dep_name, None)
            if not isinstance(dep_param, InterdependentParameterMixin):
                raise TypeError(
                    f"Dependent parameter '{dep_name}' must be an instance of "
                    f"InterdependentParameterMixin."
                )
            dep_param.add_dependent_parameter(self)

    def add_dependent_parameter(
        self,
        parameter: "InterdependentParameterMixin"
    ) -> None:
        """
        Add a dependent parameter to this parameter's list.

        Args:
            parameter: The dependent parameter to add.
        """
        if parameter not in self._dependent_params:
            self._dependent_params.append(parameter)

    def _handle_on_cache_change(
        self,
        *,
        value_old: Any,
        value_new: Any,
        raw_value_old: Any,
        raw_value_new: Any
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
            raw_value_new=raw_value_new
        )

        for dep_param in self._dependent_params:
            if dep_param.dependency_update_method:
                dep_param.dependency_update_method()
                cast(ParameterBase, dep_param).get()

        if self.dependency_update_method:
            self.dependency_update_method()


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
