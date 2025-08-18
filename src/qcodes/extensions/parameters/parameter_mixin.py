"""
Provides the `ParameterMixin` base class, which allows modular extensions for
QCoDeS parameter classes in a structured way.

Key Features:
-------------
- Ensures naming consistency via enforced naming conventions (e.g., class names must end with "ParameterMixin").
- Provides a framework for checking compatibility between mixins and parameter base classes.
- Supports multiple mixin composition when explicitly marked as compatible.
- Logs warnings or raises errors for invalid mixin combinations or unsupported base classes.

Intended Usage:
---------------
This module is intended to be subclassed to create mixins that encapsulate
additional behavior for QCoDeS parameters.

See Also:
---------
- Other mixins in the `qcodes.extensions.parameters` module

"""

from __future__ import annotations

import logging
import warnings
from typing import Any, ClassVar

from qcodes.parameters import ParameterBase

log = logging.getLogger(__name__)


class ParameterMixin:
    """
    A mixin for extending QCoDeS Parameters with additional functionalities.

    This mixin enforces naming conventions and verifies compatibility with
    subclasses of `ParameterBase`. The class name must end with ``ParameterMixin``.

    If multiple mixins are combined, set the class attribute
    ``_PARAMETER_MIXIN_CLASSES_COMPATIBLE = True`` to indicate compatibility.

    Each mixin should define:

    - ``_COMPATIBLE_BASES``: A list of `ParameterBase` subclasses it supports.
    - ``_INCOMPATIBLE_BASES``: A list of `ParameterBase` subclasses it should not be used with.

    If compatibility is not explicitly defined, warnings are issued.
    If an incompatibility is declared, an error is raised.

    Attributes:
        _COMPATIBLE_BASES (List[type[ParameterBase]]): List of compatible ParameterBase subclasses.
        _INCOMPATIBLE_BASES (List[type[ParameterBase]]): List of explicitly incompatible base classes.
        _PARAMETER_MIXIN_CLASSES_COMPATIBLE (bool): Set to True if this mixin can be combined with others.

    Examples:

    .. code-block:: python

        class NewFeatureParameterMixin(ParameterMixin):
            # Adds NewFeature-related functionality to Parameters.
            _COMPATIBLE_BASES = [Parameter]
            _INCOMPATIBLE_BASES = []

        class ABParameterMixin(AParameterMixin, BParameterMixin):
            # Combines A and B ParameterMixins.
            _PARAMETER_MIXIN_CLASSES_COMPATIBLE = True
            _COMPATIBLE_BASES = [Parameter]
            _INCOMPATIBLE_BASES = []

    """

    _COMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = []
    _INCOMPATIBLE_BASES: ClassVar[list[type[ParameterBase]]] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if "_COMPATIBLE_BASES" not in cls.__dict__:
            cls._COMPATIBLE_BASES = []

        if "_INCOMPATIBLE_BASES" not in cls.__dict__:
            cls._INCOMPATIBLE_BASES = []

        applied_mixin_leaf_list = cls._get_leaf_classes(
            base_type=ParameterMixin, exclude_base_type=ParameterBase
        )

        all_applied_mixins = cls._get_mixin_classes(
            base_type=ParameterMixin, exclude_base_type=ParameterBase
        )

        apply_to_parameter_base = False
        parameter_base_leaf: type[ParameterBase] | None = None
        if issubclass(cls, ParameterBase):
            apply_to_parameter_base = True
            parameter_base_leaves = cls._get_leaf_classes(
                base_type=ParameterBase, exclude_base_type=ParameterMixin
            )

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
            if not cls.__name__.endswith("ParameterMixin"):
                raise ValueError(
                    f"Class name '{cls.__name__}' must end with 'ParameterMixin'."
                )

            if hasattr(cls, "_COMPATIBLE_BASES") and not isinstance(
                cls._COMPATIBLE_BASES, list
            ):
                raise TypeError(
                    f"{cls.__name__} must define _COMPATIBLE_BASES as a list."
                )

            if hasattr(cls, "_INCOMPATIBLE_BASES") and not isinstance(
                cls._INCOMPATIBLE_BASES, list
            ):
                raise TypeError(
                    f"{cls.__name__} must define _INCOMPATIBLE_BASES as a list."
                )

        multiple_mixin_leaves = len(applied_mixin_leaf_list) > 1
        parameter_mixin_classes_compatible = getattr(
            cls, "_PARAMETER_MIXIN_CLASSES_COMPATIBLE", False
        )
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
                    message = "Multiple ParameterMixin are combined without a being declared compatible."
                    log.warning(message)
                    warnings.warn(message, UserWarning)

                if cls._COMPATIBLE_BASES == []:
                    all_compatible_bases_sets = [
                        set(aml._COMPATIBLE_BASES)
                        for aml in applied_mixin_leaf_list
                        if hasattr(aml, "_COMPATIBLE_BASES")
                    ]

                    common_compatible_bases = list(
                        set.intersection(*all_compatible_bases_sets)
                    )

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
        all_mixins: list[type[ParameterMixin]],
        mixin_leaves: list[type[ParameterMixin]],
        parameter_base_leaf: type[ParameterBase],
    ) -> None:
        """
        Check compatibility between applied ParameterMixin classes and the ParameterBase subclass.

        Only ParameterMixin-to-ParameterBase compatibility is considered:
        - Raise TypeError if any applied mixin or the parameter base class is explicitly incompatible with the other.
        - Issue warnings if no explicit compatibility declaration exists between a mixin leaf and the ParameterBase leaf.
        """
        for mixin in all_mixins:
            mixin_incompatible = set(getattr(mixin, "_INCOMPATIBLE_BASES", []))
            if parameter_base_leaf in mixin_incompatible:
                message = (
                    f"{mixin.__name__} is incompatible with "
                    f"{parameter_base_leaf.__name__}."
                )
                raise TypeError(message)

        for mixin_leaf in mixin_leaves:
            mixin_leaf_compatible = set(getattr(mixin_leaf, "_COMPATIBLE_BASES", []))
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
        all_applied_mixins: list[type[ParameterMixin]],
        parameter_base_leaf: type[ParameterBase] | None,
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
        base_doc = (
            parameter_base_leaf.__doc__
            if parameter_base_leaf
            else "No documentation available."
        )

        mixin_docs_text = "\n\n".join(mixin_docs) if mixin_docs else "None"

        additional_doc = (
            f"This Parameter has been extended by the following ParameterMixins: \n    "
            f"{', '.join(mixin_names) if mixin_names else 'None'}\n"
            f"Base Class: "
            f"{parameter_base_leaf.__name__ if parameter_base_leaf else 'None'}\n"
            f"Base Class Docstring:\n"
            f"{base_doc}\n"
            f"Mixin Docstrings:\n"
            f"{mixin_docs_text}"
        )
        original_doc = cls.__doc__ or ""
        cls.__doc__ = (original_doc.strip() + "\n\n" + additional_doc).strip()

    @classmethod
    def _get_leaf_classes(
        cls, base_type: type, exclude_base_type: type | None = None
    ) -> list[type]:
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
            base
            for base in mro
            if issubclass(base, base_type) and base is not base_type
        ]

        if exclude_base_type:
            mixin_classes = [
                base
                for base in mixin_classes
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
    def _get_mixin_classes(
        cls, base_type: type, exclude_base_type: type | None = None
    ) -> list[type]:
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
            base
            for base in mro
            if issubclass(base, base_type) and base is not base_type
        ]

        if exclude_base_type:
            mixin_classes = [
                base
                for base in mixin_classes
                if not issubclass(base, exclude_base_type)
            ]

        return mixin_classes
