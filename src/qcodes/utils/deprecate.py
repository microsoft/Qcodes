from __future__ import annotations

import sys
import warnings
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class QCoDeSDeprecationWarning(RuntimeWarning):
    """
    A DeprecationWarning used internally in QCoDeS. This
    fixes `DeprecationWarning` being suppressed by default.
    """


def _warn_deprecated_module(module_name: str, replacement: str) -> None:
    """Warn that the module ``module_name`` is a deprecated alias.

    Call this from the body of the deprecated module.

    Args:
        module_name: The ``__name__`` of the deprecated module.
        replacement: The module that the public names should be imported from.

    """
    warnings.warn(
        f"The module {module_name!r} is deprecated and will be removed in a "
        f"future release. Import from {replacement!r} instead.",
        QCoDeSDeprecationWarning,
        stacklevel=2,
    )


class _KeepAttributesShadowingSubmodulesModule(ModuleType):
    """A module whose public attributes are not replaced by same named submodules.

    When a submodule is imported, the import system sets it as an attribute of
    its parent package. If the package exports an object with the same name as
    a (deprecated) submodule, e.g. the ``SR830`` class and the
    ``stanford_research.SR830`` module, importing the submodule would otherwise
    replace the exported object with the submodule.
    """

    def __setattr__(self, name: str, value: object) -> None:
        if (
            isinstance(value, ModuleType)
            and value.__name__ == f"{self.__name__}.{name}"
            and name in self.__dict__
            and not isinstance(self.__dict__[name], ModuleType)
        ):
            return
        super().__setattr__(name, value)


def _keep_attributes_shadowing_submodules(module_name: str) -> None:
    """
    Prevent the import of a submodule from replacing an attribute of the same name
    on the module ``module_name``.

    Use this in a package that exports objects with the same names as deprecated
    module aliases. Call it from the package ``__init__`` before importing the
    exported objects.

    Args:
        module_name: The ``__name__`` of the package.

    """
    sys.modules[module_name].__class__ = _KeepAttributesShadowingSubmodulesModule


def _make_deprecated_typevars_getattr(
    module_name: str,
    deprecated: dict[str, Any],
    fallback: Callable[[str], Any] | None = None,
) -> Callable[[str], Any]:
    """Return a module-level ``__getattr__`` that emits deprecation warnings
    for removed TypeVar / type-alias / other names.

    Args:
        module_name: The ``__name__`` of the calling module.
        deprecated: Mapping of ``{name: object}`` for names that should
            still be importable but trigger a warning.
        fallback: Optional existing ``__getattr__`` to delegate to for
            names not in *deprecated*.

    Returns:
        A ``__getattr__`` function suitable for assignment at module level.

    """

    def __getattr__(name: str) -> Any:
        if name in deprecated:
            warnings.warn(
                f"Importing {name!r} from {module_name!r} is deprecated. "
                f"This name is no longer used and will be removed in a "
                f"future version.",
                QCoDeSDeprecationWarning,
                stacklevel=2,
            )
            return deprecated[name]
        if fallback is not None:
            return fallback(name)
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    return __getattr__
