from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class QCoDeSDeprecationWarning(RuntimeWarning):
    """
    A DeprecationWarning used internally in QCoDeS. This
    fixes `DeprecationWarning` being suppressed by default.
    """


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
