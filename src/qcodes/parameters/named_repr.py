from __future__ import annotations

from typing import Any


def named_repr(obj: Any) -> str:
    """Enhance the standard repr() with the object's name attribute."""
    s = f"<{obj.__module__}.{type(obj).__name__}: {obj.name!s} at {id(obj)}>"
    return s
