from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def qcodes_abstractmethod[**input, output](
    funcobj: "Callable[input, output]",
) -> "Callable[input, output]":
    """
    A decorator indicating abstract methods.

    This is heavily inspired by the decorator of the same name in
    the ABC standard library. But we make our own version because
    we actually want to allow the class with the abstract method to be
    instantiated and we will use this property to detect if the
    method is abstract and should be overwritten.
    """
    funcobj.__qcodes_is_abstract_method__ = True  # type: ignore[attr-defined]
    return funcobj


if not TYPE_CHECKING:
    from typing import TypeVar

    from typing_extensions import ParamSpec

    from qcodes.utils.deprecate import _make_deprecated_typevars_getattr

    __getattr__ = _make_deprecated_typevars_getattr(
        __name__,
        {
            "input": ParamSpec("input"),
            "output": TypeVar("output"),
        },
    )
