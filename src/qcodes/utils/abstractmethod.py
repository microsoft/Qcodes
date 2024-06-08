from typing import TYPE_CHECKING, TypeVar

from typing_extensions import ParamSpec

if TYPE_CHECKING:
    from collections.abc import Callable

input = ParamSpec("input")
output = TypeVar("output")


def qcodes_abstractmethod(
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
