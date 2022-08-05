from typing import Any, Callable


def qcodes_abstractmethod(funcobj: Callable[..., Any]) -> Callable[..., Any]:
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
