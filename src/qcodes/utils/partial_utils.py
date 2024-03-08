from functools import partial
from typing import Any, Callable


def partial_with_docstring(
    func: Callable[..., Any], docstring: str, **kwargs: Any
) -> Callable[..., Any]:
    """
    We want to have a partial function which will allow us to access the docstring
    through the python built-in help function. This is particularly important
    for client-facing driver methods, whose arguments might not be obvious.

    Consider the follow example why this is needed:

    >>> from functools import partial
    >>> def f():
    >>> ... pass
    >>> g = partial(f)
    >>> g.__doc__ = "bla"
    >>> help(g) # this will print the docstring of partial and not the docstring set above

    Args:
        func: A function that its docstring will be accessed.
        docstring: The docstring of the corresponding function.
        **kwargs: Keyword arguments passed to the function.
    """
    ex = partial(func, **kwargs)

    def inner(*inner_args: Any, **inner_kwargs: Any) -> Any:
        return ex(*inner_args, **inner_kwargs)

    inner.__doc__ = docstring

    return inner
