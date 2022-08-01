from asyncio import iscoroutinefunction
from inspect import signature


def is_function(f: object, arg_count: int, coroutine: bool = False) -> bool:
    """
    Check and require a function that can accept the specified number of
    positional arguments, which either is or is not a coroutine
    type casting "functions" are allowed, but only in the 1-argument form.

    Args:
        f: Function to check.
        arg_count: Number of argument f should accept.
        coroutine: Is a coroutine.

    Return:
        bool: is function and accepts the specified number of arguments.

    """
    if not isinstance(arg_count, int) or arg_count < 0:
        raise TypeError("arg_count must be a non-negative integer")

    if not (callable(f) and bool(coroutine) is iscoroutinefunction(f)):
        return False

    if isinstance(f, type):
        # for type casting functions, eg int, str, float
        # only support the one-parameter form of these,
        # otherwise the user should make an explicit function.
        return arg_count == 1

    try:
        sig = signature(f)
    except ValueError:
        # some built-in functions/methods don't describe themselves to inspect
        # we already know it's a callable and coroutine is correct.
        return True

    try:
        inputs = [0] * arg_count
        sig.bind(*inputs)
        return True
    except TypeError:
        return False
