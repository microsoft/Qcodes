import operator
from asyncio import iscoroutinefunction
from inspect import signature


def is_function(f, arg_count, coroutine=False):
    """
    Check and require a function that can accept the specified number of
    positional arguments, which either is or is not a coroutine
    type casting "functions" are allowed, but only in the 1-argument form

    Args:
        f (callable): function to check
        arg_count (int): number of argument f should accept
        coroutine (bool): is a coroutine. Default: False

    Return:
        bool: is function and accepts the specified number of arguments

    """
    if not isinstance(arg_count, int) or arg_count < 0:
        raise TypeError('arg_count must be a non-negative integer')

    if not (callable(f) and bool(coroutine) is iscoroutinefunction(f)):
        return False

    # shouldn't need to do this, but is_function fails for a
    # DeferredOperations object, as signature implicitly does ==
    # with it.
    if isinstance(f, DeferredOperations):
        return arg_count == 0

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


class DeferredOperations:
    pass


# functional forms not in the operator module, so we get
# the order of arguments correct

def _and(a, b):
    return a and b


def _rand(a, b):
    return b and a


def _or(a, b):
    return a or b


def _ror(a, b):
    return b or a


def _rsub(a, b):
    return b - a


def _rtruediv(a, b):
    return b / a


def _rfloordiv(a, b):
    return b // a


def _rmod(a, b):
    return b % a


def _rpow(a, b):
    return b ** a
