from asyncio import iscoroutinefunction
from inspect import getargspec, ismethod
from collections import Iterable
import math
import logging
from datetime import datetime


def is_sequence(obj):
    '''
    is an object a sequence? We do not consider strings to be sequences,
    but note that mappings (dicts) and unordered sequences (sets) ARE
    sequences by this definition.
    '''
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))


def is_function(f, arg_count, coroutine=False):
    '''
    require a function with the specified number of positional arguments
    (and no kwargs) which either is or is not a coroutine
    type casting "functions" are allowed, but only in the 1-argument form
    '''
    if not isinstance(arg_count, int) or arg_count < 0:
        raise TypeError('arg_count must be a non-negative integer')

    if not (callable(f) and bool(coroutine) is iscoroutinefunction(f)):
        return False

    if isinstance(f, type):
        # for type casting functions, eg int, str, float
        # only support the one-parameter form of these,
        # otherwise the user should make an explicit function.
        return arg_count == 1

    argspec = getargspec(f)
    if argspec.varargs:
        # we can't check the arg count if there's a *args parameter
        # so you're on your own at that point
        # unfortunately, the asyncio.coroutine decorator wraps with
        # *args and **kw so we can't count arguments with the old async
        # syntax, only the new syntax.
        return True

    # getargspec includes 'self' in the arg count, even though
    # it's not part of calling the function. So take it out.
    return len(argspec.args) - ismethod(f) == arg_count


# could use numpy.arange here, but
# a) we don't want to require that as a dep so low level
# b) I'd like to be more flexible with the sign of step
def permissive_range(start, stop, step):
    '''
    returns range (as a list of values) with floating point step

    inputs:
        start, stop, step

    always starts at start and moves toward stop,
    regardless of the sign of step
    '''
    signed_step = abs(step) * (1 if stop > start else -1)
    # take off a tiny bit for rounding errors
    step_count = math.ceil((stop - start) / signed_step - 1e-10)
    return [start + i * signed_step for i in range(step_count)]


def wait_secs(finish_datetime):
    delay = (finish_datetime - datetime.now()).total_seconds()
    if delay < 0:
        logging.warning('negative delay {} sec'.format(delay))
        return 0
    return delay


def make_unique(s, existing):
    n = 1
    s_out = s
    existing = set(existing)

    while s_out in existing:
        n += 1
        s_out = '{}_{}'.format(s, n)

    return s_out
