from typing import Callable, Type
from functools import wraps
from time import sleep
import cProfile


def strip_qc(d, keys=('instrument', '__class__')):
    # depending on how you run the tests, __module__ can either
    # have qcodes on the front or not. Just strip it off.
    for key in keys:
        if key in d:
            d[key] = d[key].replace('qcodes.tests.', 'tests.')
    return d


def retry_until_does_not_throw(
        exception_class_to_expect: Type[Exception]=AssertionError,
        tries: int=5,
        delay: float=0.1
) -> Callable:
    """
    Call the decorated function given number of times with given delay between
    the calls until it does not throw an exception of a given class.

    If the function throws an exception of a different class, it gets propagated
    outside (i.e. the function is not called anymore).

    Usage:
        >>  x = False  # let's assume that another thread has access to "x",
                       # and it is going to change "x" to "True" very soon
        >>  @retry_until_does_not_throw() ...
            def assert_x_is_true(): ...
                assert x, "x is still False..." ...
        >>  assert_x_is_true()  # depending on the settings of
                                # "retry_until_does_not_throw", it will keep
                                # calling the function (with breaks in between)
                                # until either it does not throw or
                                # the number of tries is exceeded.

    Args:
        exception_class_to_expect
            Only in case of this exception the function will be called again
        tries
            Number of times to retry calling the function before giving up
        delay
            Delay between retries of the function call, in seconds

    Returns:
        A callable that runs the decorated function until it does not throw
        a given exception
    """
    def retry_until_passes_decorator(func: Callable):

        @wraps(func)
        def func_retry(*args, **kwargs):
            tries_left = tries - 1
            while tries_left > 0:
                try:
                    return func(*args, **kwargs)
                except exception_class_to_expect:
                    tries_left -= 1
                    sleep(delay)
            # the very last attempt to call the function is outside
            # the "try-except" clause, so that the exception can propagate
            # up the call stack
            return func(*args, **kwargs)

        return func_retry

    return retry_until_passes_decorator


def profile(func):
    """
    Decorator that profiles the wrapped function with cProfile.

    It produces a '.prof' file in the current working directory
    that has the name of the executed function.

    Use the 'Stats' class from the 'pstats' module to read the file,
    analyze the profile data (for example, 'p.sort_stats('tottime')'
    where 'p' is an instance of the 'Stats' class), and print the data
    (for example, 'p.print_stats()').
    """
    def wrapper(*args, **kwargs):
        profile_filename = func.__name__ + '.prof'
        profiler = cProfile.Profile()
        result = profiler.runcall(func, *args, **kwargs)
        profiler.dump_stats(profile_filename)
        return result
    return wrapper
