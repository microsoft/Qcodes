from __future__ import annotations

import cProfile
import os
from functools import wraps
from time import sleep
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import pytest
from typing_extensions import ParamSpec

from qcodes.metadatable import MetadatableWithName

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import ExceptionInfo


T = TypeVar("T")
P = ParamSpec("P")

def retry_until_does_not_throw(
    exception_class_to_expect: type[Exception] = AssertionError,
    tries: int = 5,
    delay: float = 0.1,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
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
        exception_class_to_expect: Only in case of this exception the
            function will be called again
        tries: Number of times to retry calling the function before giving up
        delay: Delay between retries of the function call, in seconds

    Returns:
        A callable that runs the decorated function until it does not throw
        a given exception
    """

    def retry_until_passes_decorator(func: Callable[P, T]) -> Callable[P, T]:

        @wraps(func)
        def func_retry(*args: P.args, **kwargs: P.kwargs) -> T:
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


def profile(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator that profiles the wrapped function with cProfile.

    It produces a '.prof' file in the current working directory
    that has the name of the executed function.

    Use the 'Stats' class from the 'pstats' module to read the file,
    analyze the profile data (for example, 'p.sort_stats('tottime')'
    where 'p' is an instance of the 'Stats' class), and print the data
    (for example, 'p.print_stats()').
    """

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        profile_filename = func.__name__ + '.prof'
        profiler = cProfile.Profile()
        result = profiler.runcall(func, *args, **kwargs)
        profiler.dump_stats(profile_filename)
        return result
    return wrapper


def error_caused_by(excinfo: ExceptionInfo[Any], cause: str) -> bool:
    """
    Helper function to figure out whether an exception was caused by another
    exception with the message provided.

    Args:
        excinfo: the output of with pytest.raises() as excinfo
        cause: the error message or a substring of it
    """

    exc_repr = excinfo.getrepr()

    chain = getattr(exc_repr, "chain", None)

    if chain is not None:
        # first element of the chain is info about the root exception
        error_location = chain[0][1]
        root_traceback = chain[0][0]
        # the error location is the most reliable data since
        # it only contains the location and the error raised.
        # however there are cases where this is empty
        # in such cases fall back to the traceback
        if error_location is not None:
            return cause in str(error_location)
        else:
            return cause in str(root_traceback)
    else:
        return False


def skip_if_no_fixtures(dbname: str | Path) -> None:
    if not os.path.exists(dbname):
        pytest.skip(
            "No db-file fixtures found. "
            "Make sure that your git clone of qcodes has submodules "
            "This can be done by executing: `git submodule update --init`"
        )


class DummyComponent(MetadatableWithName):

    """Docstring for DummyComponent."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def __str__(self) -> str:
        return self.full_name

    def set(self, value: float) -> float:
        value = value * 2
        return value

    @property
    def short_name(self) -> str:
        return self.name

    @property
    def full_name(self) -> str:
        return self.full_name
