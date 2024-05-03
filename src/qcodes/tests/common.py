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
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from pytest import ExceptionInfo


def strip_qc(
    d: dict[str, Any], keys: Sequence[str] = ("instrument", "__class__")
) -> dict[str, Any]:
    # depending on how you run the tests, __module__ can either
    # have qcodes on the front or not. Just strip it off.
    for key in keys:
        if key in d:
            d[key] = d[key].replace('qcodes.tests.', 'tests.')
    return d

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


def compare_dictionaries(
    dict_1: Mapping[Any, Any],
    dict_2: Mapping[Any, Any],
    dict_1_name: str | None = "d1",
    dict_2_name: str | None = "d2",
    path: str = "",
) -> tuple[bool, str]:
    """
    Compare two dictionaries recursively to find non matching elements.

    Args:
        dict_1: First dictionary to compare.
        dict_2: Second dictionary to compare.
        dict_1_name: Optional name of the first dictionary used in the
                     differences string.
        dict_2_name: Optional name of the second dictionary used in the
                     differences string.

    Returns:
        Tuple: Are the dicts equal and the difference rendered as
               a string.

    """
    err = ""
    key_err = ""
    value_err = ""
    old_path = path
    for k in dict_1.keys():
        path = old_path + f"[{k}]"
        if k not in dict_2.keys():
            key_err += f"Key {dict_1_name}{path} not in {dict_2_name}\n"
        elif isinstance(dict_1[k], dict) and isinstance(dict_2[k], dict):
            err += compare_dictionaries(
                dict_1[k], dict_2[k], dict_1_name, dict_2_name, path
            )[1]
        else:
            match = dict_1[k] == dict_2[k]

            # if values are equal-length numpy arrays, the result of
            # "==" is a bool array, so we need to 'all' it.
            # In any other case "==" returns a bool
            # TODO(alexcjohnson): actually, if *one* is a numpy array
            # and the other is another sequence with the same entries,
            # this will compare them as equal. Do we want this, or should
            # we require exact type match?
            if hasattr(match, "all"):
                match = match.all()

            if not match:
                value_err += (
                    f'Value of "{dict_1_name}{path}" ("{dict_1[k]}", type"{type(dict_1[k])}") not same as\n'
                    f'  "{dict_2_name}{path}" ("{dict_2[k]}", type"{type(dict_2[k])}")\n\n'
                )

    for k in dict_2.keys():
        path = old_path + f"[{k}]"
        if k not in dict_1.keys():
            key_err += f"Key {dict_2_name}{path} not in {dict_1_name}\n"

    dict_differences = key_err + value_err + err
    if len(dict_differences) == 0:
        dicts_equal = True
    else:
        dicts_equal = False
    return dicts_equal, dict_differences
