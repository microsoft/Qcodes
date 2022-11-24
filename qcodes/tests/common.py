import copy
import cProfile
import os
import tempfile
from contextlib import contextmanager
from functools import wraps
from time import sleep
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Hashable,
    Optional,
    Tuple,
    Type,
)

import pytest

import qcodes
from qcodes.configuration import Config, DotDict
from qcodes.metadatable import Metadatable
from qcodes.utils import deprecate

if TYPE_CHECKING:
    from pytest import ExceptionInfo


def strip_qc(d, keys=('instrument', '__class__')):
    # depending on how you run the tests, __module__ can either
    # have qcodes on the front or not. Just strip it off.
    for key in keys:
        if key in d:
            d[key] = d[key].replace('qcodes.tests.', 'tests.')
    return d


def retry_until_does_not_throw(
        exception_class_to_expect: Type[Exception] = AssertionError,
        tries: int = 5,
        delay: float = 0.1
) -> Callable[..., Any]:
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
    def retry_until_passes_decorator(func: Callable[..., Any]):

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


def error_caused_by(excinfo: 'ExceptionInfo[Any]', cause: str) -> bool:
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


def skip_if_no_fixtures(dbname):
    if not os.path.exists(dbname):
        pytest.skip(
            "No db-file fixtures found. "
            "Make sure that your git clone of qcodes has submodules "
            "This can be done by executing: `git submodule update --init`"
        )


class DumyPar(Metadatable):

    """Docstring for DumyPar. """

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.full_name = name

    def __str__(self):
        return self.full_name

    def set(self, value):
        value = value * 2
        return value


@deprecate(reason="Unused internally", alternative="default_config fixture")
@contextmanager
def default_config(user_config: Optional[str] = None):
    """
    Context manager to temporarily establish default config settings.
    This is achieved by overwriting the config paths of the user-,
    environment-, and current directory-config files with the path of the
    config file in the qcodes repository.
    Additionally the current config object `qcodes.config` gets copied and
    reestablished.

    Args:
        user_config: represents the user config file content.
    """
    home_file_name = Config.home_file_name
    schema_home_file_name = Config.schema_home_file_name
    env_file_name = Config.env_file_name
    schema_env_file_name = Config.schema_env_file_name
    cwd_file_name = Config.cwd_file_name
    schema_cwd_file_name = Config.schema_cwd_file_name

    Config.home_file_name = ''
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_name = os.path.join(tmpdirname, 'user_config.json')
        file_name_schema = os.path.join(tmpdirname, 'user_config_schema.json')
        if user_config is not None:
            with open(file_name, 'w') as f:
                f.write(user_config)

        Config.home_file_name = file_name
        Config.schema_home_file_name = file_name_schema
        Config.env_file_name = ''
        Config.schema_env_file_name = ''
        Config.cwd_file_name = ''
        Config.schema_cwd_file_name = ''

        default_config_obj: Optional[DotDict] = copy.\
            deepcopy(qcodes.config.current_config)
        qcodes.config = Config()

        try:
            yield
        finally:
            Config.home_file_name = home_file_name
            Config.schema_home_file_name = schema_home_file_name
            Config.env_file_name = env_file_name
            Config.schema_env_file_name = schema_env_file_name
            Config.cwd_file_name = cwd_file_name
            Config.schema_cwd_file_name = schema_cwd_file_name

            qcodes.config.current_config = default_config_obj


@deprecate(reason="Unused internally", alternative="reset_config_on_exit fixture")
@contextmanager
def reset_config_on_exit() -> Generator[None, None, None]:
    """
    Context manager to clean any modification of the in memory config on exit

    """
    default_config_obj: Optional[DotDict] = copy.deepcopy(
        qcodes.config.current_config
    )

    try:
        yield
    finally:
        qcodes.config.current_config = default_config_obj


def compare_dictionaries(
    dict_1: Dict[Hashable, Any],
    dict_2: Dict[Hashable, Any],
    dict_1_name: Optional[str] = "d1",
    dict_2_name: Optional[str] = "d2",
    path: str = "",
) -> Tuple[bool, str]:
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
        path = old_path + "[%s]" % k
        if k not in dict_2.keys():
            key_err += f"Key {dict_1_name}{path} not in {dict_2_name}\n"
        else:
            if isinstance(dict_1[k], dict) and isinstance(dict_2[k], dict):
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
                        'Value of "{}{}" ("{}", type"{}") not same as\n'
                        '  "{}{}" ("{}", type"{}")\n\n'
                    ).format(
                        dict_1_name,
                        path,
                        dict_1[k],
                        type(dict_1[k]),
                        dict_2_name,
                        path,
                        dict_2[k],
                        type(dict_2[k]),
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
