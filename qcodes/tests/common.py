import os
import tempfile
from typing import Callable, Type, TYPE_CHECKING, Optional
from contextlib import contextmanager
from functools import wraps
from time import sleep
import cProfile
import copy

import qcodes
from qcodes.utils.metadata import Metadatable
from qcodes.configuration import Config, DotDict

from _pytest._code.code import ExceptionChainRepr
if TYPE_CHECKING:
    from _pytest._code.code import ExceptionInfo


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


def error_caused_by(excinfo: 'ExceptionInfo', cause: str) -> bool:
    """
    Helper function to figure out whether an exception was caused by another
    exception with the message provided.

    Args:
        excinfo: the output of with pytest.raises() as excinfo
        cause: the error message or a substring of it
    """

    exc_repr = excinfo.getrepr()
    assert isinstance(exc_repr, ExceptionChainRepr)
    chain = exc_repr.chain
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
