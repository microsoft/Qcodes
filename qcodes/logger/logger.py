"""
This module defines functions to setup the logging of QCoDeS.
Calling :func:`start_all_logging` will setup logging according to
the default configuration.

"""

import io
import logging
# logging.handlers is not imported by logging. This extra import is necessary
import logging.handlers

import os
from pathlib import Path
from collections import OrderedDict
from contextlib import contextmanager
from copy import copy

from typing import Optional, Union, Sequence

import qcodes as qc

log = logging.getLogger(__name__)

LevelType = Union[int, str]

LOGGING_DIR = "logs"
LOGGING_SEPARATOR = ' ¦ '
""":data:`LOGGING_SEPARATOR` defines the str used to separate parts of the log
 message.
"""
HISTORY_LOG_NAME = "command_history.log"
PYTHON_LOG_NAME = 'qcodes.log'
QCODES_USER_PATH_ENV = 'QCODES_USER_PATH'

FORMAT_STRING_DICT = OrderedDict([
    ('asctime', 's'),
    ('name', 's'),
    ('levelname', 's'),
    ('module', 's'),
    ('funcName', 's'),
    ('lineno', 'd'),
    ('message', 's')])
""":data:`FORMAT_STRING_DICT` defines the format used in logging messages.
"""


# The handler of the root logger that get set up by `start_logger` are globals
# for this modules scope, as it is intended to only use a single file and
# console hander.
console_handler: Optional[logging.Handler] = None
file_handler: Optional[logging.Handler] = None


def get_formatter() -> logging.Formatter:
    """
    Returns :class:`logging.Formatter` according to
    :data:`FORMAT_STRING_DICT`
    """
    format_string_items = [f'%({name}){fmt}'
                           for name, fmt in FORMAT_STRING_DICT.items()]
    format_string = LOGGING_SEPARATOR.join(format_string_items)
    return logging.Formatter(format_string)


def get_console_handler() -> Optional[logging.Handler]:
    """
    Get handle that prints messages from the root logger to the console.
    Returns ``None`` if :func:`start_logger` has not been called.
    """
    global console_handler
    return console_handler


def get_file_handler() -> Optional[logging.Handler]:
    """
    Get a handle that streams messages from the root logger to the qcodes log
    file. To setup call :func:`start_logger`.
    Returns ``None`` if :func:`start_logger` has not been called.
    """
    global file_handler
    return file_handler


def get_level_name(level: Union[str, int]) -> str:
    """
    Get a logging level name from either a logging level code or logging level
    name. Will return the output of :func:`logging.getLevelName` if called with
    an int. If called with a str it will return the str supplied.
    """
    if isinstance(level, str):
        return level
    elif isinstance(level, int):
        return logging.getLevelName(level)
    else:
        raise RuntimeError('get_level_name: '
                           f'Cannot to convert level {level} of type '
                           f'{type(level)} to logging level name. Need '
                           'string or int.')


def get_level_code(level: Union[str, int]) -> int:
    """
    Get a logging level code from either a logging level string or a logging
    level code. Will return the output of :func:`logging.getLevelName` if
    called with a str. If called with an int it will return the int supplied.
    """
    if isinstance(level, int):
        return level
    elif isinstance(level, str):
        # It is possible to get the level code from the
        # `getLevelName` call due to backwards compatibillity to an earlier
        # bug:
        # >>> import logging
        # >>> print(logging.getLevelName('DEBUG'))
        return logging.getLevelName(level)
    else:
        raise RuntimeError('get_level_code: '
                           f'Cannot to convert level {level} of type '
                           f'{type(level)} to logging level code. Need '
                           'string or int.')


def _get_qcodes_user_path() -> str:
    """
    Get ``~/.qcodes`` path or if defined the path defined in the
    ``QCODES_USER_PATH`` environment variable.

    Returns:
        user_path: path to the user qcodes directory
    """
    path = os.environ.get(QCODES_USER_PATH_ENV,
                          os.path.join(Path.home(), '.qcodes'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_log_file_name() -> str:
    """
    Get the full path to the logfile currently used.
    """
    return os.path.join(_get_qcodes_user_path(),
                        LOGGING_DIR,
                        PYTHON_LOG_NAME)


def start_logger() -> None:
    """
    Start logging of messages passed through the python logging module.
    This sets up logging to a time based logging.
    This means that all logging messages on or above
    ``filelogginglevel`` will be written to `pythonlog.log`
    All logging messages on or above ``consolelogginglevel``
    will be written to stderr.
    ``filelogginglevel`` and ``consolelogginglevel`` are defined in the
    ``qcodesrc.json`` file.

    """
    global console_handler
    global file_handler

    # set loggers to the supplied levels
    for name, level in qc.config.logger.logger_levels.items():
        logging.getLogger(name).setLevel(level)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # remove previously set handlers
    for handler in (console_handler, file_handler):
        if handler is not None:
            handler.close()
            root_logger.removeHandler(handler)

    # add qcodes handlers
    # console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(qc.config.logger.console_level)
    console_handler.setFormatter(get_formatter())
    root_logger.addHandler(console_handler)

    # file
    filename = get_log_file_name()
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    file_handler = logging.handlers.TimedRotatingFileHandler(filename,
                                                             when='midnight')

    file_handler.setLevel(qc.config.logger.file_level)
    file_handler.setFormatter(get_formatter())
    root_logger.addHandler(file_handler)

    # capture any warnings from the warnings module
    logging.captureWarnings(capture=True)

    log.info("QCoDes logger setup completed")


def start_command_history_logger(log_dir: Optional[str] = None) -> None:
    """
    Start logging of the history of the interactive command shell.
    Works only with IPython and Jupyter. Call function again to set new path
    to log file.

    Args:
        log_dir: directory where log shall be stored to. If left out, defaults
            to ``~/.qcodes/logs/command_history.log``
    """
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is None:
        log.warning("Command history can't be saved"
                    " outside of IPython/Jupyter")
        return

    log_dir = log_dir or os.path.join(_get_qcodes_user_path(), LOGGING_DIR)
    filename = os.path.join(log_dir, HISTORY_LOG_NAME)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    ipython.magic("%logstop")
    ipython.magic("%logstart -t -o {} {}".format(filename, "append"))
    log.info("Started logging IPython history")


def start_all_logging() -> None:
    """
    Starts python log module logging and ipython command history logging.
    """
    start_logger()
    start_command_history_logger()


@contextmanager
def handler_level(level: LevelType,
                  handler: Union[logging.Handler,
                                 Sequence[logging.Handler]]):
    """
    Context manager to temporarily change the level of handlers.

    Example:
        >>> with logger.handler_level(level=logging.DEBUG, handler=[h1, h1]):
        >>>     root_logger.debug('this is now visible')

    Args:
        level: level to set the handlers to
        handler: handle or sequence of handlers which to change
    """
    if isinstance(handler, logging.Handler):
        handler = (handler,)
    original_levels = [h.level for h in handler]
    for h in handler:
        h.setLevel(level)
    try:
        yield
    finally:
        for h, original_level in zip(handler, original_levels):
            h.setLevel(original_level)


@contextmanager
def console_level(level: LevelType):
    """
    Context manager to temporarily change the level of the qcodes console
    handler.

    Example:
        >>> with logger.console_level(level=logging.DEBUG):
        >>>     root_logger.debug('this is now visible')

    Args:
        level: level to set the console handler to
    """
    global console_handler
    if console_handler is None:
        raise RuntimeError("Console handler is None. Cannot set the level"
                           " on it")
    with handler_level(level, handler=console_handler):
        yield


class LogCapture:

    """
    Context manager to grab all log messages, optionally
    from a specific logger.

    Example:
        >>> with LogCapture() as logs:
        >>>     code_that_makes_logs(...)
        >>> log_str = logs.value

    """

    def __init__(self, logger=logging.getLogger(),
                 level: Optional[LevelType]=None) -> None:
        self.logger = logger
        self.level = level or logging.DEBUG

        self.stashed_handlers = copy(self.logger.handlers)
        for h in self.stashed_handlers:
            self.logger.removeHandler(h)

    def __enter__(self):
        self.log_capture = io.StringIO()
        self.string_handler = logging.StreamHandler(self.log_capture)
        self.string_handler.setLevel(self.level)
        self.logger.addHandler(self.string_handler)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.logger.removeHandler(self.string_handler)
        self.value = self.log_capture.getvalue()
        self.log_capture.close()

        for h in self.stashed_handlers:
            self.logger.addHandler(h)
