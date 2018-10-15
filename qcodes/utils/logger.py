import io
import logging

import os
from pathlib import Path
from collections import OrderedDict
from contextlib import contextmanager
from copy import copy

from typing import Optional, List, Union, Sequence

import pandas
from pandas.core.series import Series

import qcodes as qc
from qcodes.instrument.base import InstrumentFilter, InstrumentBase

LevelType = Union[int, str]


log = logging.getLogger(__name__)

LOGGING_DIR = "logs"
LOGGING_SEPARATOR = ' ¦ '
HISTORY_LOG_NAME = "command_history.log"
PYTHON_LOG_NAME = 'qcodes.log'
QCODES_USER_PATH_ENV = 'QCODES_USER_PATH'

FORMAT_STRING_DICT = OrderedDict([
    ('asctime', 's'),
    ('name', 's'),
    ('levelname', 's'),
    ('funcName', 's'),
    ('lineno', 'd'),
    ('message', 's')])
FORMAT_STRING_ITEMS = [f'%({name}){fmt}'
                       for name, fmt in FORMAT_STRING_DICT.items()]

# The handler of the root logger that get set up by `start_logger` are globals
# for this modules scope, as it is intended to only use a single file and
# console hander.
console_handler: Optional[logging.Handler] = None
file_handler: Optional[logging.Handler] = None


def get_console_handler() -> Optional[logging.Handler]:
    """
    Get h that prints messages from the root logger to the console.
    Returns `None` if `start_logger` had not been called.
    """
    global console_handler
    return console_handler


def get_file_handler() -> Optional[logging.Handler]:
    """
    Get h that streams messages from the root logger to the qcdoes log
    file. To setup call `start_logger`.
    Returns `None` if `start_logger` had not been called
    """
    global file_handler
    return file_handler


def get_level_name(level: Union[str, int]) -> str:
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
    if isinstance(level, int):
        return level
    elif isinstance(level, str):
        # seemingly the type is defined incorrectly in the logging module
        # in a fresh ipython session:
        # >>> import logging
        # >>> print(logging.getLevelName('DEBUG'))
        # works as expected from the documentation
        return logging.getLevelName(level)  # type: ignore
    else:
        raise RuntimeError('get_level_code: '
                           f'Cannot to convert level {level} of type '
                           f'{type(level)} to logging level code. Need '
                           'string or int.')

def _get_qcodes_user_path() -> str:
    """
    Get '~/.qcodes' path or if defined the path defined in the QCODES_USER_PATH
    environment varaible.

    Returns:
        user_path: path to the user qcodes directory
    """
    path = os.environ.get(QCODES_USER_PATH_ENV,
                          os.path.join(Path.home(), '.qcodes'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_log_file_name() -> str:
    return os.path.join(_get_qcodes_user_path(),
                        LOGGING_DIR,
                        PYTHON_LOG_NAME)


def start_logger() -> None:
    """
    Logging of messages passed through the python logging module
    This sets up logging to a time based logging.
    This means that all logging messages on or above
    `filelogginglevel` will be written to pythonlog.log
    All logging messages on or above `consolelogginglevel`
    will be written to stderr.
    `filelogginglevel` and `consolelogginglevel` are defined in the
    qcodesrc.json file.

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

    format_string = LOGGING_SEPARATOR.join(FORMAT_STRING_ITEMS)
    formatter = logging.Formatter(format_string)

    # console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(qc.config.logger.console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # file
    filename = get_log_file_name()
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    file_handler = logging.handlers.TimedRotatingFileHandler(filename,
                                                             when='midnight')
    file_handler.setLevel(qc.config.logger.file_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # capture any warnings from the warnings module
    logging.captureWarnings(capture=True)

    log.info("QCoDes logger setup completed")


def start_command_history_logger(log_dir: Optional[str]=None) -> None:
    """
    logging of the history of the interactive command shell
    works only with ipython. Call function again to set new path to log file.

    Args:
        log_dir: directory where log shall be stored to. If left out, defaults
            to '~/.qcodes/logs/command_history.log'
    """
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is None:
        log.warn("Command history can't be saved outside of IPython/jupyter")
        return

    log_dir = log_dir or os.path.join(_get_qcodes_user_path(), LOGGING_DIR)
    filename = os.path.join(log_dir, HISTORY_LOG_NAME)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    ipython.magic("%logstop")
    ipython.magic("%logstart -t -o {} {}".format(filename, "append"))
    log.info("Started logging IPython history")


def start_all_logging() -> None:
    """
    Starts python log module logging and ipython comand history logging.
    """
    start_logger()
    start_command_history_logger()


def log_to_dataframe(log: List[str]=None,
                     columns: Optional[List[str]]=None,
                     separator: Optional[str]=None) -> pandas.DataFrame:
    """
    Return the provided or default log string as a pandas DataFrame.

    Unless `qcodes.utils.logger.LOGGING_SEPARATOR` or
    `qcodes.utils.logger.FORMAT_STRING...` have been changed using the default
    for the columns and separtor arguments is encouraged.

    Lines starting with a digit are ignored. In the log setup of `start_logging`
    Traceback messages are also logged. These start with a digit.

    Args:
        log: log content
        columns: column headers for the returned dataframe, defaults to
            columns used by handlers set up by `start_logger`.
        separator: seperator of the logfile to seperate the columns, defaults to
            separtor used by handlers set up by `start_logger`.

    Retruns:
        Pandas DataFrame containing the log content.
    """
    separator = separator or LOGGING_SEPARATOR
    columns = columns or list(FORMAT_STRING_DICT.keys())
    # note: if we used commas as separators, pandas read_csv
    # could be faster than this string comprehension

    split_cont = [line.split(separator) for line in log
                  if line[0].isdigit()]  # avoid tracebacks
    dataframe = pandas.DataFrame(split_cont, columns=columns)

    return dataframe


def logfile_to_dataframe(logfile: Optional[str]=None,
                         columns: Optional[List[str]]=None,
                         separator: Optional[str]=None) -> pandas.DataFrame:
    """
    Return the provided or default logfile as a pandas DataFrame.

    Unless `qcodes.utils.logger.LOGGING_SEPARATOR` or
    `qcodes.utils.logger.FORMAT_STRING...` have been changed using the default
    for the columns and separtor arguments is encouraged.

    Lines starting with a digit are ignored. In the log setup of `start_logging`
    Traceback messages are also logged. These start with a digit.

    Args:
        logfile: name of the logfile; defaults to current default log file.
        columns: column headers for the returned dataframe, defaults to
            columns used by handlers set up by `start_logger`.
        separator: seperator of the logfile to seperate the columns, defaults to
            separtor used by handlers set up by `start_logger`.

    Retruns:
        Pandas DataFrame containing the logfile content.
    """
    logfile = logfile or get_log_file_name()
    with open(logfile) as f:
        raw_cont = f.readlines()

    return log_to_dataframe(raw_cont, columns, separator)


def time_difference(firsttimes: Series,
                    secondtimes: Series,
                    use_first_series_labels: bool=True) -> Series:
    """
    Calculate the time differences between two series
    containing time stamp strings as their values.

    Args:
        firsttimes: The oldest times
        secondtimes: The youngest times
        use_first_series_labels: If true, the returned Series
            has the same labels as firsttimes. Else it has
            the labels of secondtimes

    Returns:
        A Series with float values of the time difference (s)
    """

    if ',' in firsttimes.iloc[0]:
        nfirsttimes = firsttimes.str.replace(',', '.')
    else:
        nfirsttimes = firsttimes

    if ',' in secondtimes.iloc[0]:
        nsecondtimes = secondtimes.str.replace(',', '.')
    else:
        nsecondtimes = secondtimes

    t0s = nfirsttimes.astype("datetime64[ns]")
    t1s = nsecondtimes.astype("datetime64[ns]")
    timedeltas = (t1s.values - t0s.values).astype('float')*1e-9

    if use_first_series_labels:
        output = pandas.Series(timedeltas, index=nfirsttimes.index)
    else:
        output = pandas.Series(timedeltas, index=nsecondtimes.index)

    return output


@contextmanager
def handler_level(level: LevelType,
                  handler:Union[logging.Handler,
                                 Sequence[logging.Handler]]):
    """
    Context manager to temporarily change the level of handlers.
    Example:
        >>> with logger.handler_level(level=logging.DEBUG, handler=[h1, h1]):
        >>>     root_logger.debug('this is now visible)

    Args:
        level: level to set the handlers to
        handler: single or sequence of handlers which to change
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
        >>>     root_logger.debug('this is now visible)

    Args:
        level: level to set the console handler to
    """
    global console_handler
    with handler_level(level, handler=console_handler):
        yield


@contextmanager
def filter_instrument(instrument: Union[InstrumentBase,
                                        Sequence[InstrumentBase]],
                      handler: Optional[
                          Union[logging.Handler,
                                Sequence[logging.Handler]]]=None,
                      level: Optional[LevelType]=None):
    """
    Context manager that adds a filter that only enables the log messages of
    the supplied instruments to pass.
    Example:
        >>> h1, h2 = logger.get_console_handler(), logger.get_file_handler()
        >>> with logger.filter_instruments((qdac, dmm2), handler=[h1, h1]):
        >>>     qdac.ch01(1)  # logged
        >>>     v = dmm2.v()  # not logged

    Args:
        level: level to set the handlers to
        handler: single or sequence of handlers which to change
    """
    if handler is None:
        global console_handler
        handler = (console_handler,)
    if isinstance(handler, logging.Handler):
        handler = (handler,)

    instrument_filter = InstrumentFilter(instrument)
    for h in handler:
        h.addFilter(instrument_filter)
    try:
        if level is not None:
            with handler_level(level, handler):
                yield
        else:
            yield
    finally:
        for h in handler:
            h.removeFilter(instrument_filter)


@contextmanager
def capture_dataframe(level: LevelType=logging.DEBUG,
                      logger: logging.Logger=None):
    """
    Context manager to capture the logs in a `pandas.DataFrame`

    Example:
        >>> with logger.capture_dataframe() as (handler, cb):
        >>>     qdac.ch01(1)  # some commands
        >>>     data_frame = cb()

    Args:
        level: level at which to capture
        handler: single or sequence of handlers which to change

    Returns:
        Tuple of handler that is used to capture the log messages and callback
        that returns the cummulative pandas.DataSet at any given
        point (within the context)
    """
    # get root logger if none is specified.
    logger = logger or logging.getLogger()
    with io.StringIO() as log_capture:
        string_handler = logging.StreamHandler(log_capture)
        string_handler.setLevel(level)
        format_string = LOGGING_SEPARATOR.join(FORMAT_STRING_ITEMS)
        formatter = logging.Formatter(format_string)
        string_handler.setFormatter(formatter)

        logger.addHandler(string_handler)
        try:
            yield string_handler, lambda: log_to_dataframe(
                log_capture.getvalue().splitlines())
        finally:
            logger.removeHandler(string_handler)


class LogCapture():

    """
    context manager to grab all log messages, optionally
    from a specific logger

    usage::

        with LogCapture() as logs:
            code_that_makes_logs(...)
        log_str = logs.value

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
