import io
import logging
import logging.handlers

import os
from pathlib import Path
from collections import OrderedDict
from contextlib import contextmanager

from typing import Optional, List

import pandas as pd
from pandas.core.series import Series

# TODO: this import here is critical:
# when creating a new config obect this imported refrence will remain pointing
# at the old config object, while imports via 'import qcodes` paired with
# `qcodes.config....` will yield the new config values.
# Also in combination with the config context manger this will not work.
# importing all of qcodes here is not a good solution either as already the
# loading process shall be logged.
from qcodes import config

log = logging.getLogger(__name__)

LOGGING_DIR = "logs"
LOGGING_SEPARATOR = ' ¦ '
HISTORY_LOG_NAME = "command_history.log"
PYTHON_LOG_NAME = 'qcodes.log'
QCODES_USER_PATH_ENV='QCODES_USER_PATH'

FORMAT_STRING_DICT = OrderedDict([
    ('asctime', 's'),
    ('name', 's'),
    ('levelname', 's'),
    ('funcName', 's'),
    ('lineno', 'd'),
    ('message', 's')])
FORMAT_STRING_ITEMS = [f'%({name}){fmt}'
                       for name, fmt in FORMAT_STRING_DICT.items()]

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
    Logging of messages passed throug the python logging module
    This sets up logging to a time based logging.
    This means that all logging messages on or above
    filelogginglevel will be written to pythonlog.log
    All logging messages on or above consolelogginglevel
    will be written to stderr.
    """

    format_string = LOGGING_SEPARATOR.join(FORMAT_STRING_ITEMS)
    formatter = logging.Formatter(format_string)
    try:
        filelogginglevel = config.core.file_loglevel
    except KeyError:
        filelogginglevel = "INFO"
    consolelogginglevel = config.core.loglevel
    ch = logging.StreamHandler()
    ch.setLevel(consolelogginglevel)
    ch.setFormatter(formatter)

    filename = get_log_file_name()
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fh1 = logging.handlers.TimedRotatingFileHandler(filename,
                                                    when='midnight')
    fh1.setLevel(filelogginglevel)
    fh1.setFormatter(formatter)
    logging.basicConfig(handlers=[ch, fh1], level=logging.DEBUG)
    # capture any warnings from the warnings module
    logging.captureWarnings(capture=True)
    log.info("QCoDes python logger setup")


def start_command_history_logger() -> None:
    """
    logging of the history of the interactive command shell
    works only with ipython
    """
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is None:
        log.warn("Command history can't be saved outside of IPython/jupyter")
        return
    ipython.magic("%logstop")
    filename = os.path.join(_get_qcodes_user_path(),
                            LOGGING_DIR,
                            HISTORY_LOG_NAME)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    ipython.magic("%logstart -t -o {} {}".format(filename, "append"))
    log.info("Started logging IPython history")


def start_all_logging() -> None:
    start_logger()
    start_command_history_logger()

def logfile_to_dataframe(logfile: Optional[str]=None,
                         columns: Optional[List[str]]=None,
                         separator: Optional[str]=None) -> pd.DataFrame:
    """
    Return the provided or default logfile as a pandas DataFrame.

    Unless `qcodes.utils.logger.LOGGING_SEPARATOR` or
    `qcodes.utils.logger.FORMAT_STRING...` have been changed using the default
    for the columns and separtor arguments is encouraged.

    Args:
        logfile: name of the logfile; defaults to current default log file.
        columns: column headers for the returned dataframe.
        separator: seperator of the logfile to seperate the columns.

    Retruns:
        Pandas DataFrame containing the logfile content.
    """

    separator = separator or LOGGING_SEPARATOR
    columns = columns or list(FORMAT_STRING_DICT.keys())
    logfile = logfile or get_log_file_name()

    with open(logfile) as f:
        raw_cont = f.readlines()

    # note: if we used commas as separators, pandas read_csv
    # could be faster than this string comprehension

    split_cont = [line.split(separator) for line in raw_cont
                  if line[0].isdigit()]  # avoid tracebacks
    dataframe = pd.DataFrame(split_cont, columns=columns)

    return dataframe


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
        output = pd.Series(timedeltas, index=nfirsttimes.index)
    else:
        output = pd.Series(timedeltas, index=nsecondtimes.index)

    return output


@contextmanager
def log_capture(logger):
    stashed_handlers = logger.handlers[:]
    for handler in stashed_handlers:
        logger.removeHandler(handler)

    log_capture = io.StringIO()
    string_handler = logging.StreamHandler(log_capture)
    string_handler.setLevel(logging.DEBUG)
    logger.addHandler(string_handler)
    try:
        yield
    finally:
        logger.removeHandler(string_handler)
        value = log_capture.getvalue()
        log_capture.close()

        for handler in stashed_handlers:
            logger.addHandler(handler)


class LogCapture():

    """
    context manager to grab all log messages, optionally
    from a specific logger

    usage::

        with LogCapture() as logs:
            code_that_makes_logs(...)
        log_str = logs.value

    """

    def __init__(self, logger=logging.getLogger()):
        self.logger = logger

        self.stashed_handlers = self.logger.handlers[:]
        for handler in self.stashed_handlers:
            self.logger.removeHandler(handler)

    def __enter__(self):
        self.log_capture = io.StringIO()
        self.string_handler = logging.StreamHandler(self.log_capture)
        self.string_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(self.string_handler)
        return self

    def __exit__(self, type, value, tb):
        self.logger.removeHandler(self.string_handler)
        self.value = self.log_capture.getvalue()
        self.log_capture.close()

        for handler in self.stashed_handlers:
            self.logger.addHandler(handler)
