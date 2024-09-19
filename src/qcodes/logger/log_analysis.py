"""
This module defines a number of functions to make it easier to
work with log messages from QCoDeS. Specifically it enables
exports of logs and log files to a :class:`pd.DataFrame`

"""

from __future__ import annotations

import io
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    import numpy as np
    import numpy.typing as npt
    import pandas as pd

from .logger import (
    FORMAT_STRING_DICT,
    LOGGING_SEPARATOR,
    LevelType,
    get_formatter,
    get_log_file_name,
)


def log_to_dataframe(
    log: Sequence[str],
    columns: Sequence[str] | None = None,
    separator: str | None = None,
) -> pd.DataFrame:
    """
    Return the provided or default log string as a :class:`pd.DataFrame`.

    Unless :data:`qcodes.logger.logger.LOGGING_SEPARATOR` or
    :data:`qcodes.logger.logger.FORMAT_STRING_DICT` have been changed using the
    default for the columns and separator arguments is encouraged.

    Lines starting with a digit are ignored. In the log setup of
    :func:`qcodes.logger.logger.start_logger`
    Traceback messages are also logged. These start with a digit.

    Args:
        log: Log content.
        columns: Column headers for the returned dataframe, defaults to
            columns used by handlers set up by
            :func:`qcodes.logger.logger.start_logger`.
        separator: Separator of the log file to separate the columns, defaults
            to separator used by handlers set up by
            :func:`qcodes.logger.logger.start_logger`.

    Returns:
        A :class:`pd.DataFrame` containing the log content.
    """
    import pandas as pd

    separator = separator or LOGGING_SEPARATOR
    columns = columns or list(FORMAT_STRING_DICT.keys())
    # note: if we used commas as separators, pandas read_csv
    # could be faster than this string comprehension

    split_cont = [
        line.split(separator) for line in log if line[0].isdigit()
    ]  # avoid tracebacks
    dataframe = pd.DataFrame(split_cont, columns=list(columns))

    return dataframe


def logfile_to_dataframe(
    logfile: str | None = None,
    columns: Sequence[str] | None = None,
    separator: str | None = None,
) -> pd.DataFrame:
    """
    Return the provided or default logfile as a :class:`pd.DataFrame`.

    Unless :data:`qcodes.logger.logger.LOGGING_SEPARATOR` or
    :data:`qcodes.logger.logger.FORMAT_STRING_DICT` have been changed using
    the default for the columns and separator arguments is encouraged.

    Lines starting with a digit are ignored. In the log setup of
    :func:`qcodes.logger.logger.start_logger`
    Traceback messages are also logged. These start with a digit.

    Args:
        logfile: Name of the logfile, defaults to current default log file.
        columns: Column headers for the returned dataframe, defaults to
            columns used by handlers set up by
            :func:`qcodes.logger.logger.start_logger`.
        separator: Separator of the logfile to separate the columns,
            defaults to separator used by handlers set up by
            :func:`qcodes.logger.logger.start_logger`.


    Returns:
        A :class:`pd.DataFrame` containing the logfile content.
    """
    logfile = logfile or get_log_file_name()
    with open(logfile) as f:
        raw_cont = f.readlines()

    return log_to_dataframe(raw_cont, columns, separator)


def time_difference(
    firsttimes: pd.Series, secondtimes: pd.Series, use_first_series_labels: bool = True
) -> pd.Series:
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
        A :class:`pd.Series`  with float values of the time difference (s)
    """
    import pandas as pd

    if "," in firsttimes.iloc[0]:
        nfirsttimes = firsttimes.str.replace(",", ".")
    else:
        nfirsttimes = firsttimes

    if "," in secondtimes.iloc[0]:
        nsecondtimes = secondtimes.str.replace(",", ".")
    else:
        nsecondtimes = secondtimes

    t0s = nfirsttimes.astype("datetime64[ns]")
    t1s = nsecondtimes.astype("datetime64[ns]")
    t1s_np: npt.NDArray[np.datetime64] = t1s.to_numpy()
    t0s_np: npt.NDArray[np.datetime64] = t0s.to_numpy()
    timedeltas = (t1s_np - t0s_np).astype("float") * 1e-9

    if use_first_series_labels:
        output = pd.Series(timedeltas, index=nfirsttimes.index)
    else:
        output = pd.Series(timedeltas, index=nsecondtimes.index)

    return output


@contextmanager
def capture_dataframe(
    level: LevelType = logging.DEBUG, logger: logging.Logger | None = None
) -> Iterator[tuple[logging.StreamHandler, Callable[[], pd.DataFrame]]]:
    """
    Context manager to capture the logs in a :class:`pd.DataFrame`

    Example:
        >>> with logger.capture_dataframe() as (handler, cb):
        >>>     qdac.ch01(1)  # some commands
        >>>     data_frame = cb()

    Args:
        level: Level at which to capture.
        logger: Logger used to capture the data. Will default to root logger if
            None is supplied.

    Returns:
        Tuple of handler that is used to capture the log messages and callback
        that returns the cumulative :class:`pd.DataFrame` at any given
        point (within the context)
    """
    # get root logger if none is specified.
    logger = logger or logging.getLogger()
    with io.StringIO() as log_capture:
        string_handler = logging.StreamHandler(log_capture)
        string_handler.setLevel(level)
        string_handler.setFormatter(get_formatter())

        logger.addHandler(string_handler)
        try:
            yield (
                string_handler,
                lambda: log_to_dataframe(log_capture.getvalue().splitlines()),
            )
        finally:
            logger.removeHandler(string_handler)
