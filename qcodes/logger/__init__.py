"""
The :mod:`qcodes.logger` module provides functionality to enable
logging of errors and debug information from QCoDeS using the default python
:mod:`logging` module. It also logs command history logging when
using IPython/Jupyter.

The module also provides functionality to filter log messages by instrument
and functions to extract log messages to :class:`pandas.DataFrame` s

"""

from .instrument_logger import filter_instrument, get_instrument_logger
from .log_analysis import (
    capture_dataframe,
    log_to_dataframe,
    logfile_to_dataframe,
    time_difference,
)
from .logger import (
    LogCapture,
    console_level,
    flush_telemetry_traces,
    get_console_handler,
    get_file_handler,
    get_level_code,
    get_level_name,
    get_log_file_name,
    handler_level,
    start_all_logging,
    start_command_history_logger,
    start_logger,
)

__all__ = [
    "LogCapture",
    "capture_dataframe",
    "console_level",
    "filter_instrument",
    "flush_telemetry_traces",
    "get_console_handler",
    "get_file_handler",
    "get_instrument_logger",
    "get_level_code",
    "get_level_name",
    "get_log_file_name",
    "handler_level",
    "log_to_dataframe",
    "logfile_to_dataframe",
    "start_all_logging",
    "start_command_history_logger",
    "start_logger",
    "time_difference",
]
