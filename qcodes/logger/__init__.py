"""
The :mod:`qcodes.logger` module provides functionality to enable
logging of errors and debug information from QCoDeS using the default python
:mod:`logging` module as well as command history logging when using
IPython/Jupyter.

The module also provides functionality to filter log messages by instrument
and functions to extract log messages to :class:`pandas.DataFrame` s

"""


from .logger import (get_console_handler, get_file_handler, get_level_name,
                     get_level_code, start_logger,
                     start_command_history_logger, start_all_logging,
                     handler_level, console_level, LogCapture)
from .instrument_logger import filter_instrument
from .log_analysis import capture_dataframe

