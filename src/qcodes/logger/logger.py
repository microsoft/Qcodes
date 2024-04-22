"""
This module defines functions to setup the logging of QCoDeS.
Calling :func:`start_all_logging` will setup logging according to
the default configuration.

"""

import io
import json
import logging
import logging.handlers
import os
import platform
import sys
from collections import OrderedDict
from contextlib import contextmanager
from copy import copy
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, Union

from typing_extensions import deprecated

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import TracebackType

import qcodes as qc
from qcodes.utils import (
    QCoDeSDeprecationWarning,
    get_all_installed_package_versions,
    get_qcodes_user_path,
    is_qcodes_installed_editably,
)

AzureLogHandler = Any
Envelope = Any

log: logging.Logger = logging.getLogger(__name__)

LevelType = Union[int, str]

LOGGING_DIR = "logs"
LOGGING_SEPARATOR = ' ¦ '
""":data:`LOGGING_SEPARATOR` defines the str used to separate parts of the log
 message.
"""
HISTORY_LOG_NAME = "command_history.log"
PYTHON_LOG_NAME = 'qcodes.log'

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
telemetry_handler: Optional["AzureLogHandler"] = None


_opencensus_filter = logging.Filter(name="opencensus")
_urllib3_connection_filter = logging.Filter(name="urllib3.connection")
_azure_monitor_opentelemetry_exporter_filter = logging.Filter(
    name="azure.monitor.opentelemetry.exporter"
)


def filter_out_telemetry_log_records(record: logging.LogRecord) -> bool:
    """
    here we filter any message that is likely to be thrown from
    opencensus/opentelemetry so it is not shown in the user console
    """
    return (
        not _opencensus_filter.filter(record)
        and not _urllib3_connection_filter.filter(record)
        and not _azure_monitor_opentelemetry_exporter_filter
    )


def get_formatter() -> logging.Formatter:
    """
    Returns :class:`logging.Formatter` according to
    :data:`FORMAT_STRING_DICT`
    """
    format_string_items = [f'%({name}){fmt}'
                           for name, fmt in FORMAT_STRING_DICT.items()]
    format_string = LOGGING_SEPARATOR.join(format_string_items)
    return logging.Formatter(format_string)


def get_formatter_for_telemetry() -> logging.Formatter:
    """
    Returns :class:`logging.Formatter` with only name, function name and
    message keywords from FORMAT_STRING_DICT
    """
    format_string_items = [f'%({name}){fmt}'
                           for name, fmt in FORMAT_STRING_DICT.items()
                           if name in ['message', 'name', 'funcName']]
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


def generate_log_file_name() -> str:
    """
    Generates the name of the log file based on process id, date, time and
    PYTHON_LOG_NAME
    """

    pid = str(os.getpid())
    dt_str = datetime.now().strftime("%y%m%d")
    python_log_name = '-'.join([dt_str, pid, PYTHON_LOG_NAME])
    return python_log_name


def get_log_file_name() -> str:
    """
    Get the full path to the log file currently used.
    """
    return os.path.join(get_qcodes_user_path(),
                        LOGGING_DIR,
                        generate_log_file_name())


def flush_telemetry_traces() -> None:
    """
    Flush the traces of the telemetry logger. If telemetry is not enabled, this
    function does nothing.
    """
    global telemetry_handler
    if qc.config.telemetry.enabled and telemetry_handler is not None:
        telemetry_handler.flush()


@deprecated(
    "OpenCensus integration is deprecated. Please use your own telemetry integration as needed, we recommend OpenTelemetry",
    category=QCoDeSDeprecationWarning,
)
def _create_telemetry_handler() -> "AzureLogHandler":
    """
    Configure, create, and return the telemetry handler
    """
    from opencensus.ext.azure.log_exporter import (  # type: ignore[import-not-found]
        AzureLogHandler,
    )
    global telemetry_handler

    # The default_custom_dimensions will appear in the "customDimensions"
    # field in Azure log analytics for every log message alongside any
    # custom dimensions that message may have. All messages additionally come
    # with custom dimensions fileName, level, lineNumber, module, and process
    default_custom_dimensions = {"pythonExecutable": sys.executable}

    class CustomDimensionsFilter(logging.Filter):
        """
        Add application-wide properties to the customDimension field of
        AzureLogHandler records
        """

        def __init__(self, custom_dimensions: dict[str, str]):
            super().__init__()
            self.custom_dimensions = custom_dimensions

        def filter(self, record: logging.LogRecord) -> bool:
            """
            Add the default custom_dimensions into the current log record
            """
            cdim = self.custom_dimensions.copy()
            cdim.update(getattr(record, "custom_dimensions", {}))
            record.custom_dimensions = cdim

            return True

    # Transport module of opencensus-ext-azure logs info 'transmission
    # succeeded' which is also exported to azure if AzureLogHandler is
    # in root_logger. The following lines stops that.
    logging.getLogger("opencensus.ext.azure.common.transport").setLevel(logging.WARNING)

    loc = qc.config.GUID_components.location
    stat = qc.config.GUID_components.work_station

    def callback_function(envelope: "Envelope") -> bool:
        envelope.tags["ai.user.accountId"] = platform.node()
        envelope.tags["ai.user.id"] = f"{loc:02x}-{stat:06x}"
        return True

    telemetry_handler = AzureLogHandler(
        connection_string=f"InstrumentationKey="
        f"{qc.config.telemetry.instrumentation_key}"
    )
    assert telemetry_handler is not None
    telemetry_handler.add_telemetry_processor(callback_function)
    telemetry_handler.setLevel(logging.INFO)
    telemetry_handler.addFilter(CustomDimensionsFilter(default_custom_dimensions))
    telemetry_handler.setFormatter(get_formatter_for_telemetry())

    return telemetry_handler


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
    global telemetry_handler

    # set loggers to the supplied levels
    for name, level in qc.config.logger.logger_levels.items():
        logging.getLogger(name).setLevel(level)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.NOTSET)

    # remove previously set handlers
    for handler in (console_handler, file_handler, telemetry_handler):
        if handler is not None:
            handler.close()
            root_logger.removeHandler(handler)

    # add qcodes handlers
    # console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(qc.config.logger.console_level)
    console_handler.setFormatter(get_formatter())
    console_handler.addFilter(filter_out_telemetry_log_records)
    root_logger.addHandler(console_handler)

    # file
    filename = get_log_file_name()
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename, when="midnight", encoding="utf-8"
    )

    file_handler.setLevel(qc.config.logger.file_level)
    file_handler.setFormatter(get_formatter())
    root_logger.addHandler(file_handler)

    # capture any warnings from the warnings module
    logging.captureWarnings(capture=True)

    if qc.config.telemetry.enabled:
        root_logger.addHandler(
            _create_telemetry_handler()  # pyright: ignore[reportDeprecated]
        )

    log.info("QCoDes logger setup completed")

    log_qcodes_versions(log)

    print(f'Qcodes Logfile : {filename}')


def start_command_history_logger(log_dir: Optional[str] = None) -> None:
    """
    Start logging of the history of the interactive command shell.
    Works only with IPython and Jupyter. Call function again to set new path
    to log file.

    Args:
        log_dir: directory where log shall be stored to. If left out, defaults
            to ``~/.qcodes/logs/command_history.log``
    """
    # get_ipython is part of the public api but IPython does
    # not use __all__ to mark this
    from IPython import get_ipython  # type: ignore[attr-defined]
    ipython = get_ipython()
    if ipython is None:
        log.warning("Command history can't be saved"
                    " outside of IPython/Jupyter")
        return

    log_dir = log_dir or os.path.join(get_qcodes_user_path(), LOGGING_DIR)
    filename = os.path.join(log_dir, HISTORY_LOG_NAME)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    ipython.magic("%logstop")
    ipython.magic("%logstart -t -o {} {}".format(filename, "append"))
    log.info("Started logging IPython history")


def log_qcodes_versions(logger: logging.Logger) -> None:
    """
    Log the version information relevant to QCoDeS. This function logs
    the currently installed qcodes version, whether QCoDeS is installed in
    editable mode, the installed versions of QCoDeS' requirements, and the
    versions of all installed packages.
    """

    qc_version = qc.__version__
    qc_e_inst = is_qcodes_installed_editably()
    ipvs = get_all_installed_package_versions()

    logger.info(f"QCoDeS version: {qc_version}")
    logger.info(f"QCoDeS installed in editable mode: {qc_e_inst}")
    logger.info(f"All installed package versions: {json.dumps(ipvs)}")


def start_all_logging() -> None:
    """
    Starts python log module logging and ipython command history logging.
    """
    start_command_history_logger()
    start_logger()


def conditionally_start_all_logging() -> None:
    """Start logging if qcodesrc.json setup for it and in tool environment.

    This function will start logging if the session is not being executed by
    a tool such as pytest and under the following conditions depending on the
    qcodes configuration of ``config.logger.start_logging_on_import``:

    For ``never``:

        don't start logging automatically

    For ``always``:

        Always start logging when not in test environment

    For ``if_telemetry_set_up``:

        Start logging if the GUID components and the instrumentation key for
        telemetry are set up, and not in a test environment.
    """
    def start_logging_on_import() -> bool:
        config = qc.config
        if config.logger.start_logging_on_import == 'always':
            return True
        elif config.logger.start_logging_on_import == 'never':
            return False
        elif config.logger.start_logging_on_import == 'if_telemetry_set_up':
            return (
                config.GUID_components.location != 0
                and config.GUID_components.work_station != 0
                and config.telemetry.instrumentation_key
                != "00000000-0000-0000-0000-000000000000"
            )
        else:
            raise RuntimeError('Error in qcodesrc validation.')

    def running_in_test_or_tool() -> bool:
        import sys
        tools = (
            'pytest.py',
            'pytest',
            r'sphinx\__main__.py',    # Sphinx docs building
            '_jb_pytest_runner.py',  # Jetbrains Pycharm
            'testlauncher.py'        # VSCode
        )
        return any(sys.argv[0].endswith(tool) for tool in tools)

    if not running_in_test_or_tool() and start_logging_on_import():
        start_all_logging()


@contextmanager
def handler_level(
    level: LevelType, handler: Union[logging.Handler, "Sequence[logging.Handler]"]
) -> "Iterator[None]":
    """
    Context manager to temporarily change the level of handlers.

    Example:
        >>> with logger.handler_level(level=logging.DEBUG, handler=[h1, h1]):
        >>>     root_logger.debug('this is now visible')

    Args:
        level: Level to set the handlers to.
        handler: Handle or sequence of handlers which to change.
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
def console_level(level: LevelType) -> "Iterator[None]":
    """
    Context manager to temporarily change the level of the qcodes console
    handler.

    Example:
        >>> with logger.console_level(level=logging.DEBUG):
        >>>     root_logger.debug('this is now visible')

    Args:
        level: Level to set the console handler to.
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

    def __init__(self, logger: logging.Logger = logging.getLogger(),
                 level: Optional[LevelType] = None) -> None:
        self.logger = logger
        self.level = level or logging.NOTSET

        self.stashed_handlers = copy(self.logger.handlers)
        for h in self.stashed_handlers:
            self.logger.removeHandler(h)

    def __enter__(self) -> 'LogCapture':
        self.log_capture = io.StringIO()
        self.string_handler = logging.StreamHandler(self.log_capture)
        self.string_handler.setLevel(self.level)
        self.logger.addHandler(self.string_handler)
        return self

    def __exit__(
        self,
        exception_type: Optional[type[BaseException]],
        exception_value: Optional[BaseException],
        traceback: Optional["TracebackType"],
    ) -> None:
        self.logger.removeHandler(self.string_handler)
        self.value = self.log_capture.getvalue()
        self.log_capture.close()

        for h in self.stashed_handlers:
            self.logger.addHandler(h)
