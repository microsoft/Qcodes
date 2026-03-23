"""
Extended tests for ``qcodes.logger`` to improve coverage of
logger.py, log_analysis.py, and instrument_logger.py.
"""

from __future__ import annotations

import logging
import os
from copy import copy
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from qcodes import logger
from qcodes.logger import (
    LogCapture,
    flush_telemetry_traces,
    get_console_handler,
    get_file_handler,
    get_level_code,
    get_level_name,
    get_log_file_name,
    start_command_history_logger,
    start_logger,
)
from qcodes.logger.log_analysis import (
    log_to_dataframe,
    logfile_to_dataframe,
    time_difference,
)
from qcodes.logger.logger import (
    FORMAT_STRING_DICT,
    LOGGING_SEPARATOR,
    PYTHON_LOG_NAME,
    console_level,
    generate_log_file_name,
    get_formatter,
    get_formatter_for_telemetry,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture(autouse=True)
def cleanup_started_logger() -> Generator[None, None, None]:
    """Cleanup state left by a test calling start_logger."""
    root_logger = logging.getLogger()
    existing_handlers = copy(root_logger.handlers)
    yield
    post_test_handlers = copy(root_logger.handlers)
    for handler in post_test_handlers:
        if handler not in existing_handlers:
            handler.close()
            root_logger.removeHandler(handler)
    logger.logger.file_handler = None
    logger.logger.console_handler = None


# ---------------------------------------------------------------------------
# Tests for get_formatter / get_formatter_for_telemetry
# ---------------------------------------------------------------------------


def test_get_formatter_returns_formatter() -> None:
    fmt = get_formatter()
    assert isinstance(fmt, logging.Formatter)
    fmt_str = fmt._fmt
    assert fmt_str is not None
    for key in FORMAT_STRING_DICT:
        assert key in fmt_str
    assert LOGGING_SEPARATOR in fmt_str


def test_get_formatter_for_telemetry() -> None:
    fmt = get_formatter_for_telemetry()
    assert isinstance(fmt, logging.Formatter)
    fmt_str = fmt._fmt
    assert fmt_str is not None
    for key in ("message", "name", "funcName"):
        assert key in fmt_str
    # telemetry formatter should NOT contain asctime
    assert "asctime" not in fmt_str


# ---------------------------------------------------------------------------
# Tests for get_console_handler / get_file_handler
# ---------------------------------------------------------------------------


def test_get_console_handler_before_start() -> None:
    assert get_console_handler() is None


def test_get_file_handler_before_start() -> None:
    assert get_file_handler() is None


def test_get_handlers_after_start() -> None:
    start_logger()
    assert get_console_handler() is not None
    assert isinstance(get_console_handler(), logging.Handler)
    assert get_file_handler() is not None
    assert isinstance(get_file_handler(), logging.Handler)


# ---------------------------------------------------------------------------
# Tests for get_level_name
# ---------------------------------------------------------------------------


def test_get_level_name_from_int() -> None:
    assert get_level_name(logging.DEBUG) == "DEBUG"
    assert get_level_name(logging.WARNING) == "WARNING"
    assert get_level_name(logging.ERROR) == "ERROR"


def test_get_level_name_from_str() -> None:
    assert get_level_name("DEBUG") == "DEBUG"
    assert get_level_name("INFO") == "INFO"


def test_get_level_name_invalid_type() -> None:
    with pytest.raises(RuntimeError, match="get_level_name"):
        get_level_name(3.14)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests for get_level_code
# ---------------------------------------------------------------------------


def test_get_level_code_from_str() -> None:
    assert get_level_code("DEBUG") == logging.DEBUG
    assert get_level_code("WARNING") == logging.WARNING


def test_get_level_code_from_int() -> None:
    assert get_level_code(logging.DEBUG) == logging.DEBUG
    assert get_level_code(logging.INFO) == logging.INFO


def test_get_level_code_invalid_type() -> None:
    with pytest.raises(RuntimeError, match="get_level_code"):
        get_level_code(3.14)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests for generate_log_file_name / get_log_file_name
# ---------------------------------------------------------------------------


def test_generate_log_file_name() -> None:
    name = generate_log_file_name()
    pid = str(os.getpid())
    assert pid in name
    assert name.endswith(PYTHON_LOG_NAME)
    # format is YYMMDD-PID-qcodes.log
    parts = name.split("-")
    assert len(parts) >= 3


def test_get_log_file_name() -> None:
    path = get_log_file_name()
    assert isinstance(path, str)
    assert path.endswith(PYTHON_LOG_NAME)
    assert "logs" in path


# ---------------------------------------------------------------------------
# Tests for flush_telemetry_traces
# ---------------------------------------------------------------------------


def test_flush_telemetry_traces_no_op() -> None:
    # When telemetry is not set up this should be a no-op
    flush_telemetry_traces()


# ---------------------------------------------------------------------------
# Tests for start_command_history_logger
# ---------------------------------------------------------------------------


def test_start_command_history_logger_outside_ipython() -> None:
    # Outside IPython, get_ipython() returns None so this should just warn
    # and return without error.
    start_command_history_logger()


# ---------------------------------------------------------------------------
# Tests for console_level
# ---------------------------------------------------------------------------


def test_console_level_without_handler_raises() -> None:
    with pytest.raises(RuntimeError, match="Console handler is None"):
        with console_level(logging.DEBUG):
            pass


def test_console_level_with_handler() -> None:
    start_logger()
    handler = get_console_handler()
    assert handler is not None
    original_level = handler.level
    with console_level(logging.DEBUG):
        assert handler.level == logging.DEBUG
    assert handler.level == original_level


# ---------------------------------------------------------------------------
# Tests for LogCapture
# ---------------------------------------------------------------------------


def test_log_capture_basic() -> None:
    test_logger = logging.getLogger("test_log_capture_basic")
    test_logger.setLevel(logging.DEBUG)

    with LogCapture(logger=test_logger, level=logging.DEBUG) as logs:
        test_logger.debug("hello from capture")

    assert "hello from capture" in logs.value


def test_log_capture_multiple_messages() -> None:
    test_logger = logging.getLogger("test_log_capture_multi")
    test_logger.setLevel(logging.DEBUG)

    with LogCapture(logger=test_logger, level=logging.DEBUG) as logs:
        test_logger.info("first message")
        test_logger.warning("second message")

    assert "first message" in logs.value
    assert "second message" in logs.value


def test_log_capture_level_filtering() -> None:
    test_logger = logging.getLogger("test_log_capture_filter")
    test_logger.setLevel(logging.DEBUG)

    with LogCapture(logger=test_logger, level=logging.WARNING) as logs:
        test_logger.debug("should not appear")
        test_logger.warning("should appear")

    assert "should not appear" not in logs.value
    assert "should appear" in logs.value


def test_log_capture_restores_handlers() -> None:
    test_logger = logging.getLogger("test_log_capture_restore")
    test_logger.setLevel(logging.DEBUG)
    dummy_handler = logging.StreamHandler()
    test_logger.addHandler(dummy_handler)
    handler_count_before = len(test_logger.handlers)

    with LogCapture(logger=test_logger):
        test_logger.info("inside")

    assert len(test_logger.handlers) == handler_count_before
    test_logger.removeHandler(dummy_handler)


# ---------------------------------------------------------------------------
# Tests for log_to_dataframe
# ---------------------------------------------------------------------------


def test_log_to_dataframe() -> None:
    sep = LOGGING_SEPARATOR
    columns = list(FORMAT_STRING_DICT.keys())
    log_line = sep.join(
        [
            "2024-01-15 10:00:00,000",
            "qcodes.logger",
            "INFO",
            "logger",
            "start_logger",
            "42",
            "Test message",
        ]
    )
    df = log_to_dataframe([log_line])
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == columns
    assert len(df) == 1
    assert df["message"].iloc[0] == "Test message"


def test_log_to_dataframe_skips_tracebacks() -> None:
    sep = LOGGING_SEPARATOR
    valid = sep.join(
        [
            "2024-01-15 10:00:00,000",
            "mod",
            "ERROR",
            "mod",
            "func",
            "1",
            "error msg",
        ]
    )
    traceback_line = "Traceback (most recent call last):"
    df = log_to_dataframe([valid, traceback_line])
    assert len(df) == 1


def test_log_to_dataframe_empty() -> None:
    df = log_to_dataframe(["Traceback line only"])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ---------------------------------------------------------------------------
# Tests for logfile_to_dataframe
# ---------------------------------------------------------------------------


def test_logfile_to_dataframe(tmp_path: Path) -> None:
    sep = LOGGING_SEPARATOR
    line = sep.join(
        [
            "2024-06-01 12:00:00,000",
            "qcodes.logger",
            "DEBUG",
            "logger",
            "test_func",
            "10",
            "file log message",
        ]
    )
    logfile = tmp_path / "test.log"
    logfile.write_text(line + "\n")
    df = logfile_to_dataframe(str(logfile))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df["message"].iloc[0].strip() == "file log message"


# ---------------------------------------------------------------------------
# Tests for time_difference
# ---------------------------------------------------------------------------


def test_time_difference_basic() -> None:
    first = pd.Series(["2024-01-01 00:00:00.000", "2024-01-01 00:00:01.000"])
    second = pd.Series(["2024-01-01 00:00:01.000", "2024-01-01 00:00:03.000"])
    result = time_difference(first, second, use_first_series_labels=True)
    assert isinstance(result, pd.Series)
    assert len(result) == 2
    assert result.iloc[0] == pytest.approx(1.0, abs=0.01)
    assert result.iloc[1] == pytest.approx(2.0, abs=0.01)


def test_time_difference_use_second_labels() -> None:
    first = pd.Series(["2024-01-01 00:00:00.000"], index=["a"])
    second = pd.Series(["2024-01-01 00:00:05.000"], index=["b"])
    result = time_difference(first, second, use_first_series_labels=False)
    assert list(result.index) == ["b"]
    assert result.iloc[0] == pytest.approx(5.0, abs=0.01)


def test_time_difference_comma_separator() -> None:
    first = pd.Series(["2024-01-01 00:00:00,000"])
    second = pd.Series(["2024-01-01 00:00:02,000"])
    result = time_difference(first, second)
    assert result.iloc[0] == pytest.approx(2.0, abs=0.01)
