"""
Tests for `qcodes.utils.logger`.
"""

import logging
import os
from copy import copy
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pytest import LogCaptureFixture

import qcodes as qc
from qcodes import logger
from qcodes.instrument import Instrument
from qcodes.instrument_drivers.american_magnetics import AMIModel430, AMIModel4303D
from qcodes.instrument_drivers.tektronix import TektronixAWG5208
from qcodes.logger.log_analysis import capture_dataframe
from tests.drivers.test_lakeshore import Model_372_Mock

if TYPE_CHECKING:
    from collections.abc import Generator

TEST_LOG_MESSAGE = "test log message"

NUM_PYTEST_LOGGERS = 4


@pytest.fixture(autouse=True)
def cleanup_started_logger() -> "Generator[None, None, None]":
    # cleanup state left by a test calling start_logger
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


@pytest.fixture
def awg5208(caplog: LogCaptureFixture) -> "Generator[TektronixAWG5208, None, None]":
    with caplog.at_level(logging.INFO):
        inst = TektronixAWG5208(
            "awg_sim",
            address="GPIB0::1::INSTR",
            pyvisa_sim_file="Tektronix_AWG5208.yaml",
        )

    try:
        yield inst
    finally:
        inst.close()


@pytest.fixture
def model372() -> "Generator[Model_372_Mock, None, None]":
    inst = Model_372_Mock(
        "lakeshore_372",
        "GPIB::3::INSTR",
        pyvisa_sim_file="lakeshore_model372.yaml",
        device_clear=False,
    )
    inst.sample_heater.range_limits([0, 0.25, 0.5, 1, 2, 3, 4, 7])
    inst.warmup_heater.range_limits([0, 0.25, 0.5, 1, 2, 3, 4, 7])
    try:
        yield inst
    finally:
        inst.close()


@pytest.fixture()
def AMI430_3D() -> (
    "Generator[tuple[AMIModel4303D, AMIModel430, AMIModel430, AMIModel430], None, None]"
):
    mag_x = AMIModel430(
        "x",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="AMI430.yaml",
        terminator="\n",
    )
    mag_y = AMIModel430(
        "y",
        address="GPIB::2::INSTR",
        pyvisa_sim_file="AMI430.yaml",
        terminator="\n",
    )
    mag_z = AMIModel430(
        "z",
        address="GPIB::3::INSTR",
        pyvisa_sim_file="AMI430.yaml",
        terminator="\n",
    )
    field_limit = [
        lambda x, y, z: x == 0 and y == 0 and z < 3,
        lambda x, y, z: np.linalg.norm([x, y, z]) < 2,
    ]
    driver = AMIModel4303D("AMI430_3D", mag_x, mag_y, mag_z, field_limit)
    try:
        yield driver, mag_x, mag_y, mag_z
    finally:
        driver.close()
        mag_x.close()
        mag_y.close()
        mag_z.close()


def test_get_log_file_name() -> None:
    fp = logger.logger.get_log_file_name().split(os.sep)
    assert str(os.getpid()) in fp[-1]
    assert logger.logger.PYTHON_LOG_NAME in fp[-1]
    assert fp[-2] == logger.logger.LOGGING_DIR
    assert fp[-3] == ".qcodes"


def test_start_logger() -> None:
    # remove all Handlers
    logger.start_logger()
    assert isinstance(logger.get_console_handler(), logging.Handler)
    assert isinstance(logger.get_file_handler(), logging.Handler)

    console_level = logger.get_level_code(qc.config.logger.console_level)
    file_level = logger.get_level_code(qc.config.logger.file_level)
    console_handler = logger.get_console_handler()
    assert console_handler is not None
    assert console_handler.level == console_level
    file_handler = logger.get_file_handler()
    assert file_handler is not None
    assert file_handler.level == file_level

    assert logging.getLogger().level == logger.get_level_code("NOTSET")


def test_start_logger_twice() -> None:
    root_logger = logging.getLogger()

    assert len(root_logger.handlers) == NUM_PYTEST_LOGGERS

    logger.start_logger()
    logger.start_logger()
    handlers = root_logger.handlers
    # we expect there to be two log handlers file+console
    # plus the existing ones from pytest
    assert len(handlers) == 2 + NUM_PYTEST_LOGGERS


def test_set_level_without_starting_raises() -> None:
    with pytest.raises(RuntimeError):
        with logger.console_level("DEBUG"):
            pass
    assert len(logging.getLogger().handlers) == NUM_PYTEST_LOGGERS


def test_handler_level() -> None:
    logger.start_logger()
    with logger.LogCapture(level=logging.INFO) as logs:
        logging.debug(TEST_LOG_MESSAGE)
    assert logs.value == ""

    with logger.LogCapture(level=logging.INFO) as logs:
        with logger.handler_level(level=logging.DEBUG, handler=logs.string_handler):
            print(logs.string_handler)
            logging.debug(TEST_LOG_MESSAGE)
    assert logs.value.strip() == TEST_LOG_MESSAGE


def test_filter_instrument(
    AMI430_3D: tuple[AMIModel4303D, AMIModel430, AMIModel430, AMIModel430],
) -> None:
    driver, mag_x, mag_y, mag_z = AMI430_3D

    logger.start_logger()

    # filter one instrument
    driver.cartesian((0, 0, 0))
    with logger.LogCapture(level=logging.DEBUG) as logs:
        with logger.filter_instrument(mag_x, handler=logs.string_handler):
            driver.cartesian((0, 0, 1))
    for line in logs.value.splitlines():
        assert "[x(AMIModel430)]" in line
        assert "[y(AMIModel430)]" not in line
        assert "[z(AMIModel430)]" not in line

    # filter multiple instruments
    driver.cartesian((0, 0, 0))
    with logger.LogCapture(level=logging.DEBUG) as logs:
        with logger.filter_instrument((mag_x, mag_y), handler=logs.string_handler):
            driver.cartesian((0, 0, 1))

    any_x = False
    any_y = False
    for line in logs.value.splitlines():
        has_x = "[x(AMIModel430)]" in line
        has_y = "[y(AMIModel430)]" in line
        has_z = "[z(AMIModel430)]" in line

        assert has_x or has_y
        assert not has_z

        any_x |= has_x
        any_y |= has_y
    assert any_x
    assert any_y


def test_filter_without_started_logger_raises(
    AMI430_3D: tuple[AMIModel4303D, AMIModel430, AMIModel430, AMIModel430],
) -> None:
    driver, mag_x, mag_y, mag_z = AMI430_3D

    # filter one instrument
    driver.cartesian((0, 0, 0))
    with pytest.raises(RuntimeError):
        with logger.filter_instrument(mag_x):
            pass


def test_capture_dataframe() -> None:
    root_logger = logging.getLogger()
    # the logger must be started to set level
    # debug on the rootlogger
    logger.start_logger()
    with capture_dataframe() as (_, cb):
        root_logger.debug(TEST_LOG_MESSAGE)
        df = cb()
    assert len(df) == 1
    assert df.message[0] == TEST_LOG_MESSAGE


def test_channels(model372: Model_372_Mock) -> None:
    """
    Test that messages logged in a channel are propagated to the
    main instrument.
    """
    inst = model372

    # set range to some other value so that it will actually be set in
    # the next call.
    inst.sample_heater.range_limits([0, 0.25, 0.5, 1, 2, 3, 4, 7])
    inst.sample_heater.set_range_from_temperature(1)
    with logger.LogCapture(level=logging.DEBUG) as logs_unfiltered:
        inst.sample_heater.set_range_from_temperature(0.1)

    # reset without capturing
    inst.sample_heater.set_range_from_temperature(1)
    # rerun with instrument filter
    with (
        logger.LogCapture(level=logging.DEBUG) as logs_filtered,
        logger.filter_instrument(inst, handler=logs_filtered.string_handler),
    ):
        inst.sample_heater.set_range_from_temperature(0.1)

    logs_filtered_2 = [
        log for log in logs_filtered.value.splitlines() if "[lakeshore" in log
    ]
    logs_unfiltered_2 = [
        log for log in logs_unfiltered.value.splitlines() if "[lakeshore" in log
    ]

    for f, u in zip(logs_filtered_2, logs_unfiltered_2):
        assert f == u


def test_channels_nomessages(model372: Model_372_Mock) -> None:
    """
    Test that messages logged in a channel are not propagated to
    any instrument.
    """
    inst = model372
    # test with wrong instrument
    mock = Instrument("mock")
    inst.sample_heater.set_range_from_temperature(1)
    with (
        logger.LogCapture(level=logging.DEBUG) as logs,
        logger.filter_instrument(mock, handler=logs.string_handler),
    ):
        inst.sample_heater.set_range_from_temperature(0.1)
    logs_2 = [log for log in logs.value.splitlines() if "[lakeshore" in log]
    assert len(logs_2) == 0
    mock.close()


@pytest.mark.usefixtures("awg5208")
def test_instrument_connect_message(caplog: LogCaptureFixture) -> None:
    """
    Test that the connect_message method logs as expected

    This test kind of belongs both here and in the tests for the instrument
    code, but it is more conveniently written here
    """

    setup_records = caplog.get_records("setup")

    idn = {"vendor": "QCoDeS", "model": "AWG5208", "serial": "1000", "firmware": "0.1"}
    expected_con_mssg = f"[awg_sim(TektronixAWG5208)] Connected to instrument: {idn}"

    assert any(rec.msg == expected_con_mssg for rec in setup_records)


def test_installation_info_logging() -> None:
    """
    Test that installation information is logged upon starting the logging
    """
    logger.start_logger()

    with open(logger.get_log_file_name()) as f:
        lines = f.readlines()

    assert "QCoDeS version:" in lines[-3]
    assert "QCoDeS installed in editable mode:" in lines[-2]
    assert "All installed package versions:" in lines[-1]
