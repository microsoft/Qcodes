"""
Tests for `qcodes.utils.logger`.
"""
import pytest
import os
import logging
from copy import copy
import qcodes.utils.logger as logger
import qcodes as qc

TEST_LOG_MESSAGE = 'test log message'

@pytest.fixture
def remove_root_handlers():
    root_logger = logging.getLogger()
    handlers = copy(root_logger.handlers)
    for handler in handlers:
        handler.close()
        root_logger.removeHandler(handler)

def test_get_log_file_name():
    fp = logger.get_log_file_name().split(os.sep)
    assert fp[-1] == logger.PYTHON_LOG_NAME
    assert fp[-2] == logger.LOGGING_DIR
    assert fp[-3] == '.qcodes'


@pytest.mark.usefixtures("remove_root_handlers")
def test_start_logger():
    # remove all Handlers
    logger.start_logger()
    assert isinstance(logger.get_console_handler(), logging.Handler)
    assert isinstance(logger.get_file_handler(), logging.Handler)

    console_level = logger.get_level_code(qc.config.logger.console_level)
    file_level = logger.get_level_code(qc.config.logger.file_level)
    assert logger.get_console_handler().level == console_level
    assert logger.get_file_handler().level == file_level

    assert logging.getLogger().level == logger.get_level_code('DEBUG')

@pytest.mark.usefixtures("remove_root_handlers")
def test_start_logger_twice():
    logger.start_logger()
    logger.start_logger()
    handlers = logging.getLogger().handlers
    for h in handlers:
        print(h.__module__)
    # there is always one logger registered
    assert len(logging.getLogger().handlers) == 2+1


@pytest.mark.usefixtures("remove_root_handlers")
def test_handler_level():
    with logger.LogCapture(level=logging.INFO) as logs:
        logging.debug(TEST_LOG_MESSAGE)
    assert logs.value == ''

    with logger.LogCapture(level=logging.INFO) as logs:
        with logger.handler_level(level=logging.DEBUG,
                                  handler=logs.string_handler):
            print(logs.string_handler)
            logging.debug(TEST_LOG_MESSAGE)
    assert logs.value.strip() == TEST_LOG_MESSAGE

@pytest.mark.usefixtures("remove_root_handlers")
def test_filter_instrument():
    from qcodes.instrument.ip_to_visa import AMI430_VISA
    from qcodes.instrument_drivers.american_magnetics.AMI430 import AMI430_3D
    import qcodes.instrument.sims as sims
    import numpy as np

    logger.start_logger()

    visalib = sims.__file__.replace('__init__.py', 'AMI430.yaml@sim')
    mag_x = AMI430_VISA('x', address='GPIB::1::INSTR', visalib=visalib,
                        terminator='\n', port=1)
    mag_y = AMI430_VISA('y', address='GPIB::2::INSTR', visalib=visalib,
                        terminator='\n', port=1)
    mag_z = AMI430_VISA('z', address='GPIB::3::INSTR', visalib=visalib,
                        terminator='\n', port=1)
    field_limit = [
        lambda x, y, z: x == 0 and y == 0 and z < 3,
        lambda x, y, z: np.linalg.norm([x, y, z]) < 2
    ]

    driver = AMI430_3D("AMI430-3D", mag_x, mag_y, mag_z, field_limit)

    # filter one instrument
    driver.cartesian((0, 0, 0))
    with logger.LogCapture(level=logging.DEBUG) as logs:
        with logger.filter_instrument(mag_x, handler=logs.string_handler):
            driver.cartesian((0, 0, 1))
    for line in logs.value.splitlines():
        assert '[x]' in line
        assert '[y]' not in line
        assert '[z]' not in line

    # filter multiple instruments
    driver.cartesian((0, 0, 0))
    with logger.LogCapture(level=logging.DEBUG) as logs:
        with logger.filter_instrument((mag_x, mag_y), handler=logs.string_handler):
            driver.cartesian((0, 0, 1))

    any_x = False
    any_y = False
    for line in logs.value.splitlines():
        has_x = '[x]' in line
        has_y = '[y]' in line
        has_z = '[z]' in line

        assert has_x or has_y
        assert not has_z

        any_x |= has_x
        any_y |= has_y
    assert any_x
    assert any_y

@pytest.mark.usefixtures("remove_root_handlers")
def test_capture_dataframe():
    root_logger = logging.getLogger()
    with logger.capture_dataframe() as (_, cb):
        root_logger.debug(TEST_LOG_MESSAGE)
        df = cb()
    assert len(df) == 1
    assert df.message[0] == TEST_LOG_MESSAGE
