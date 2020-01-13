"""
Tests for `qcodes.utils.logger`.
"""
import pytest
import os
import logging
from copy import copy
import qcodes.logger as logger
from qcodes.logger.log_analysis import capture_dataframe
import qcodes as qc


TEST_LOG_MESSAGE = 'test log message'


@pytest.fixture
def remove_root_handlers():
    root_logger = logging.getLogger()
    handlers = copy(root_logger.handlers)
    for handler in handlers:
        handler.close()
        root_logger.removeHandler(handler)
    logger.logger.file_handler = None
    logger.logger.console_handler = None


@pytest.fixture
def awg5208():

    from qcodes.instrument_drivers.tektronix.AWG5208 import AWG5208
    import qcodes.instrument.sims as sims
    visalib = sims.__file__.replace('__init__.py',
                                    'Tektronix_AWG5208.yaml@sim')

    logger.start_logger()

    inst = AWG5208('awg_sim',
                   address='GPIB0::1::INSTR',
                   visalib=visalib)

    try:
        yield inst
    finally:
        inst.close()


@pytest.fixture
def model372():
    import qcodes.instrument.sims as sims
    from qcodes.tests.drivers.test_lakeshore import Model_372_Mock

    logger.LOGGING_SEPARATOR = ' - '

    logger.start_logger()

    visalib = sims.__file__.replace('__init__.py',
                                    'lakeshore_model372.yaml@sim')

    inst = Model_372_Mock('lakeshore_372', 'GPIB::3::INSTR',
                          visalib=visalib, device_clear=False)
    inst.sample_heater.range_limits([0, 0.25, 0.5, 1, 2, 3, 4, 7])
    inst.warmup_heater.range_limits([0, 0.25, 0.5, 1, 2, 3, 4, 7])
    try:
        yield inst
    finally:
        inst.close()


@pytest.fixture()
def AMI430_3D():
    from qcodes.instrument.ip_to_visa import AMI430_VISA
    from qcodes.instrument_drivers.american_magnetics.AMI430 import AMI430_3D
    import qcodes.instrument.sims as sims
    import numpy as np
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
    try:
        yield driver, mag_x, mag_y, mag_z
    finally:
        driver.close()
        mag_x.close()
        mag_y.close()
        mag_z.close()


def test_get_log_file_name():
    fp = logger.logger.get_log_file_name().split(os.sep)
    assert str(os.getpid()) in fp[-1]
    assert logger.logger.PYTHON_LOG_NAME in fp[-1]
    assert fp[-2] == logger.logger.LOGGING_DIR
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
    # there is always one logger registered from pytest
    # and the telemetry logger is always off in the tests
    assert len(handlers) == 2+1

@pytest.mark.usefixtures("remove_root_handlers")
def test_set_level_without_starting_raises():
    with pytest.raises(RuntimeError):
        with logger.console_level('DEBUG'):
            pass
    # there is always one logger registered from pytest
    assert len(logging.getLogger().handlers) == 1


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
def test_filter_instrument(AMI430_3D):

    driver, mag_x, mag_y, mag_z = AMI430_3D

    logger.start_logger()

    # filter one instrument
    driver.cartesian((0, 0, 0))
    with logger.LogCapture(level=logging.DEBUG) as logs:
        with logger.filter_instrument(mag_x, handler=logs.string_handler):
            driver.cartesian((0, 0, 1))
    for line in logs.value.splitlines():
        assert '[x(AMI430_VISA)]' in line
        assert '[y(AMI430_VISA)]' not in line
        assert '[z(AMI430_VISA)]' not in line

    # filter multiple instruments
    driver.cartesian((0, 0, 0))
    with logger.LogCapture(level=logging.DEBUG) as logs:
        with logger.filter_instrument((mag_x, mag_y), handler=logs.string_handler):
            driver.cartesian((0, 0, 1))

    any_x = False
    any_y = False
    for line in logs.value.splitlines():
        has_x = '[x(AMI430_VISA)]' in line
        has_y = '[y(AMI430_VISA)]' in line
        has_z = '[z(AMI430_VISA)]' in line

        assert has_x or has_y
        assert not has_z

        any_x |= has_x
        any_y |= has_y
    assert any_x
    assert any_y


@pytest.mark.usefixtures("remove_root_handlers")
def test_filter_without_started_logger_raises(AMI430_3D):

    driver, mag_x, mag_y, mag_z = AMI430_3D

    # filter one instrument
    driver.cartesian((0, 0, 0))
    with pytest.raises(RuntimeError):
        with logger.filter_instrument(mag_x):
            pass


@pytest.mark.usefixtures("remove_root_handlers")
def test_capture_dataframe():
    root_logger = logging.getLogger()
    with capture_dataframe() as (_, cb):
        root_logger.debug(TEST_LOG_MESSAGE)
        df = cb()
    assert len(df) == 1
    assert df.message[0] == TEST_LOG_MESSAGE


@pytest.mark.usefixtures("remove_root_handlers")
def test_channels(model372):
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
    with logger.LogCapture(level=logging.DEBUG) as logs_filtered,\
            logger.filter_instrument(inst,
                                     handler=logs_filtered.string_handler):
        inst.sample_heater.set_range_from_temperature(0.1)

    logs_filtered = [l for l in logs_filtered.value.splitlines()
                        if '[lakeshore' in l]
    logs_unfiltered = [l for l in logs_unfiltered.value.splitlines()
                        if '[lakeshore' in l]

    for f, u in zip(logs_filtered, logs_unfiltered):
        assert f == u


@pytest.mark.usefixtures("remove_root_handlers")
def test_channels_nomessages(model372):
    """
    Test that messages logged in a channel are not propagated to
    any instrument.
    """
    inst = model372
    # test with wrong instrument
    mock = qc.Instrument('mock')
    inst.sample_heater.set_range_from_temperature(1)
    with logger.LogCapture(level=logging.DEBUG) as logs,\
            logger.filter_instrument(mock, handler=logs.string_handler):
        inst.sample_heater.set_range_from_temperature(0.1)
    logs = [l for l in logs.value.splitlines()
            if '[lakeshore' in l]
    assert len(logs) == 0
    mock.close()


@pytest.mark.usefixtures("remove_root_handlers", "awg5208")
def test_instrument_connect_message():
    """
    Test that the connect_message method logs as expected

    This test kind of belongs both here and in the tests for the instrument
    code, but it is more conveniently written here
    """

    with open(logger.get_log_file_name(), 'r') as f:
        lines = f.readlines()

    con_mssg_log_line = lines[-1]

    sep = logger.logger.LOGGING_SEPARATOR

    con_mss = con_mssg_log_line.split(sep)[-1]
    idn = {"vendor": "QCoDeS", "model": "AWG5208",
           "serial": "1000", "firmware": "0.1"}
    expected_con_mssg = ("[awg_sim(AWG5208)] Connected to instrument: "
                         f"{idn}\n")

    assert con_mss == expected_con_mssg


@pytest.mark.usefixtures("remove_root_handlers")
def test_installation_info_logging():
    """
    Test that installation information is logged upon starting the logging
    """
    logger.start_logger()

    with open(logger.get_log_file_name(), 'r') as f:
        lines = f.readlines()

    assert 'QCoDeS version:' in lines[-3]
    assert 'QCoDeS installed in editable mode:' in lines[-2]
    assert 'QCoDeS requirements versions:' in lines[-1]
