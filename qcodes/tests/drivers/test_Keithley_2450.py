import logging
import pytest

from qcodes.instrument_drivers.tektronix.Keithley_2450 import Keithley2450, ParameterError

import qcodes.instrument.sims as sims
visalib = sims.__file__.replace('__init__.py', 'Keithley_2450.yaml@sim')


@pytest.fixture(scope='module')
def k2450():
    """
    Create a Keithley 2450 instrument
    """
    driver = Keithley2450('k2450', address='GPIB::2::INSTR', visalib=visalib)
    yield driver
    driver.close()


def test_wrong_mode(caplog):
    """
    Starting an instrument in the wrong mode should result in a warning. Additionally, no
    parameters should be available, other then the parameters "IDN" and "timeout" which
    are created by the Instrument and VisaInstrument parent classes
    """
    with caplog.at_level(logging.WARNING):
        instrument = Keithley2450('wrong_mode', address='GPIB::1::INSTR', visalib=visalib)
        assert "The instrument is an unsupported language mode" in caplog.text
        assert list(instrument.parameters.keys()) == ["IDN", "timeout"]


def test_change_source_function(k2450):
    """
    The parameters available on the source sub-module depend on the function. We test the following
    1) When we are in the "current" mode, we cannot get/set the voltage parameter. When we switch
    to "voltage" mode, the cannot get/set the current parameter
    2) The properties 'range', 'auto_range' and 'limit' return the appropriate parameter given the
    function.
    """
    assert k2450.source.function() == "current"
    assert k2450.source.range is k2450.source.parameters["_range_current"]
    assert k2450.source.auto_range is k2450.source.parameters["_auto_range_current"]
    assert k2450.source.limit is k2450.source.parameters["_limit_current"]

    with pytest.raises(
            ParameterError,
            match="'k2450.source.function' has value 'current'. This should be 'voltage'"
    ):
        k2450.source.voltage()

    k2450.source.function("voltage")
    assert k2450.source.range is k2450.source.parameters["_range_voltage"]
    assert k2450.source.auto_range is k2450.source.parameters["_auto_range_voltage"]
    assert k2450.source.limit is k2450.source.parameters["_limit_voltage"]

    with pytest.raises(
            ParameterError,
            match="'k2450.source.function' has value 'voltage'. This should be 'current'"
    ):
        k2450.source.current()


def test_source_change_error(k2450):
    """
    If the sense function is in resistance mode, test that an error is generated
    when we change the source function
    """
    k2450.source.function("voltage")
    k2450.sense.function("resistance")
    with pytest.raises(
        RuntimeError,
        match="Cannot change the source function while sense function is in 'resistance' mode"
    ):
        k2450.source.function("current")

    k2450.sense.function("voltage")
    k2450.source.function("current")


def test_sense_current_mode(k2450):
    """
    We test the following
    1) When we are in the "current" mode, we cannot get/set the voltage or resistance parameter.
    2) The properties 'range' and 'auto_range' return the parameters associated with current
    """
    assert k2450.sense.function("current")
    assert k2450.sense.range is k2450.sense.parameters["_range_current"]
    assert k2450.sense.auto_range is k2450.sense.parameters["_auto_range_current"]

    for parameter_name in ["voltage", "resistance"]:
        with pytest.raises(
                ParameterError,
                match="'k2450.sense.function' has value 'current'. This should be '([a-z]*)'"
        ):
            parameter = k2450.sense.parameters[parameter_name]
            parameter.get()


def test_sense_voltage_mode(k2450):
    """
    We test the following
    1) When we are in the "voltage" mode, we cannot get/set the current or resistance parameter.
    2) The properties 'range' and 'auto_range' return the parameters associated with voltage
    """
    k2450.sense.function("voltage")
    assert k2450.sense.range is k2450.sense.parameters["_range_voltage"]
    assert k2450.sense.auto_range is k2450.sense.parameters["_auto_range_voltage"]

    for parameter_name in ["current", "resistance"]:
        with pytest.raises(
                ParameterError,
                match="'k2450.sense.function' has value 'voltage'. This should be '([a-z]*)'"
        ):
            parameter = k2450.sense.parameters[parameter_name]
            parameter.get()


def test_sense_resistance_mode(k2450):
    """
    We test the following
    1) When we are in the "resistance" mode, we cannot get/set the voltage or current parameter.
    2) The properties 'range' and 'auto_range' return the parameters associated with resistance
    """
    k2450.sense.function("resistance")
    assert k2450.sense.range is k2450.sense.parameters["_range_resistance"]
    assert k2450.sense.auto_range is k2450.sense.parameters["_auto_range_resistance"]

    for parameter_name in ["current", "voltage"]:
        with pytest.raises(
                ParameterError,
                match="'k2450.sense.function' has value 'resistance'. This should be '([a-z]*)'"
        ):
            parameter = k2450.sense.parameters[parameter_name]
            parameter.get()
