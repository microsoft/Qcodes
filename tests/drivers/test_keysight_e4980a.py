import re

import pytest

import qcodes.instrument_drivers.Keysight.keysight_e4980a as E4980A


@pytest.fixture(name="driver")
def _make_driver():
    instr = E4980A.KeysightE4980A(
        "E4980A", address="GPIB::1::INSTR", pyvisa_sim_file="Keysight_E4980A.yaml"
    )
    yield instr
    instr.close()


def test_idn(driver) -> None:
    assert {
        "firmware": "A.02.10",
        "model": "E4980A",
        "serial": "MY46516036",
        "vendor": "Keysight Technologies",
    } == driver.IDN()


def test_raise_error_for_volt_level_query_when_signal_set_as_current(driver) -> None:
    driver.current_level(0.01)
    msg = re.escape(
        "Cannot get voltage level as signal is set with current level parameter."
    )
    with pytest.raises(RuntimeError, match=msg):
        driver.voltage_level()


def test_voltage_level_set_method(driver) -> None:
    driver.voltage_level(3)
    assert driver.voltage_level() == 3


def test_signal_mode_parameter(driver) -> None:
    driver.voltage_level(2)
    assert driver.signal_mode() == "Voltage"

    driver.current_level(0.005)
    assert driver.signal_mode() == "Current"


def test_raise_error_for_curr_level_query_when_signal_set_as_voltage(driver) -> None:
    driver.voltage_level(1)
    msg = re.escape(
        "Cannot get current level as signal is set with voltage level parameter."
    )
    with pytest.raises(RuntimeError, match=msg):
        driver.current_level()


def test_current_level_set_method(driver) -> None:
    driver.current_level(0.003)
    assert driver.current_level() == 0.003
