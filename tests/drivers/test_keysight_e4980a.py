import re
from typing import TYPE_CHECKING

import pytest

import qcodes.instrument_drivers.Keysight.keysight_e4980a as E4980A

if TYPE_CHECKING:
    from collections.abc import Generator

    from pytest_mock import MockerFixture


@pytest.fixture(name="driver")
def _make_driver() -> "Generator[E4980A.KeysightE4980A, None, None]":
    instr = E4980A.KeysightE4980A(
        "E4980A", address="GPIB::1::INSTR", pyvisa_sim_file="Keysight_E4980A.yaml"
    )
    yield instr
    instr.close()


def test_idn(driver: E4980A.KeysightE4980A) -> None:
    assert {
        "firmware": "A.02.10",
        "model": "E4980A",
        "serial": "MY46516036",
        "vendor": "Keysight Technologies",
    } == driver.IDN()


def test_raise_error_for_volt_level_query_when_signal_set_as_current(
    driver: E4980A.KeysightE4980A,
) -> None:
    driver.current_level(0.01)
    msg = re.escape(
        "Cannot get voltage level as signal is set with current level parameter."
    )
    with pytest.raises(RuntimeError, match=msg):
        driver.voltage_level()


def test_voltage_level_set_method(driver: E4980A.KeysightE4980A) -> None:
    driver.voltage_level(3)
    assert driver.voltage_level() == 3


def test_signal_mode_parameter(driver: E4980A.KeysightE4980A) -> None:
    driver.voltage_level(2)
    assert driver.signal_mode() == "Voltage"

    driver.current_level(0.005)
    assert driver.signal_mode() == "Current"


def test_raise_error_for_curr_level_query_when_signal_set_as_voltage(
    driver: E4980A.KeysightE4980A,
) -> None:
    driver.voltage_level(1)
    msg = re.escape(
        "Cannot get current level as signal is set with voltage level parameter."
    )
    with pytest.raises(RuntimeError, match=msg):
        driver.current_level()


def test_current_level_set_method(driver: E4980A.KeysightE4980A) -> None:
    driver.current_level(0.003)
    assert driver.current_level() == 0.003


def test_range_is_float(driver: E4980A.KeysightE4980A) -> None:
    impedance_range = driver.range()
    assert isinstance(impedance_range, float)
    assert impedance_range == 100.0

    driver.range(1000)
    assert driver.range() == 1000.0


@pytest.mark.parametrize(
    ("action", "command"),
    [("open", ":CORRection:OPEN"), ("short", ":CORRection:SHORt")],
)
def test_correction_methods_send_command(
    driver: E4980A.KeysightE4980A, mocker: "MockerFixture", action: str, command: str
) -> None:
    # these used to be parameters, so calling them without an argument only
    # returned the cached value and did not send anything
    write_raw = mocker.spy(driver, "write_raw")
    if action == "open":
        driver.correction.open()
    else:
        driver.correction.short()

    write_raw.assert_called_once_with(command)
    # the instrument does not reply, so the next query is unaffected
    assert driver.IDN()["model"] == "E4980A"
