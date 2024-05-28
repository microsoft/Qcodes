import numpy as np
import pytest

from qcodes.instrument_drivers.Keysight import (
    Keysight34465A,
)


@pytest.fixture(scope='function')
def driver():
    keysight_sim = Keysight34465A(
        "keysight_34465A_sim",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="Keysight_34465A.yaml",
    )

    try:
        yield keysight_sim
    finally:
        Keysight34465A.close_all()


@pytest.fixture(scope='function')
def driver_with_read_and_fetch_mocked(val_volt):
    keysight_sim = Keysight34465A(
        "keysight_34465A_sim",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="Keysight_34465A.yaml",
    )

    def get_ask_with_read_mock(original_ask, read_value):
        def ask_with_read_mock(cmd: str) -> str:
            if cmd in ("READ?", "FETCH?"):
                return read_value
            else:
                return original_ask(cmd)
        return ask_with_read_mock

    keysight_sim.ask = get_ask_with_read_mock(keysight_sim.ask, val_volt)
    try:
        yield keysight_sim
    finally:
        Keysight34465A.close_all()


def test_init(driver) -> None:
    idn = driver.IDN()
    assert idn['vendor'] == 'Keysight'
    assert idn['model'] == '34465A'


def test_has_dig_option(driver) -> None:
    assert True is driver.has_DIG


def test_model_flag(driver) -> None:
    assert True is driver.is_34465A_34470A


def test_reset(driver) -> None:
    driver.reset()


def test_NPLC(driver) -> None:
    assert driver.NPLC.get() == 10.0
    driver.NPLC.set(0.2)
    assert driver.NPLC.get() == 0.2
    driver.NPLC.set(10.0)


@pytest.mark.parametrize("val_volt", ['100.0'])
def test_get_voltage(driver_with_read_and_fetch_mocked, val_volt) -> None:
    voltage = driver_with_read_and_fetch_mocked.volt.get()
    assert voltage == 100.0


@pytest.mark.parametrize("val_volt", ['9.9e37'])
def test_get_voltage_plus_inf(driver_with_read_and_fetch_mocked, val_volt) -> None:
    voltage = driver_with_read_and_fetch_mocked.volt.get()
    assert voltage == np.inf


@pytest.mark.parametrize("val_volt", ['-9.9e37'])
def test_get_voltage_minus_inf(driver_with_read_and_fetch_mocked, val_volt) -> None:
    voltage = driver_with_read_and_fetch_mocked.volt.get()
    assert voltage == -np.inf


@pytest.mark.xfail(run=False, reason="If the test is run, it will pass "
                                     "but all tests after this one will "
                                     "fail. The problem is coming from "
                                     "timetrace().")
@pytest.mark.parametrize("val_volt", ['10, 9.9e37, -9.9e37'])
def test_get_timetrace(driver_with_read_and_fetch_mocked, val_volt) -> None:
    driver_with_read_and_fetch_mocked.timetrace_npts(3)
    assert driver_with_read_and_fetch_mocked.timetrace_npts() == 3
    voltage = driver_with_read_and_fetch_mocked.timetrace()
    assert (voltage == np.array([10.0, np.inf, -np.inf])).all()


def test_set_get_autorange(driver) -> None:
    ar = driver.autorange.get()
    assert ar == 'OFF'
    driver.autorange.set('ON')
    ar = driver.autorange.get()
    assert ar == 'ON'
    driver.autorange.set('OFF')
    ar = driver.autorange.get()
    assert ar == 'OFF'


def test_increase_decrease_range(driver) -> None:
    driver_range_user = driver.ranges[2]
    driver.increase_range(driver_range_user)
    assert driver.range() == driver.ranges[3]
    driver.increase_range(driver_range_user, 2)
    assert driver.range() == driver.ranges[4]
    driver.decrease_range(driver_range_user)
    assert driver.range() == driver.ranges[1]
    driver.decrease_range(driver_range_user, -2)
    assert driver.range() == driver.ranges[0]
    driver_range_user = driver.ranges[3]
    driver.decrease_range(driver_range_user, -2)
    assert driver.range() == driver.ranges[1]


def test_display_text(driver) -> None:

    original_text = driver.display.text()
    assert original_text == ""

    new_text = "qwe"
    driver.display.text(new_text)
    assert driver.display.text() == new_text

    driver.display.clear()
    # pyvisa-sim always has a response so read here even if the instrument
    # normally doesn't return anything from the visa clear command.
    res = driver.visa_handle.read()
    # since the clear function also calls `text.get` behind the scenes
    # after calling clear on the instrument this read will be out of sync
    # and return the result from the read command. However, this is the
    # unprocessed string from the device so it will contain an extra set of
    # quotation marks.
    assert res == f'"{new_text}"'

    # since pyvisa sim does not actually implement the clear
    # command we have to manually reset the text to avoid leaking state from
    # this test
    driver.display.text(original_text)
    restored_text = driver.display.text()
    assert restored_text == original_text
