import pytest

import qcodes.instrument.sims as sims
from qcodes.instrument_drivers.Keysight.Keysight_34465A_submodules import \
    Keysight_34465A
visalib = sims.__file__.replace('__init__.py', 'Keysight_34465A.yaml@sim')


@pytest.fixture(scope='function')
def driver():
    keysight_sim = Keysight_34465A('keysight_34465A_sim',
                                   address='GPIB::1::INSTR',
                                   visalib=visalib)
    try:
        yield keysight_sim
    finally:
        keysight_sim.close()


def test_init(driver):
    idn = driver.IDN()
    assert idn['vendor'] == 'Keysight'


def test_has_dig_option(driver):
    assert True is driver.has_DIG


def test_model_flag(driver):
    assert True is driver.is_34465A_34470A


def test_NPLC(driver):
    assert driver.NPLC.get() == 10.0
    driver.NPLC.set(0.2)
    assert driver.NPLC.get() == 0.2
    driver.NPLC.set(10.0)


def test_get_voltage(driver):
    voltage = driver.volt.get()
    assert voltage == 10.0


def test_set_get_autorange(driver):
    ar = driver.autorange.get()
    assert ar == 'OFF'
    driver.autorange.set('ON')
    ar = driver.autorange.get()
    assert ar == 'ON'
    driver.autorange.set('OFF')
    ar = driver.autorange.get()
    assert ar == 'OFF'


def test_display_text(driver):

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

