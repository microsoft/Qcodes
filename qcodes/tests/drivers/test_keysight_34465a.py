import pytest

import qcodes.instrument.sims as sims
from qcodes.instrument_drivers.Keysight.Keysight_34465A import Keysight_34465A
visalib = sims.__file__.replace('__init__.py', 'Keysight_34465A.yaml@sim')


@pytest.fixture(scope='function')
def driver():
    keysight_sim = Keysight_34465A('keysight_34465A_sim',
                                   address='GPIB::1::65535::INSTR',
                                   visalib=visalib)

    yield keysight_sim

    keysight_sim.close()


def test_init(driver):
    idn = driver.IDN()
    assert idn['vendor'] == 'Keysight'


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
