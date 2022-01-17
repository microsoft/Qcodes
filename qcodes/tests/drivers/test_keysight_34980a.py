# pylint: disable=redefined-outer-name
import pytest
import logging

from qcodes.instrument_drivers.Keysight.keysight_34980a import Keysight34980A
import qcodes.instrument.sims as sims

VISALIB = sims.__file__.replace('__init__.py', 'keysight_34980A.yaml@sim')


@pytest.fixture(scope="module")
def switch_driver():
    inst = Keysight34980A('keysight_34980A_sim',
                          address='GPIB::1::INSTR',
                          visalib=VISALIB)

    try:
        yield inst
    finally:
        inst.close()


def test_safety_interlock_during_init(switch_driver, caplog):
    """
    to check if a warning would show when initialize the instrument with a
    module in safety interlock state. This test has to be placed first if
    the scope is set to be "module".
    """
    msg = [
        x.message for x in caplog.get_records('setup')
        if x.levelno == logging.WARNING
    ]
    assert "safety interlock" in msg[0]


def test_get_idn(switch_driver):
    """
    to check if the instrument attributes are set correctly after getting
    the IDN
    """
    assert switch_driver.IDN() == {
        "vendor": "Keysight",
        "model": "34980A",
        "serial": "1000",
        "firmware": "0.1"
    }


def test_scan_slots(switch_driver):
    """
    to check if the submodule attributes are set correctly after scanning
    every slot
    """
    assert len(switch_driver.system_slots_info) == 2

    assert switch_driver.system_slots_info[1] == {
        "vendor": "Agilent Technologies",
        "model": "34934A-8x64",
        "serial": "AB10000000",
        "firmware": "1.00"
    }

    assert switch_driver.system_slots_info[3] == {
        "vendor": "Agilent Technologies",
        "model": "34934A-4x32",
        "serial": "AB10000001",
        "firmware": "1.00"
    }


def test_safety_interlock(switch_driver, caplog):
    """
    to check if a warning would show when talk to a module that is in safety
    interlock state
    """
    switch_driver.module[3].write('*CLS')
    with caplog.at_level(logging.DEBUG):
        assert "safety interlock" in caplog.text
