import pytest
import logging
import io

from qcodes.instrument_drivers.Keysight.keysight_34980A import Keysight_34980A
import qcodes.instrument.sims as sims

VISALIB = sims.__file__.replace('__init__.py', 'keysight_34980A.yaml@sim')


@pytest.fixture(scope="module")
def driver():
    inst = Keysight_34980A('keysight_34980A_sim',
                           address='GPIB::1::INSTR',
                           visalib=VISALIB)

    inst.log.setLevel(logging.DEBUG)
    iostream = io.StringIO()
    lh = logging.StreamHandler(iostream)
    inst.log.logger.addHandler(lh)

    try:
        yield inst
    finally:
        inst.close()


def test_get_idn(driver):
    """
    to check if the instrument attributes are set correctly after getting the IDN
    """
    assert driver.IDN() == {
        "vendor": "Keysight",
        "model": "34980A",
        "serial": "1000",
        "firmware": "0.1"
    }


def test_scan_slots(driver):
    """
    to check if the submodule attributes are set correctly after scanning every slot
    """
    assert driver.system_slots_info[1] == {
        "vendor": "Agilent Technologies",
        "module": "34934A-8x64",
        "serial": "AB10000000",
        "firmware": "1.00"
    }

    assert driver.system_slots_info[3] == {
        "vendor": "Agilent Technologies",
        "module": "34934A-4x32",
        "serial": "AB10000001",
        "firmware": "1.00"
    }


def test_connection(driver):
    """
    to check if a channel is closed or open
    """
    assert driver.module[1].is_closed(2, 3) is False
    assert driver.module[1].is_open(2, 3) is True


def test_safety_interlock(driver, caplog):
    """
    to check if a module is at safety interlock state
    """
    driver.module[3].clear_status()
    with caplog.at_level(logging.DEBUG):
        assert "safety interlock" in caplog.text


def test_protection_mode(driver):
    """
    to check the protection mode (34934A module only)
    """
    assert driver.module[1].protection_mode() == 'AUTO100'

