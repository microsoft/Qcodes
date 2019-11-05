import pytest
import logging
import io

from qcodes.instrument_drivers.Keysight.Keysight_34980A import Keysight_34980A
import qcodes.instrument.sims as sims

visalib = sims.__file__.replace('__init__.py', 'Keysight_34980A.yaml@sim')


@pytest.fixture(scope="function")
def driver():
    inst = Keysight_34980A('keysight_34980A_sim',
                           address='GPIB::1::INSTR',
                           visalib=visalib)

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
    Instrument attributes are set correctly after getting the IDN
    """
    assert driver.IDN() == {
        "vendor": "Keysight",
        "model": "34980A",
        "serial": "1000",
        "firmware": "0.1"
    }


def test_scan_slots(driver):
    """

    :param driver:
    :return:
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

    :param driver:
    :return:
    """
    assert not driver._is_closed('(@3405)')
    assert driver._is_open('(@3405)')


def test_safety_interlock(driver, caplog):
    """

    :param driver:
    :return:
    """
    driver.module_in_slot[3].clear_status()
    with caplog.at_level(logging.DEBUG):
        assert "safety interlock" in caplog.text


def test_protection_mode(driver):
    """
    34934A only
    :param driver:
    :return:
    """
    assert driver.module_in_slot[1].protection_mode() == 'AUTO100'


def test_numbering_table_34934a(driver):
    pass
