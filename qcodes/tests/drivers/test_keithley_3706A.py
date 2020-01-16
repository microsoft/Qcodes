import pytest

from qcodes.instrument_drivers.tektronix.Keithley_3706A_Matrix_channels\
    import Keithley_3706A

import qcodes.instrument.sims as sims
visalib = sims.__file__.replace('__init__.py', 'Keithley_3706A.yaml@sim')


@pytest.fixture(scope='function')
def driver():
    driver = Keithley_3706A('Keithley_3706A',
                            address='GPIB::13::INSTR',
                            visalib=visalib)
    yield driver
    driver.close()


def test_idn(driver):
    assert {'firmware': '01.56a',
            'model': '3706A-SNFP',
            'serial': '04447336',
            'vendor': 'KEITHLEY INSTRUMENTS INC.'} == driver.get_idn()


def test_switch_card_idn(driver):
    assert {'slot_no': "1", 'model': "3730", 'mtype': "6x16 High Density Matrix",
            'firmware': "01.40h", 'serial': "4447332"} == driver.get_switch_cards()[0]

