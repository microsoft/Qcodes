import pytest

from qcodes.instrument_drivers.tektronix.Keithley_3706A_Matrix_channels\
    import Keithley_3706A

import qcodes.instrument.sims as sims
visalib = sims.__file__.replace('__init__.py', 'Keithley_3706A.yaml@sim')


@pytest.fixture(scope='function')
def driver():
    driver = Keithley_3706A('Keithley_3706A',
                            address='GPIB::11::INSTR',
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


def test_installed_card_id(driver):
    assert ['1', '2', '3'] == driver._get_slot_id()


def test_slot_names(driver):
    assert ['slot1', 'slot2', 'slot3'] == driver._get_slot_name()


def test_get_interlock_state(driver):
    dict_list = []
    for i in ['1', '2', '3']:
        dictionary = {'slot_no': i,
                      'state': 'Interlock is disengaged'}
        dict_list.append(dictionary)
    assert tuple(dict_list) == driver.get_interlock_state()


def test_get_number_of_rows(driver):
    assert [6, 6, 6] == driver._get_number_of_rows()


def test_get_number_of_columns(driver):
    assert [16, 16, 16] == driver._get_number_of_columns()


def test_get_rows(driver):
    rows_in_a_slot = ['1', '2', '3', '4', '5', '6']
    assert rows_in_a_slot == driver._get_rows()[0]
    assert rows_in_a_slot == driver._get_rows()[1]
    assert rows_in_a_slot == driver._get_rows()[2]


def test_get_columns(driver):
    columns_in_a_slot = ['01', '02', '03', '04', '05', '06', '07', '08', '09',
                         '10', '11', '12', '13', '14', '15', '16']
    assert columns_in_a_slot == driver._get_columns()[0]
    assert columns_in_a_slot == driver._get_columns()[1]
    assert columns_in_a_slot == driver._get_columns()[2]
