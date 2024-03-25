import pytest
from unittest.mock import MagicMock, patch
from pyvisa import VisaIOError
from Cryomagnetics4G_visa import CryomagneticsModel4G, StatusByte, SweepMode, Cryo4GException, Cryo4GWarning

@pytest.fixture
def cryo_instrument():
    with patch('pyvisa.ResourceManager.open_resource', return_value=MagicMock()):
        cryo = CryomagneticsModel4G('test_cryo', 'GPIB::1::INSTR')
        yield cryo
        cryo.close()

def test_initialization(cryo_instrument):
    assert cryo_instrument.name == 'test_cryo'
    assert cryo_instrument.address == 'GPIB::1::INSTR'
    assert cryo_instrument.terminator == '\r\n'

def test_get_status_byte(cryo_instrument):
    status_byte = cryo_instrument._get_status_byte(0b10101010)
    assert status_byte.sweep_mode_active == True
    assert status_byte.standby_mode_active == False
    assert status_byte.quench_condition_present == True
    assert status_byte.power_module_failure == False
    assert status_byte.message_available == True
    assert status_byte.extended_status_byte == False
    assert status_byte.master_summary_status == True
    assert status_byte.menu_mode == False

def test_set_llim(cryo_instrument):
    cryo_instrument.units = MagicMock(return_value='T')
    cryo_instrument._set_llim(-5)
    cryo_instrument.write_raw.assert_called_with('LLIM -50.0')

def test_get_llim(cryo_instrument):
    cryo_instrument.units = MagicMock(return_value='T')
    cryo_instrument.ask_raw = MagicMock(return_value='-50.0')
    assert cryo_instrument._get_llim() == -5.0

def test_set_ulim(cryo_instrument):
    cryo_instrument.units = MagicMock(return_value='T')
    cryo_instrument._set_ulim(5)
    cryo_instrument.write_raw.assert_called_with('ULIM 50.0')

def test_get_ulim(cryo_instrument):
    cryo_instrument.units = MagicMock(return_value='T')
    cryo_instrument.ask_raw = MagicMock(return_value='50.0')
    assert cryo_instrument._get_ulim() == 5.0

def test_set_field_zero(cryo_instrument):
    cryo_instrument._set_field(0)
    cryo_instrument.write_raw.assert_called_with('SWEEP ZERO')

def test_set_field_positive(cryo_instrument):
    cryo_instrument._set_ulim = MagicMock()
    cryo_instrument._wait_for_field_setpoint = MagicMock()
    cryo_instrument._set_field(5)
    cryo_instrument._set_ulim.assert_called_with(50.0)
    cryo_instrument.write_raw.assert_called_with('SWEEP UP')
    cryo_instrument._wait_for_field_setpoint.assert_called_with(50.0)

def test_set_field_negative(cryo_instrument):
    cryo_instrument._set_llim = MagicMock()
    cryo_instrument._wait_for_field_setpoint = MagicMock()
    cryo_instrument._set_field(-5)
    cryo_instrument._set_llim.assert_called_with(-50.0)
    cryo_instrument.write_raw.assert_called_with('SWEEP DOWN')
    cryo_instrument._wait_for_field_setpoint.assert_called_with(-50.0)

def test_get_field(cryo_instrument):
    cryo_instrument.units = MagicMock(return_value='T')
    cryo_instrument.ask_raw = MagicMock(return_value='50.0')
    assert cryo_instrument._get_field() == 5.0

def test_write_raw_retry(cryo_instrument):
    cryo_instrument.write_raw = MagicMock(side_effect=[VisaIOError, None])
    cryo_instrument.device_clear = MagicMock()
    cryo_instrument.write_raw('test')
    assert cryo_instrument.write_raw.call_count == 2
    assert cryo_instrument.device_clear.call_count == 1

def test_ask_raw_retry(cryo_instrument):
    cryo_instrument.ask_raw = MagicMock(side_effect=[VisaIOError, '42'])
    cryo_instrument.device_clear = MagicMock()
    assert cryo_instrument.ask_raw('test') == '42'
    assert cryo_instrument.ask_raw.call_count == 2
    assert cryo_instrument.device_clear.call_count == 1
