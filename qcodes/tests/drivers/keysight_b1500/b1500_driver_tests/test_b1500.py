from unittest.mock import MagicMock

import pytest
from pyvisa import VisaIOError

from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1500 import \
    KeysightB1500
from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1517A import \
    B1517A
from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1520A import \
    B1520A
from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1530A import \
    B1530A
from qcodes.instrument_drivers.Keysight.keysightb1500.constants import \
    SlotNr, ChNr, CALResponse


@pytest.fixture
def b1500():
    try:
        resource_name = 'insert_Keysight_B2200_VISA_resource_name_here'
        instance = KeysightB1500('SPA',
                                 address=resource_name)
    except (ValueError, VisaIOError):
        # Either there is no VISA lib installed or there was no real
        # instrument found at the specified address => use simulated instrument
        import qcodes.instrument.sims as sims
        path_to_yaml = sims.__file__.replace('__init__.py',
                                             'keysight_b1500.yaml')

        instance = KeysightB1500('SPA',
                                 address='GPIB::1::INSTR',
                                 visalib=path_to_yaml + '@sim'
                                 )

    instance.get_status()
    instance.reset()

    yield instance

    instance.close()


def test_make_module_from_model_name():
    mainframe = MagicMock()

    with pytest.raises(NotImplementedError):
        KeysightB1500.from_model_name(model='unsupported_module', slot_nr=0,
                                      parent=mainframe, name='dummy')

    smu = KeysightB1500.from_model_name(model='B1517A', slot_nr=1,
                                        parent=mainframe, name='dummy')

    assert isinstance(smu, B1517A)

    cmu = KeysightB1500.from_model_name(model='B1520A', slot_nr=2,
                                        parent=mainframe)

    assert isinstance(cmu, B1520A)

    wgfmu = KeysightB1500.from_model_name(model='B1530A', slot_nr=3,
                                          parent=mainframe)

    assert isinstance(wgfmu, B1530A)


def test_init(b1500):
    assert hasattr(b1500, 'smu1')
    assert hasattr(b1500, 'smu2')
    assert hasattr(b1500, 'cmu1')
    assert hasattr(b1500, 'wgfmu1')


def test_submodule_access_by_class(b1500):
    assert b1500.smu1 in b1500.by_kind['SMU']
    assert b1500.smu2 in b1500.by_kind['SMU']
    assert b1500.cmu1 in b1500.by_kind['CMU']
    assert b1500.wgfmu1 in b1500.by_kind['WGFMU']


def test_submodule_access_by_slot(b1500):
    assert b1500.smu1 is b1500.by_slot[SlotNr.SLOT01]
    assert b1500.smu2 is b1500.by_slot[SlotNr.SLOT02]
    assert b1500.cmu1 is b1500.by_slot[3]
    assert b1500.wgfmu1 is b1500.by_slot[6]


def test_submodule_access_by_channel(b1500):
    assert b1500.smu1 is b1500.by_channel[ChNr.SLOT_01_CH1]
    assert b1500.smu2 is b1500.by_channel[ChNr.SLOT_02_CH1]
    assert b1500.cmu1 is b1500.by_channel[ChNr.SLOT_03_CH1]
    assert b1500.wgfmu1 is b1500.by_channel[ChNr.SLOT_06_CH1]
    assert b1500.wgfmu1 is b1500.by_channel[6]
    assert b1500.wgfmu1 is b1500.by_channel[ChNr.SLOT_06_CH2]


def test_enable_multiple_channels(b1500):
    mock_write = MagicMock()
    b1500.write = mock_write

    b1500.enable_channels([1, 2, 3])

    mock_write.assert_called_once_with("CN 1,2,3")


def test_disable_multiple_channels(b1500):
    mock_write = MagicMock()
    b1500.write = mock_write

    b1500.disable_channels([1, 2, 3])

    mock_write.assert_called_once_with("CL 1,2,3")


def test_use_nplc_for_high_speed_adc(b1500):
    mock_write = MagicMock()
    b1500.write = mock_write

    b1500.use_nplc_for_high_speed_adc()
    mock_write.assert_called_once_with("AIT 0,2")

    mock_write.reset_mock()

    b1500.use_nplc_for_high_speed_adc(3)
    mock_write.assert_called_once_with("AIT 0,2,3")


def test_use_nplc_for_high_resolution_adc(b1500):
    mock_write = MagicMock()
    b1500.write = mock_write

    b1500.use_nplc_for_high_resolution_adc()
    mock_write.assert_called_once_with("AIT 1,2")

    mock_write.reset_mock()

    b1500.use_nplc_for_high_resolution_adc(8)
    mock_write.assert_called_once_with("AIT 1,2,8")


def test_autozero_enabled(b1500):
    mock_write = MagicMock()
    b1500.write = mock_write

    assert b1500.autozero_enabled() is False

    b1500.autozero_enabled(True)
    mock_write.assert_called_once_with("AZ 1")
    assert b1500.autozero_enabled() is True

    mock_write.reset_mock()

    b1500.autozero_enabled(False)
    mock_write.assert_called_once_with("AZ 0")
    assert b1500.autozero_enabled() is False


def test_use_manual_mode_for_high_speed_adc(b1500):
    mock_write = MagicMock()
    b1500.write = mock_write

    b1500.use_manual_mode_for_high_speed_adc()
    mock_write.assert_called_once_with("AIT 0,1")

    mock_write.reset_mock()

    b1500.use_manual_mode_for_high_speed_adc(n=1)
    mock_write.assert_called_once_with("AIT 0,1,1")

    mock_write.reset_mock()

    b1500.use_manual_mode_for_high_speed_adc(n=8)
    mock_write.assert_called_once_with("AIT 0,1,8")


def test_self_calibration_successful(b1500):
    mock_ask = MagicMock()
    b1500.ask = mock_ask

    mock_ask.return_value = '0'

    response = b1500.self_calibration()

    assert response == CALResponse(0)
    mock_ask.assert_called_once_with('*CAL?')


def test_self_calibration_failed(b1500):
    mock_ask = MagicMock()
    b1500.ask = mock_ask

    expected_response = CALResponse(1) + CALResponse(64)
    mock_ask.return_value = '65'

    response = b1500.self_calibration()

    assert response == expected_response
    mock_ask.assert_called_once_with('*CAL?')


def test_error_message(b1500):
    mock_ask = MagicMock()
    b1500.ask = mock_ask
    mock_ask.return_value = '0,"No Error."'

    response = b1500.error_message()
    assert '0,"No Error."' == response
    mock_ask.assert_called_once_with(f'ERRX?')
