from unittest.mock import MagicMock

from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1517A import \
    B1517A
from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1500_module import \
    parse_module_query_response, format_dcorr_response, _DCORRResponse
from qcodes.instrument_drivers.Keysight.keysightb1500.constants import \
    SlotNr, DCORR


def test_is_enabled():
    mainframe = MagicMock()

    # Use concrete subclass because B1500Module does not assign channels
    smu = B1517A(parent=mainframe, name='B1517A', slot_nr=1)

    mainframe.ask.return_value = 'CN 1,2,4,8'
    assert smu.is_enabled()
    mainframe.ask.assert_called_once_with('*LRN? 0')

    mainframe.reset_mock(return_value=True)
    mainframe.ask.return_value = 'CN 2,4,8'
    assert not smu.is_enabled()
    mainframe.ask.assert_called_once_with('*LRN? 0')


def test_enable_outputs():
    mainframe = MagicMock()

    slot_nr = 1
    # Use concrete subclass because B1500Module does not assign channels
    smu = B1517A(parent=mainframe, name='B1517A', slot_nr=slot_nr)

    smu.enable_outputs()
    mainframe.write.assert_called_once_with(f'CN {slot_nr}')


def test_disable_outputs():
    mainframe = MagicMock()

    slot_nr = 1
    # Use concrete subclass because B1500Module does not assign channels
    smu = B1517A(parent=mainframe, name='B1517A', slot_nr=slot_nr)

    smu.disable_outputs()
    mainframe.write.assert_called_once_with(f'CL {slot_nr}')


def test_parse_module_query_response():
    response = 'B1517A,0;B1517A,0;B1520A,0;0,0;0,0;0,0;0,0;0,0;0,0;0,0'
    expected = {SlotNr.SLOT01: 'B1517A',
                SlotNr.SLOT02: 'B1517A',
                SlotNr.SLOT03: 'B1520A'}

    actual = parse_module_query_response(response)

    assert actual == expected


def test_format_dcorr_response():
    resp_str1 = format_dcorr_response(
        _DCORRResponse(mode=DCORR.Mode.Cp_G, primary=0.001, secondary=0.34))
    assert resp_str1 == 'Mode: Cp_G, Primary Cp: 0.001 F, Secondary G: 0.34 S'

    resp_str2 = format_dcorr_response(
        _DCORRResponse(mode=DCORR.Mode.Ls_Rs, primary=0.2, secondary=3.0))
    assert resp_str2 == 'Mode: Ls_Rs, Primary Ls: 0.2 H, Secondary Rs: 3.0 Ω'
