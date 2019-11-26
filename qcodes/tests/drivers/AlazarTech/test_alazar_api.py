"""Tests for Alazar DLL API

This suite of tests is expected to be executed on a Windows PC with a single
Alazar board installed.
"""

import os
import gc
import logging

import pytest

from qcodes.instrument_drivers.AlazarTech.ATS import AlazarTech_ATS
from qcodes.instrument_drivers.AlazarTech.ats_api import AlazarATSAPI
from qcodes.instrument_drivers.AlazarTech.dll_wrapper import DllWrapperMeta
from qcodes.instrument_drivers.AlazarTech.constants import ERROR_CODES, \
    API_SUCCESS, Capability


def _skip_if_alazar_dll_and_boards_not_installed():
    if not os.path.exists(AlazarTech_ATS.dll_path + '.dll'):
        return pytest.mark.skip(
            "Alazar API DLL was not found in 'AlazarTech_ATS.dll_path'.")

    return pytest.mark.skipif(
        len(AlazarTech_ATS.find_boards()) != 1,
        reason='No, or more than one Alazar boards are installed on this PC.')


pytestmark = _skip_if_alazar_dll_and_boards_not_installed()


# Set the following constants to correct values, they are used in tests below.
SYSTEM_ID = 1
BOARD_ID = 1


@pytest.fixture
def alazar():
    alazar = AlazarTech_ATS('alazar', system_id=SYSTEM_ID, board_id=BOARD_ID)
    yield alazar
    alazar.close()


@pytest.fixture
def alazar_api():
    yield AlazarATSAPI(AlazarTech_ATS.dll_path)


def test_alazar_api_singleton_behavior(caplog):
    using_msg = lambda dll_path: (
        f"Using existing instance for DLL path {dll_path}.")
    creating_msg = lambda dll_path: (
        f"Creating new instance for DLL path {dll_path}.")

    assert DllWrapperMeta._instances == {}

    with caplog.at_level(logging.DEBUG):
        api1 = AlazarATSAPI(AlazarTech_ATS.dll_path)
    assert DllWrapperMeta._instances == {AlazarTech_ATS.dll_path: api1}
    assert caplog.records[-1].message == creating_msg(AlazarTech_ATS.dll_path)
    caplog.clear()

    with caplog.at_level(logging.DEBUG):
        api2 = AlazarATSAPI(AlazarTech_ATS.dll_path)
    assert api2 is api1
    assert DllWrapperMeta._instances == {AlazarTech_ATS.dll_path: api1}
    assert caplog.records[-1].message == using_msg(AlazarTech_ATS.dll_path)
    caplog.clear()

    # Indeed, this actually exposes a vulnarability of the setup. As far as
    # LoadLibrary from ctypes is concerned, both "..\AlazarApi" and
    # "..\AlazarApi.dll" would result in the same loaded library with even
    # the same `_handle` value. But here we will abuse this in order to create
    # a new instance of the Alazar API class by using the same DLL file.
    # This should probably be fixed.
    dll_path_3 = AlazarTech_ATS.dll_path + '.dll'
    with caplog.at_level(logging.DEBUG):
        api3 = AlazarATSAPI(dll_path_3)
    assert api3 is not api1
    assert api3 is not api2
    assert DllWrapperMeta._instances == {AlazarTech_ATS.dll_path: api1,
                                         dll_path_3: api3}
    assert caplog.records[-1].message == creating_msg(dll_path_3)
    caplog.clear()

    del api2
    gc.collect()
    assert DllWrapperMeta._instances == {AlazarTech_ATS.dll_path: api1,
                                         dll_path_3: api3}

    del api1
    gc.collect()
    assert DllWrapperMeta._instances == {dll_path_3: api3}

    del api3
    gc.collect()
    assert DllWrapperMeta._instances == {}


def test_find_boards():
    boards = AlazarTech_ATS.find_boards()
    assert len(boards) == 1
    assert boards[0]['system_id'] == SYSTEM_ID
    assert boards[0]['board_id'] == BOARD_ID


def test_get_board_info(alazar_api):
    info = AlazarTech_ATS.get_board_info(api=alazar_api,
                                         system_id=SYSTEM_ID,
                                         board_id=BOARD_ID)
    assert {'system_id', 'board_id', 'board_kind',
            'max_samples', 'bits_per_sample'} == set(list(info.keys()))
    assert info['system_id'] == SYSTEM_ID
    assert info['board_id'] == BOARD_ID


def test_idn(alazar):
    idn = alazar.get_idn()
    assert {'firmware', 'model', 'serial', 'vendor', 'CPLD_version',
            'driver_version', 'SDK_version', 'latest_cal_date', 'memory_size',
            'asopc_type', 'pcie_link_speed', 'pcie_link_width',
            'bits_per_sample', 'max_samples'
            } == set(list(idn.keys()))
    assert idn['vendor'] == 'AlazarTech'
    assert idn['model'][:3] == 'ATS'


def test_return_codes_are_correct(alazar_api):
    """
    Test correctness of the coded return codes (success, failure, unknowns),
    and consistency with what `AlazarErrorToText` function returns.
    """
    for code, msg in ERROR_CODES.items():
        real_msg = alazar_api.error_to_text(code)
        assert real_msg in msg

    assert alazar_api.error_to_text(API_SUCCESS) == 'ApiSuccess'

    lower_unknown = API_SUCCESS - 1
    assert alazar_api.error_to_text(lower_unknown) == 'Unknown'

    upper_unknown = max(list(ERROR_CODES.keys())) + 1
    assert alazar_api.error_to_text(upper_unknown) == 'Unknown'


def test_get_channel_info_convenient(alazar):
    bps, max_s = alazar.api.get_channel_info_(alazar._handle)
    assert isinstance(bps, int)
    assert isinstance(max_s, int)


def test_get_cpld_version_convenient(alazar):
    cpld_ver = alazar.api.get_cpld_version_(alazar._handle)
    assert isinstance(cpld_ver, str)
    assert len(cpld_ver.split('.')) == 2


def test_get_driver_version_convenient(alazar_api):
    driver_ver = alazar_api.get_driver_version_()
    assert isinstance(driver_ver, str)
    assert len(driver_ver.split('.')) == 3


def test_get_sdk_version_convenient(alazar_api):
    sdk_ver = alazar_api.get_sdk_version_()
    assert isinstance(sdk_ver, str)
    assert len(sdk_ver.split('.')) == 3


def test_query_capability_convenient(alazar):
    cap = Capability.GET_SERIAL_NUMBER
    cap_value = alazar.api.query_capability_(alazar._handle, cap)
    assert isinstance(cap_value, int)


def test_writing_and_reading_registers(alazar):
    """
    The approach is to read the register that includes information about
    trigger holdoff parameter, and write the same value back to the board.
    """
    trigger_holdoff_register_offset = 58
    orig_val = alazar._read_register(trigger_holdoff_register_offset)
    alazar._write_register(trigger_holdoff_register_offset, orig_val)


def test_get_num_channels():
    assert 1 == AlazarTech_ATS.get_num_channels(1)
    assert 1 == AlazarTech_ATS.get_num_channels(8)

    assert 2 == AlazarTech_ATS.get_num_channels(3)
    assert 2 == AlazarTech_ATS.get_num_channels(10)

    assert 4 == AlazarTech_ATS.get_num_channels(15)
    assert 8 == AlazarTech_ATS.get_num_channels(255)
    assert 16 == AlazarTech_ATS.get_num_channels(65535)

    with pytest.raises(RuntimeError, match='0'):
        AlazarTech_ATS.get_num_channels(0)
    with pytest.raises(RuntimeError, match='17'):
        AlazarTech_ATS.get_num_channels(17)
    with pytest.raises(RuntimeError, match='100'):
        AlazarTech_ATS.get_num_channels(100)
