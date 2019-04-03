"""Tests DLL wrapper infrastructure"""


import logging
import gc
import os

import pytest

from qcodes.instrument_drivers.AlazarTech.dll_wrapper import WrappedDll, \
    DllWrapperMeta


pytestmark = pytest.mark.skipif(
    os.name != 'nt', reason='These tests are relevant only for Windows')


def test_wrapped_dll_singleton_behavior(caplog):
    using_msg = lambda dll_path: (
        f"Using existing instance for DLL path {dll_path}.")
    creating_msg = lambda dll_path: (
        f"Creating new instance for DLL path {dll_path}.")
    dll_path_1 = 'ntdll.dll'
    dll_path_3 = 'kernel32.dll'

    assert DllWrapperMeta._instances == {}

    with caplog.at_level(logging.DEBUG):
        dll_1 = WrappedDll(dll_path_1)
    assert DllWrapperMeta._instances == {dll_path_1: dll_1}
    assert caplog.records[-1].message == creating_msg(dll_path_1)
    caplog.clear()

    with caplog.at_level(logging.DEBUG):
        dll_2 = WrappedDll(dll_path_1)
    assert dll_2 is dll_1
    assert DllWrapperMeta._instances == {dll_path_1: dll_1}
    assert caplog.records[-1].message == using_msg(dll_path_1)
    caplog.clear()

    with caplog.at_level(logging.DEBUG):
        dll_3 = WrappedDll(dll_path_3)
    assert dll_3 is not dll_1
    assert dll_3 is not dll_2
    assert DllWrapperMeta._instances == {dll_path_1: dll_1,
                                         dll_path_3: dll_3}
    assert caplog.records[-1].message == creating_msg(dll_path_3)
    caplog.clear()

    del dll_2
    gc.collect()
    assert DllWrapperMeta._instances == {dll_path_1: dll_1,
                                         dll_path_3: dll_3}

    del dll_1
    gc.collect()
    assert DllWrapperMeta._instances == {dll_path_3: dll_3}

    del dll_3
    gc.collect()
    assert DllWrapperMeta._instances == {}
