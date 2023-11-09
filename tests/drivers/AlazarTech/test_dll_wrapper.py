"""Tests DLL wrapper infrastructure"""


import gc
import logging
import os
from weakref import WeakValueDictionary

import pytest
from pytest import LogCaptureFixture

from qcodes.instrument_drivers.AlazarTech.dll_wrapper import DllWrapperMeta, WrappedDll

pytestmark = pytest.mark.skipif(
    os.name != 'nt', reason='These tests are relevant only for Windows')


def test_wrapped_dll_singleton_behavior(caplog: LogCaptureFixture) -> None:
    def using_msg(dll_path):
        return f"Using existing instance for DLL path {dll_path}."

    def creating_msg(dll_path):
        return f"Creating new instance for DLL path {dll_path}."
    dll_path_1 = 'ntdll.dll'
    dll_path_3 = 'kernel32.dll'

    assert DllWrapperMeta._instances == WeakValueDictionary()

    with caplog.at_level(logging.DEBUG):
        dll_1 = WrappedDll(dll_path_1)
    assert DllWrapperMeta._instances == WeakValueDictionary({dll_path_1: dll_1})
    assert caplog.records[-1].message == creating_msg(dll_path_1)
    caplog.clear()

    with caplog.at_level(logging.DEBUG):
        dll_2 = WrappedDll(dll_path_1)
    assert dll_2 is dll_1
    assert DllWrapperMeta._instances == WeakValueDictionary({dll_path_1: dll_1})
    assert caplog.records[-1].message == using_msg(dll_path_1)
    caplog.clear()

    with caplog.at_level(logging.DEBUG):
        dll_3 = WrappedDll(dll_path_3)
    assert dll_3 is not dll_1
    assert dll_3 is not dll_2
    assert DllWrapperMeta._instances == WeakValueDictionary(
        {dll_path_1: dll_1, dll_path_3: dll_3}
    )
    assert caplog.records[-1].message == creating_msg(dll_path_3)
    caplog.clear()

    del dll_2
    gc.collect()
    assert DllWrapperMeta._instances == WeakValueDictionary(
        {dll_path_1: dll_1, dll_path_3: dll_3}
    )

    del dll_1
    gc.collect()
    assert DllWrapperMeta._instances == WeakValueDictionary({dll_path_3: dll_3})

    del dll_3
    gc.collect()
    assert DllWrapperMeta._instances == WeakValueDictionary({})
