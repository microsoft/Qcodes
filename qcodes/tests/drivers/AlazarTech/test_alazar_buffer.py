import ctypes
import os

import pytest

from qcodes.instrument_drivers.AlazarTech.ATS import Buffer
from qcodes.instrument_drivers.AlazarTech.ATS import os as ats_os


pytestmark = pytest.mark.skipif(
    os.name != 'nt', reason='These tests are relevant only for Windows')


def test_buffer_initiates_only_on_windows(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(ats_os, 'name', 'nt')

        Buffer(ctypes.c_uint8, 25)

    with monkeypatch.context() as m:
        m.setattr(ats_os, 'name', 'other_os')

        with pytest.raises(Exception, match="Unsupported OS"):
            Buffer(ctypes.c_uint8, 25)


def test_buffer_is_allocated_when_initiated():
    b = Buffer(ctypes.c_uint8, 25)
    assert b._allocated is True


@pytest.mark.parametrize('ctype', (ctypes.c_uint8, ctypes.c_uint16,
                                   ctypes.c_uint32, ctypes.c_int32,
                                   ctypes.c_float))
def test_supported_ctypes_for_sample(ctype):
    Buffer(ctype, 8)
