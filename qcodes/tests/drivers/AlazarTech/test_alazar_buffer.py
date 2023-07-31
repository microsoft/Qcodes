import ctypes

import pytest

from qcodes.instrument_drivers.AlazarTech.ATS import Buffer


def test_buffer_is_allocated_when_initiated() -> None:
    b = Buffer(ctypes.c_uint8, 25)
    assert b._allocated is True


@pytest.mark.parametrize('ctype', (ctypes.c_uint8, ctypes.c_uint16,
                                   ctypes.c_uint32, ctypes.c_int32,
                                   ctypes.c_float))
def test_supported_ctypes_for_sample(ctype) -> None:
    Buffer(ctype, 8)
