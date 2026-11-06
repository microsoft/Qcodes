import ctypes
from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.AlazarTech.ATS import Buffer

if TYPE_CHECKING:
    from qcodes.instrument_drivers.AlazarTech.ATS import CtypesTypes


def test_buffer_is_allocated_when_initiated() -> None:
    b = Buffer(ctypes.c_uint8, 25)
    assert b._allocated is True


@pytest.mark.parametrize(
    "ctype",
    (ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint32, ctypes.c_int32, ctypes.c_float),
)
def test_supported_ctypes_for_sample(ctype: "CtypesTypes") -> None:
    Buffer(ctype, 8)
