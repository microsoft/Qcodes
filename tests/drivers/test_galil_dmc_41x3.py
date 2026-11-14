"""Tests for the Galil DMC-41x3 driver using a fake ``gclib`` module.

The driver requires the vendor ``gclib`` package, which cannot be installed
from PyPI, so it is replaced by a minimal fake that records the calls made.
"""

import importlib
import sys
import types
from typing import TYPE_CHECKING

import pytest

from qcodes.instrument import Instrument

if TYPE_CHECKING:
    from collections.abc import Generator

_DRIVER_MODULE = "qcodes.instrument_drivers.Galil.dmc_41x3"


class _FakeGclibConnection:
    def __init__(self) -> None:
        self.is_open = False

    def GOpen(self, address: str) -> None:
        self.is_open = True

    def GClose(self) -> None:
        self.is_open = False

    def GInfo(self) -> str:
        return "192.168.1.2, DMC4133 Rev 1.0b, 12345"


@pytest.fixture(name="driver_module")
def _driver_module_with_fake_gclib(
    monkeypatch: pytest.MonkeyPatch,
) -> "Generator[types.ModuleType, None, None]":
    fake_gclib = types.ModuleType("gclib")
    fake_gclib.py = _FakeGclibConnection  # type: ignore[attr-defined]
    fake_gclib.GclibError = Exception  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gclib", fake_gclib)
    for name in ("qcodes.instrument_drivers.Galil", _DRIVER_MODULE):
        monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        yield importlib.import_module(_DRIVER_MODULE)
    finally:
        # Do not leave the driver bound to the fake module behind.
        for name in ("qcodes.instrument_drivers.Galil", _DRIVER_MODULE):
            sys.modules.pop(name, None)


def test_close_closes_connection_and_unregisters_instrument(
    driver_module: types.ModuleType,
) -> None:
    controller = driver_module.GalilMotionController("galil", address="192.168.1.2")
    connection = controller.g
    assert connection.is_open

    controller.close()

    assert not connection.is_open
    assert not Instrument.exist("galil")
    # closing it again does nothing
    controller.close()
    # the name can be reused once the instrument has been closed
    driver_module.GalilMotionController("galil", address="192.168.1.2").close()
