"""Tests for the MiniCircuits USB HID mixin using fake hardware libraries.

The mixin requires ``pywinusb`` on Windows and a connected device, and the
``Minicircuits`` package imports ``clr`` from ``pythonnet``, so both are
replaced by minimal fakes. The fake device records the calls made.
"""

import importlib
import sys
import types
from typing import TYPE_CHECKING, Any

import pytest

from qcodes.instrument import Instrument

if TYPE_CHECKING:
    from collections.abc import Generator

_PACKAGE = "qcodes.instrument_drivers.Minicircuits"
_MIXIN_MODULE = f"{_PACKAGE}.USBHIDMixin"


class _FakeHidDevice:
    def __init__(self) -> None:
        self.is_open = False

    def open(self) -> None:
        self.is_open = True

    def close(self) -> None:
        self.is_open = False

    def set_raw_data_handler(self, handler: Any) -> None:
        pass


class _FakeHid:
    def __init__(self) -> None:
        self.device = _FakeHidDevice()

    def HidDeviceFilter(self, **kwargs: Any) -> Any:
        device = self.device

        class _Filter:
            def get_devices(self) -> list[_FakeHidDevice]:
                return [device]

        return _Filter()


def _driver_modules() -> list[str]:
    return [
        name
        for name in sys.modules
        if name == _PACKAGE or name.startswith(f"{_PACKAGE}.")
    ]


@pytest.fixture(name="fake_hid")
def _fake_hid(monkeypatch: pytest.MonkeyPatch) -> "Generator[_FakeHid, None, None]":
    monkeypatch.setitem(sys.modules, "clr", types.ModuleType("clr"))
    for name in _driver_modules():
        monkeypatch.delitem(sys.modules, name)
    try:
        mixin_module = importlib.import_module(_MIXIN_MODULE)
        fake_hid = _FakeHid()
        monkeypatch.setattr(mixin_module, "hid", fake_hid)
        # the real check requires Windows and an installed pywinusb
        monkeypatch.setattr(
            mixin_module.MiniCircuitsHIDMixin,
            "_check_hid_import",
            staticmethod(lambda: None),
        )
        yield fake_hid
    finally:
        # Do not leave the drivers bound to the fake modules behind.
        for name in _driver_modules():
            sys.modules.pop(name)


def test_close_closes_device_and_unregisters_instrument(fake_hid: _FakeHid) -> None:
    mixin_class = importlib.import_module(_MIXIN_MODULE).MiniCircuitsHIDMixin
    instrument = mixin_class("minicircuits")
    assert fake_hid.device.is_open

    instrument.close()

    assert not fake_hid.device.is_open
    assert not Instrument.exist("minicircuits")
    # the name can be reused once the instrument has been closed
    mixin_class("minicircuits").close()
