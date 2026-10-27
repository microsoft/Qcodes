"""
Tests for the Galil DMC-41x3 driver.

The driver needs the ``gclib`` package from Galil, which is not available on
PyPI, so a minimal stand-in for ``gclib`` is installed for these tests.
"""

import importlib
import sys
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from qcodes.instrument import Instrument

if TYPE_CHECKING:
    from collections.abc import Generator

    from qcodes.instrument_drivers.Galil.dmc_41x3 import GalilDMC4133Controller

_DRIVER_MODULES = (
    "qcodes.instrument_drivers.Galil.dmc_41x3",
    "qcodes.instrument_drivers.Galil",
)


class FakeGclibError(Exception):
    pass


class FakeGclibConnection:
    """
    Stand-in for ``gclib.py``. Queries are answered from ``replies``, a
    list of replies per command that are returned in turn.
    """

    def __init__(self) -> None:
        self.replies: dict[str, list[str]] = {}
        self.commands: list[str] = []

    def GOpen(self, address: str) -> None:
        pass

    def GInfo(self) -> str:
        return "192.168.1.30, DMC4133 Rev 1.2c, 12345"

    def GCommand(self, command: str) -> str:
        command = command.removesuffix("\r")
        self.commands.append(command)
        replies = self.replies.get(command)
        return replies.pop(0) if replies else ""

    def GTimeout(self, timeout: int) -> None:
        pass

    def GClose(self) -> None:
        pass


def _make_fake_gclib() -> ModuleType:
    gclib = ModuleType("gclib")
    gclib.py = FakeGclibConnection  # type: ignore[attr-defined]
    gclib.GclibError = FakeGclibError  # type: ignore[attr-defined]
    return gclib


@pytest.fixture(name="galil")
def _make_galil(
    monkeypatch: pytest.MonkeyPatch,
) -> "Generator[GalilDMC4133Controller, None, None]":
    monkeypatch.setitem(sys.modules, "gclib", _make_fake_gclib())
    for name in _DRIVER_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    dmc_41x3 = importlib.import_module("qcodes.instrument_drivers.Galil.dmc_41x3")

    driver = dmc_41x3.GalilDMC4133Controller("galil", address="192.168.1.30")
    try:
        yield driver
    finally:
        driver.close()
        # GalilMotionController.close does not call Instrument.close, so the
        # instrument would stay registered under its name
        Instrument.close(driver)
        # do not leave the driver imported with the stand-in gclib
        for name in _DRIVER_MODULES:
            sys.modules.pop(name, None)


def _connection(galil: "GalilDMC4133Controller") -> FakeGclibConnection:
    assert isinstance(galil.g, FakeGclibConnection)
    return galil.g


def test_idn(galil: "GalilDMC4133Controller") -> None:
    assert galil.IDN() == {
        "vendor": "Galil Motion Control, Inc.",
        "model": "DMC4133",
        "serial": "12345",
        "firmware": "1.2c",
    }


def test_absolute_position(galil: "GalilDMC4133Controller") -> None:
    _connection(galil).replies["PA ?,?,?"] = [":", "100, 200, 300:"]

    assert galil.absolute_position() == {"A": 100, "B": 200, "C": 300}


def test_absolute_position_is_read_only(galil: "GalilDMC4133Controller") -> None:
    # setting the absolute position used to only update the cached value
    # without moving the motors
    assert not galil.absolute_position.settable
    with pytest.raises(NotImplementedError):
        galil.absolute_position({"A": 1, "B": 2, "C": 3})


@pytest.mark.parametrize(("reply", "expected"), [("1.0000", "off"), ("0.0000", "on")])
def test_on_off_status(
    galil: "GalilDMC4133Controller", reply: str, expected: str
) -> None:
    # gclib can return an empty reply or the bare ':' prompt under rapid
    # polling, which is retried
    _connection(galil).replies["MG _MOA"] = ["", ":", reply]

    assert galil.motor_a.on_off_status() == expected


def test_off_when_error_occurs(galil: "GalilDMC4133Controller") -> None:
    _connection(galil).replies["MG _OEA"] = [":", "3.0000"]

    assert galil.motor_a.off_when_error_occurs() == "enable for all"


def test_unparsable_reply_raises(galil: "GalilDMC4133Controller") -> None:
    _connection(galil).replies["MG _MOA"] = [":"] * 20

    with pytest.raises(RuntimeError, match="Could not parse reply to 'MG _MOA'"):
        galil.motor_a.on_off_status()
