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

from qcodes import instrument_drivers
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
        self.closed = False
        self.timeout = 5000

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
        self.timeout = timeout

    def GClose(self) -> None:
        self.closed = True


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
    # importing the driver also binds ``Galil`` on the parent package
    original_galil = instrument_drivers.__dict__.get("Galil")
    dmc_41x3 = importlib.import_module("qcodes.instrument_drivers.Galil.dmc_41x3")

    driver = dmc_41x3.GalilDMC4133Controller("galil", address="192.168.1.30")
    try:
        yield driver
    finally:
        driver.close()
        # do not leave the driver imported with the stand-in gclib
        for name in _DRIVER_MODULES:
            sys.modules.pop(name, None)
        if original_galil is None:
            instrument_drivers.__dict__.pop("Galil", None)
        else:
            instrument_drivers.__dict__["Galil"] = original_galil


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

    with pytest.raises(RuntimeError, match="Got only empty replies to 'MG _MOA'"):
        galil.motor_a.on_off_status()


def test_parameters_retry_empty_replies(galil: "GalilDMC4133Controller") -> None:
    # every query retries empty replies, not only the ones that used to
    # have their own retry loop
    _connection(galil).replies["MG _SPA"] = ["", ":", "2000.0000"]
    _connection(galil).replies["VS ?"] = [":", "4000.0000:"]

    assert galil.motor_a.speed() == 2000
    assert galil.plane_ab.vector_speed() == 4000


def test_malformed_position_raises(galil: "GalilDMC4133Controller") -> None:
    _connection(galil).replies["PA ?,?,?"] = ["100, 200"]

    with pytest.raises(ValueError, match="Expected three positions"):
        galil.absolute_position()


def test_wait_till_motion_complete(galil: "GalilDMC4133Controller") -> None:
    # an empty reply while waiting counts as "not yet finished" and is polled
    # again; the wait ends when every axis reports 0
    connection = _connection(galil)
    connection.replies["MG _BGA"] = ["1.0000", "", "0.0000"]
    connection.replies["MG _BGB"] = ["0.0000"] * 3
    connection.replies["MG _BGC"] = [":", "0.0000", "0.0000"]

    galil.wait_till_motion_complete(timeout_s=5)

    assert connection.commands.count("MG _BGA") == 3
    # the I/O timeout is restored after polling
    assert connection.timeout == 5000


def test_close(galil: "GalilDMC4133Controller") -> None:
    connection = _connection(galil)
    driver_class = type(galil)

    galil.close()

    assert connection.closed
    # the closed controller is no longer registered, so the name can be reused
    assert not Instrument.exist("galil")
    reopened = driver_class("galil", address="192.168.1.30")
    reopened.close()

    # closing again does nothing
    galil.close()
