"""Tests for the Galil DMC-41x3 driver using a fake ``gclib`` module.

The driver requires the vendor ``gclib`` package, which cannot be installed
from PyPI, so it is replaced by a minimal fake that records the calls made.
"""

import importlib
import sys
import types
from typing import TYPE_CHECKING

import pytest

import qcodes.instrument_drivers
from qcodes.instrument import Instrument

if TYPE_CHECKING:
    from collections.abc import Generator

    from qcodes.instrument_drivers.Galil.dmc_41x3 import GalilDMC4133Controller

_DRIVER_MODULE = "qcodes.instrument_drivers.Galil.dmc_41x3"


class _FakeGclibError(Exception):
    pass


class _FakeGclibConnection:
    """
    Fake of ``gclib.py``. Queries are answered from ``replies``, a list of
    replies per command that are returned in turn.
    """

    def __init__(self) -> None:
        self.is_open = False
        self.replies: dict[str, list[str]] = {}
        self.commands: list[str] = []
        self.timeout = 5000

    def GOpen(self, address: str) -> None:
        self.is_open = True

    def GClose(self) -> None:
        self.is_open = False

    def GInfo(self) -> str:
        return "192.168.1.2, DMC4133 Rev 1.0b, 12345"

    def GCommand(self, command: str) -> str:
        command = command.removesuffix("\r")
        self.commands.append(command)
        replies = self.replies.get(command)
        return replies.pop(0) if replies else ""

    def GTimeout(self, timeout: int) -> None:
        self.timeout = timeout


@pytest.fixture(name="driver_module")
def _driver_module_with_fake_gclib(
    monkeypatch: pytest.MonkeyPatch,
) -> "Generator[types.ModuleType, None, None]":
    fake_gclib = types.ModuleType("gclib")
    fake_gclib.py = _FakeGclibConnection  # type: ignore[attr-defined]
    fake_gclib.GclibError = _FakeGclibError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gclib", fake_gclib)
    for name in ("qcodes.instrument_drivers.Galil", _DRIVER_MODULE):
        monkeypatch.delitem(sys.modules, name, raising=False)
    # importing the package sets it as an attribute of its parent package
    monkeypatch.delattr(qcodes.instrument_drivers, "Galil", raising=False)
    try:
        yield importlib.import_module(_DRIVER_MODULE)
    finally:
        # Do not leave the driver bound to the fake module behind.
        for name in ("qcodes.instrument_drivers.Galil", _DRIVER_MODULE):
            sys.modules.pop(name, None)
        if hasattr(qcodes.instrument_drivers, "Galil"):
            delattr(qcodes.instrument_drivers, "Galil")


@pytest.fixture(name="galil")
def _make_galil(
    driver_module: types.ModuleType,
) -> "Generator[GalilDMC4133Controller, None, None]":
    driver = driver_module.GalilDMC4133Controller("galil", address="192.168.1.2")
    try:
        yield driver
    finally:
        driver.close()


def _connection(galil: "GalilDMC4133Controller") -> _FakeGclibConnection:
    assert isinstance(galil.g, _FakeGclibConnection)
    return galil.g


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


def test_idn(galil: "GalilDMC4133Controller") -> None:
    assert galil.IDN() == {
        "vendor": "Galil Motion Control, Inc.",
        "model": "DMC4133",
        "serial": "12345",
        "firmware": "1.0b",
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


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("1.0000", "off"),
        ("0.0000", "on"),
        # with LZ0 the controller does not strip leading zeros
        ("0000000001.0000", "off"),
    ],
)
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
