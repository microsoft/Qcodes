from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.basel import BaselSP983a, BaselSP983c

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(name="sp983c")
def _make_sp983c() -> "Generator[BaselSP983c, None, None]":
    driver = BaselSP983c("sp983c")
    try:
        yield driver
    finally:
        driver.close()


@pytest.fixture(name="sp983a")
def _make_sp983a() -> "Generator[BaselSP983a, None, None]":
    driver = BaselSP983a(
        "sp983a", address="GPIB::1::INSTR", pyvisa_sim_file="Basel_SP983a.yaml"
    )
    try:
        yield driver
    finally:
        driver.close()


def test_sp983c_gain(sp983c: BaselSP983c) -> None:
    assert sp983c.gain() == 1e8

    sp983c.gain(1e6)
    assert sp983c.gain() == 1e6

    with pytest.raises(ValueError):
        sp983c.gain(2e6)


def test_sp983c_fcut(sp983c: BaselSP983c) -> None:
    assert sp983c.fcut() == 1e3

    sp983c.fcut(30)
    assert sp983c.fcut() == 30

    with pytest.raises(ValueError):
        sp983c.fcut(50)


@pytest.mark.parametrize("gain", [1e5, 1e6, 1e7, 1e8, 1e9])
def test_sp983a_gain(sp983a: BaselSP983a, gain: float) -> None:
    sp983a.gain(gain)
    value = sp983a.gain()
    assert isinstance(value, float)
    assert value == gain


@pytest.mark.parametrize("fcut", [30, 100, 300, 1e3, 3e3, 10e3, 30e3, 100e3, 1e6])
def test_sp983a_fcut(sp983a: BaselSP983a, fcut: float) -> None:
    sp983a.fcut(fcut)
    assert sp983a.fcut() == fcut


def test_sp983a_fcut_rejects_unknown_value(sp983a: BaselSP983a) -> None:
    with pytest.raises(ValueError):
        sp983a.fcut(50)


def test_sp983a_overload_status(sp983a: BaselSP983a) -> None:
    assert sp983a.overload_status() == "Overload: OFF"
