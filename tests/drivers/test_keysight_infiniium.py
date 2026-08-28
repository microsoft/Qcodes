"""Tests for the Keysight Infiniium driver using a pyvisa-sim backend."""

from typing import TYPE_CHECKING, cast

import pytest

from qcodes.instrument import Instrument
from qcodes.instrument_drivers.Keysight.Infiniium import (
    DSOTraceParam,
    KeysightInfiniium,
    KeysightInfiniiumChannel,
)
from qcodes.validators import Arrays

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(name="driver")
def _make_driver() -> "Iterator[KeysightInfiniium]":
    driver = KeysightInfiniium(
        "infiniium_sim",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="Keysight_Infiniium.yaml",
    )
    yield driver
    driver.close()


@pytest.fixture(name="orphan_trace")
def _make_orphan_trace() -> "Iterator[DSOTraceParam]":
    """A trace parameter attached to an instrument that is not an Infiniium."""
    instrument = Instrument("not_an_infiniium")
    trace = DSOTraceParam(
        name="trace",
        instrument=cast("KeysightInfiniiumChannel", instrument),
        channel="CHAN1",
        vals=Arrays(shape=(10,)),
    )
    yield trace
    instrument.close()


def test_idn(driver: KeysightInfiniium) -> None:
    assert driver.IDN() == {
        "vendor": "Keysight Technologies",
        "model": "MSOS254A",
        "serial": "MY00000000",
        "firmware": "06.60.00902",
    }


def test_capabilities(driver: KeysightInfiniium) -> None:
    assert driver.min_bw == 1.0e7
    assert driver.max_bw == 5.0e10
    assert driver.min_pts == 16
    assert driver.max_pts == 2_000_000
    assert driver.min_srat == 1.0e3
    assert driver.max_srat == 2.0e11


def test_channel_trace_root_instrument(driver: KeysightInfiniium) -> None:
    assert driver.ch1.trace.root_instrument is driver


def test_function_trace_root_instrument(driver: KeysightInfiniium) -> None:
    assert driver.func1.trace.root_instrument is driver


def test_root_instrument_raises_for_foreign_parent(
    orphan_trace: DSOTraceParam,
) -> None:
    with pytest.raises(
        RuntimeError, match="not bound to a KeysightInfiniium instrument"
    ):
        _ = orphan_trace.root_instrument


def test_setpoints_raises_for_foreign_parent(orphan_trace: DSOTraceParam) -> None:
    with pytest.raises(RuntimeError, match="Invalid type for parent instrument"):
        _ = orphan_trace.setpoints


def test_channel_setpoints_updates_time_axis(driver: KeysightInfiniium) -> None:
    assert driver.cache_setpoints() is False
    setpoints = driver.ch1.trace.setpoints
    assert setpoints == (driver.ch1.time_axis,)
    # setpoints have been refreshed from the preamble
    assert driver.ch1.time_axis.points == 1000
    assert driver.ch1.time_axis.xorigin == -5.0e-8
    assert driver.ch1.time_axis.xincrement == 1.0e-10
    assert driver.ch1.trace.unit == "V"


def test_channel_setpoints_are_cached(driver: KeysightInfiniium) -> None:
    driver.cache_setpoints(True)
    setpoints = driver.ch1.trace.setpoints
    assert setpoints == (driver.ch1.time_axis,)
    # the preamble was never queried so the axis keeps its initial values
    assert driver.ch1.time_axis.points == 1


def test_function_fft_setpoints(driver: KeysightInfiniium) -> None:
    assert driver.func1.function() == "FFTMAGNITUDE"
    setpoints = driver.func1.trace.setpoints
    assert setpoints == (driver.func1.frequency_axis,)
    assert driver.func1.frequency_axis.points == 1000
    assert driver.func1.frequency_axis.xorigin == -5.0e-8
    assert driver.func1.frequency_axis.xincrement == 1.0e-10


def test_function_non_fft_setpoints(driver: KeysightInfiniium) -> None:
    assert driver.func2.function() == "ADD"
    setpoints = driver.func2.trace.setpoints
    assert setpoints == (driver.func2.time_axis,)
    assert driver.func2.time_axis.points == 1000


def test_update_fft_setpoints_raises_for_channel(driver: KeysightInfiniium) -> None:
    with pytest.raises(
        RuntimeError, match="FFT setpoints can only be updated for a function parameter"
    ):
        driver.ch1.trace.update_fft_setpoints()
