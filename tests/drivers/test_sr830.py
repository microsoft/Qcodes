"""Tests for the SR830 driver.

Most tests use the pyvisa-sim simulation in ``SR830.yaml``. The buffer
parameters guard against being attached to anything other than an SR830 and
that guard is exercised with a dummy instrument.
"""

import re
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest

from qcodes.instrument_drivers.mock_instruments import DummyInstrument
from qcodes.instrument_drivers.stanford_research import SR830
from qcodes.instrument_drivers.stanford_research.SR830 import (
    ChannelBuffer,
    ChannelTrace,
)
from qcodes.parameters import Parameter
from qcodes.validators import Arrays

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(name="sr830")
def _make_sr830() -> "Generator[SR830, None, None]":
    driver = SR830(
        "sr830",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="SR830.yaml",
        terminator="\n",
    )
    try:
        yield driver
    finally:
        driver.close()


def test_channel_buffers_are_created(sr830: SR830) -> None:
    # Creating the driver must not warn that ChannelBuffer drops the kwargs
    # that add_parameter passes on (warnings are errors in the test suite).
    for channel in (1, 2):
        buffer = sr830.parameters[f"ch{channel}_databuffer"]
        assert isinstance(buffer, ChannelBuffer)
        assert buffer.instrument is sr830
        assert buffer.channel == channel


@pytest.mark.parametrize("value", [10e-6, 100e-3, 1, 1.0, 30e3])
def test_time_constant(sr830: SR830, value: float) -> None:
    sr830.time_constant(value)
    assert sr830.time_constant() == value


def test_complex_voltage(sr830: SR830) -> None:
    value = sr830.complex_voltage()
    assert isinstance(value, complex)
    assert value == complex(1.5e-3, -2.5e-4)


@pytest.mark.parametrize("value", [62.5e-3, 0.5, 1, 512, "Trigger"])
def test_buffer_sample_rate(sr830: SR830, value: float | Literal["Trigger"]) -> None:
    sr830.buffer_SR(value)
    assert sr830.buffer_SR() == value


@pytest.mark.parametrize(("sample_rate", "dt"), [(0.25, 4.0), (1, 1.0), (512, 1 / 512)])
def test_sweep_setpoints_are_sample_times(
    sr830: SR830, sample_rate: float, dt: float
) -> None:
    sr830.buffer_SR(sample_rate)
    # bin k is sampled at k / sample_rate; the sim reports 5 stored points
    np.testing.assert_allclose(sr830.sweep_setpoints(), np.arange(5) * dt)


def test_buffer_setpoints_are_sample_times(sr830: SR830) -> None:
    sr830.buffer_SR(4)
    buffer = sr830.ch1_databuffer
    assert isinstance(buffer, ChannelBuffer)
    buffer.prepare_buffer_readout()
    assert buffer.shape == (5,)
    assert buffer.setpoints is not None
    np.testing.assert_allclose(np.asarray(buffer.setpoints[0]), np.arange(5) * 0.25)


def test_buffer_setpoints_with_trigger(sr830: SR830) -> None:
    sr830.buffer_SR("Trigger")
    buffer = sr830.ch1_databuffer
    assert isinstance(buffer, ChannelBuffer)
    buffer.prepare_buffer_readout()
    assert buffer.setpoint_names == ("trig_events",)
    assert buffer.setpoints is not None
    np.testing.assert_array_equal(np.asarray(buffer.setpoints[0]), np.arange(5))


def test_sweep_setpoints_unit_follows_sample_rate(sr830: SR830) -> None:
    gate = Parameter("gate", label="Gate voltage", unit="V", set_cmd=None)
    setpoints = sr830.sweep_setpoints

    # the sim starts at a constant sample rate (1 Hz), so setpoints are times
    sr830.set_sweep_parameters(gate, 0, 1, n_points=5)
    assert (setpoints.unit, setpoints.label) == ("s", "Time")

    sr830.buffer_SR("Trigger")
    assert (setpoints.unit, setpoints.label) == ("V", "Gate voltage")
    np.testing.assert_allclose(setpoints(), np.linspace(0, 1, 5))

    sr830.buffer_SR(1)
    assert (setpoints.unit, setpoints.label) == ("s", "Time")

    # switching back to Trigger restores the sweep unit and label
    sr830.buffer_SR("Trigger")
    assert (setpoints.unit, setpoints.label) == ("V", "Gate voltage")


def test_set_sweep_parameters_with_trigger(sr830: SR830) -> None:
    gate = Parameter("gate", label="Gate voltage", unit="V", set_cmd=None)
    setpoints = sr830.sweep_setpoints

    sr830.buffer_SR("Trigger")
    assert (setpoints.unit, setpoints.label) == ("", "sweep_setpoints")

    sr830.set_sweep_parameters(gate, 0, 1, n_points=5, label="Gate")
    assert (setpoints.unit, setpoints.label) == ("V", "Gate")

    sr830.buffer_SR(1)
    assert (setpoints.unit, setpoints.label) == ("s", "Time")


@pytest.fixture(name="not_an_sr830")
def _make_not_an_sr830() -> "Generator[DummyInstrument, None, None]":
    instrument = DummyInstrument("not_an_sr830", gates=[])
    try:
        yield instrument
    finally:
        instrument.close()


def test_channel_trace_requires_sr830(not_an_sr830: DummyInstrument) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid parent instrument. ChannelBuffer can only live on an SR830."
        ),
    ):
        ChannelTrace(
            "ch1_datatrace",
            channel=1,
            instrument=not_an_sr830,
            vals=Arrays(shape=(1,)),
        )


def test_channel_buffer_requires_sr830(not_an_sr830: DummyInstrument) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid parent instrument. ChannelBuffer can only live on an SR830."
        ),
    ):
        ChannelBuffer(
            "ch1_databuffer",
            instrument=not_an_sr830,  # type: ignore[arg-type]
            channel=1,
        )


@pytest.mark.parametrize("channel", [0, 3])
def test_buffer_parameters_reject_invalid_channel(
    not_an_sr830: DummyInstrument, channel: int
) -> None:
    match = re.escape("Invalid channel specifier. SR830 only has channels 1 and 2.")
    with pytest.raises(ValueError, match=match):
        ChannelTrace(
            f"ch{channel}_datatrace",
            channel=channel,
            instrument=not_an_sr830,
            vals=Arrays(shape=(1,)),
        )
    with pytest.raises(ValueError, match=match):
        ChannelBuffer(
            f"ch{channel}_databuffer",
            instrument=not_an_sr830,  # type: ignore[arg-type]
            channel=channel,
        )
