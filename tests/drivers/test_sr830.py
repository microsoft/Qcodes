"""Tests for the SR830 driver.

Most tests use the pyvisa-sim simulation in ``SR830.yaml``. The buffer
parameters guard against being attached to anything other than an SR830 and
that guard is exercised with a dummy instrument.
"""

import re
from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.mock_instruments import DummyInstrument
from qcodes.instrument_drivers.stanford_research import SR830
from qcodes.instrument_drivers.stanford_research.SR830 import (
    ChannelBuffer,
    ChannelTrace,
)
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
