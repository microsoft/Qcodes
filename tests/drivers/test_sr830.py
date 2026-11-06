"""Tests for the parameter classes of the SR830 driver.

The driver itself needs a connected instrument, but the buffer parameters
guard against being attached to anything other than an SR830 and that guard
can be exercised with any instrument.
"""

import re
from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.mock_instruments import DummyInstrument
from qcodes.instrument_drivers.stanford_research.SR830 import (
    ChannelBuffer,
    ChannelTrace,
)
from qcodes.validators import Arrays

if TYPE_CHECKING:
    from collections.abc import Generator


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
