import sys
from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.mock_instruments import DummyChannelInstrument

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def dummy_channel_instrument() -> "Generator[DummyChannelInstrument, None, None]":
    instrument = DummyChannelInstrument(name="testdummy")
    try:
        yield instrument
    finally:
        instrument.close()


@pytest.fixture
def assert_raises_match() -> str:
    if sys.version_info >= (3, 11):
        return "Value should either be valid"
    else:
        return ""


def test_set_multi_channel_instrument_parameter(
    dummy_channel_instrument: DummyChannelInstrument, assert_raises_match: str
):
    """Tests :class:`MultiChannelInstrumentParameter` set method."""
    for name, param in dummy_channel_instrument.channels[0].parameters.items():
        if not param.settable:
            continue

        channel_parameter = getattr(dummy_channel_instrument.channels, name)

        getval = channel_parameter.get()

        channel_parameter.set(getval[0])

        # Assert channel parameter setters accept what the getter returns (PR #6073)
        channel_parameter.set(getval)

        with pytest.raises(TypeError, match=assert_raises_match):
            channel_parameter.set(getval[:-1])

        with pytest.raises(TypeError, match=assert_raises_match):
            channel_parameter.set(getval + (getval[-1],))

        with pytest.raises(TypeError):
            channel_parameter.set(object())
