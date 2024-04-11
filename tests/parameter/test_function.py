from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.mock_instruments import DummyChannelInstrument

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(name="channel_instr")
def _make_channel_instr() -> "Iterator[DummyChannelInstrument]":
    instr = DummyChannelInstrument("dummy")
    try:
        yield instr
    finally:
        instr.close()


def test_function_name(channel_instr: DummyChannelInstrument) -> None:
    assert channel_instr.A.log_my_name.short_name == "log_my_name"
    assert channel_instr.A.log_my_name.full_name == "dummy_ChanA_log_my_name"
