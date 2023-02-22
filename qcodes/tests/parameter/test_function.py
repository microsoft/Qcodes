from typing import Iterator

import pytest

from qcodes.tests.instrument_mocks import DummyChannelInstrument


@pytest.fixture(name="channel_instr")
def _make_channel_instr() -> Iterator[DummyChannelInstrument]:
    instr = DummyChannelInstrument("dummy")
    try:
        yield instr
    finally:
        instr.close()


def test_function_name(channel_instr: DummyChannelInstrument) -> None:
    assert channel_instr.A.log_my_name.short_name == "log_my_name"
    assert channel_instr.A.log_my_name.full_name == "dummy_ChanA_log_my_name"
