from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.mock_instruments import DummyInstrument
from qcodes.parameters import InstrumentRefParameter

if TYPE_CHECKING:
    from collections.abc import Generator


class DummyHolder(DummyInstrument):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.test = self.add_parameter(
            "test",
            parameter_class=InstrumentRefParameter,
            initial_value=None,
        )


@pytest.fixture(name="instrument_a")
def _make_instrument_a() -> "Generator[DummyHolder, None, None]":

    a = DummyHolder("dummy_holder")
    yield a
    a.close()


@pytest.fixture(name="instrument_d")
def _make_instrument_d() -> "Generator[DummyInstrument, None, None]":
    d = DummyInstrument("dummy")
    yield d
    d.close()


def test_get_instr(instrument_a: DummyHolder, instrument_d: DummyInstrument) -> None:

    instrument_a.test.set(instrument_d.name)

    assert instrument_a.test.get() == instrument_d.name
    assert instrument_a.test.get_instr() == instrument_d
