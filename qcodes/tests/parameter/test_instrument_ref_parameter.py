import pytest

from qcodes.instrument.parameter import InstrumentRefParameter
from qcodes.tests.instrument_mocks import DummyInstrument


@pytest.fixture()
def instrument_a():
    a = DummyInstrument('dummy_holder')
    try:
        yield a
    finally:
        a.close()


@pytest.fixture()
def instrument_d():
    d = DummyInstrument('dummy')
    try:
        yield d
    finally:
        d.close()


def test_get_instr(instrument_a, instrument_d):
    instrument_a.add_parameter('test', parameter_class=InstrumentRefParameter)

    instrument_a.test.set(instrument_d.name)

    assert instrument_a.test.get() == instrument_d.name
    assert instrument_a.test.get_instr() == instrument_d
