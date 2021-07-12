import pytest

from qcodes.tests.instrument_mocks import DummyAttrInstrument


@pytest.fixture(name="dummy_attr_instr")
def _make_dummy_attr_instr():
    dummy_attr_instr = DummyAttrInstrument("dummy_attr_instr")
    yield dummy_attr_instr
    dummy_attr_instr.close()


def test_parameter_registration_on_instr(dummy_attr_instr):
    """Test that an instrument that have parameters defined as attrs"""
    assert dummy_attr_instr.ch1.instrument is dummy_attr_instr
    assert dummy_attr_instr.parameters["ch1"] is dummy_attr_instr.ch1

    assert (
        dummy_attr_instr.snapshot()["parameters"]["ch1"]["full_name"]
        == "dummy_attr_instr_ch1"
    )
