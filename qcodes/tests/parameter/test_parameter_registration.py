import pytest

from qcodes.instrument.parameter import Parameter
from qcodes.tests.instrument_mocks import DummyAttrInstrument
from qcodes.utils.deprecate import QCoDeSDeprecationWarning


class BrokenParameter(Parameter):
    """
    A parameter that incorrectly sets instrument to the super-class parameter,
    instead, it should forward it via ``instrument`` argument
    """

    def __init__(self, name, instrument, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._instrument = instrument


class BrokenParameter2(Parameter):
    """A parameter that does not pass kwargs to the _BaseParameter class"""

    def __init__(self, name, instrument, set_cmd, get_cmd):
        super().__init__(name=name, instrument=instrument)


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


def test_parameter_registration_with_non_instr_passing_parameter(dummy_attr_instr):
    with pytest.warns(
        QCoDeSDeprecationWarning,
        match="Parameter brokenparameter did not correctly register itself on instrument dummy_attr_instr",
    ):
        dummy_attr_instr.add_parameter(
            name="brokenparameter",
            parameter_class=BrokenParameter,
            set_cmd=None,
            get_cmd=None,
        )
    # test that even if the parameter does not pass instrument to the baseclass
    # it will still be registered on the instr
    assert "brokenparameter" in dummy_attr_instr.parameters.keys()


def test_parameter_registration_with_non_kwargs_passing_parameter(dummy_attr_instr):
    with pytest.warns(
        QCoDeSDeprecationWarning,
        match="does not correctly pass kwargs to its baseclass",
    ):
        dummy_attr_instr.add_parameter(
            name="brokenparameter2",
            parameter_class=BrokenParameter2,
            set_cmd=None,
            get_cmd=None,
        )
    # test that even if the parameter does not pass kwargs (bind_to_instrument specifically)
    # to the baseclass it will still be registered on the instr
    assert "brokenparameter2" in dummy_attr_instr.parameters.keys()
