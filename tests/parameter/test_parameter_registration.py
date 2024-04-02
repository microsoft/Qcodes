from typing import TYPE_CHECKING, Any

import pytest

from qcodes.instrument_drivers.mock_instruments import DummyAttrInstrument
from qcodes.parameters import Parameter
from qcodes.utils import QCoDeSDeprecationWarning

if TYPE_CHECKING:
    from collections.abc import Generator

    from qcodes.instrument import InstrumentBase


class BrokenParameter(Parameter):
    """
    A parameter that incorrectly sets instrument to the super-class parameter,
    instead, it should forward it via ``instrument`` argument
    """

    def __init__(
        self, name: str, instrument: "InstrumentBase", *args: Any, **kwargs: Any
    ):
        super().__init__(name, *args, **kwargs)
        self._instrument = instrument


class BrokenParameter2(Parameter):
    """A parameter that does not pass kwargs to the ParameterBase class"""

    def __init__(
        self, name: str, instrument: "InstrumentBase", set_cmd: Any, get_cmd: Any
    ):
        super().__init__(name=name, instrument=instrument)


@pytest.fixture(name="dummy_attr_instr")
def _make_dummy_attr_instr() -> "Generator[DummyAttrInstrument, None, None]":
    dummy_attr_instr = DummyAttrInstrument("dummy_attr_instr")
    yield dummy_attr_instr
    dummy_attr_instr.close()


def test_parameter_registration_on_instr(dummy_attr_instr: DummyAttrInstrument) -> None:
    """Test that an instrument that have parameters defined as attrs"""
    assert dummy_attr_instr.ch1.instrument is dummy_attr_instr
    assert dummy_attr_instr.parameters["ch1"] is dummy_attr_instr.ch1

    assert (
        dummy_attr_instr.snapshot()["parameters"]["ch1"]["full_name"]
        == "dummy_attr_instr_ch1"
    )


def test_parameter_registration_with_non_instr_passing_parameter(
    dummy_attr_instr: DummyAttrInstrument,
) -> None:
    with pytest.warns(
        QCoDeSDeprecationWarning,
        match="Parameter brokenparameter did not correctly "
        "register itself on instrument dummy_attr_instr",
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


def test_parameter_registration_with_non_kwargs_passing_parameter(
    dummy_attr_instr: DummyAttrInstrument,
) -> None:
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
    # test that even if the parameter does not pass kwargs
    # (bind_to_instrument specifically)
    # to the baseclass it will still be registered on the instr
    assert "brokenparameter2" in dummy_attr_instr.parameters.keys()
