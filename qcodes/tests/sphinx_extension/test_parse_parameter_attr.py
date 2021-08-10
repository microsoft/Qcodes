import pytest
from sphinx.util.inspect import safe_getattr

from qcodes.instrument.base import InstrumentBase
from qcodes.instrument.visa import VisaInstrument
from qcodes.instrument_drivers.ZI.ZIUHFLI import ZIUHFLI
from qcodes.sphinx_extensions.parse_parameter_attr import (
    ParameterProxy,
    qcodes_parameter_attr_getter,
)


class DummyTestClass(InstrumentBase):

    myattr: str = "ClassAttribute"
    """
    A class attribute
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.other_attr = "InstanceAttribute"
        """
        An instance attribute
        """


def test_extract_class_attr():
    a = safe_getattr(DummyTestClass, "myattr")
    assert a == "ClassAttribute"

    b = qcodes_parameter_attr_getter(DummyTestClass, "myattr")
    assert b == "ClassAttribute"


def test_extract_instance_attr():
    with pytest.raises(AttributeError):
        safe_getattr(DummyTestClass, "other_attr")

    b = qcodes_parameter_attr_getter(DummyTestClass, "other_attr")
    assert isinstance(b, ParameterProxy)
    assert repr(b) == '"InstanceAttribute"'


def test_instrument_base_get_attr():
    parameters = qcodes_parameter_attr_getter(InstrumentBase, "parameters")
    assert isinstance(parameters, ParameterProxy)
    assert repr(parameters) == "{}"


def test_visa_instr_get_attr():
    parameters = qcodes_parameter_attr_getter(VisaInstrument, "parameters")
    assert isinstance(parameters, ParameterProxy)
    assert repr(parameters) == "{}"


def test_zi():
    scope = qcodes_parameter_attr_getter(ZIUHFLI, "scope")
    # not currently correctly resolved since init is decorated
