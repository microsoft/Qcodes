import pytest
from sphinx.util.inspect import safe_getattr

from qcodes.instrument.base import InstrumentBase
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
    assert a == "SomeAttribute"

    b = qcodes_parameter_attr_getter(DummyTestClass, "myattr")
    assert b == "SomeAttribute"


def test_extract_instance_attr():
    with pytest.raises(AttributeError):
        safe_getattr(DummyTestClass, "other_attr")

    b = qcodes_parameter_attr_getter(DummyTestClass, "other_attr")
    assert isinstance(b, ParameterProxy)
    assert repr(b) == '"InstanceAttribute"'
