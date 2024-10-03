import pytest
from sphinx.util.inspect import safe_getattr

from qcodes.instrument import InstrumentBase
from qcodes.parameters import Parameter
from qcodes.sphinx_extensions.parse_parameter_attr import (
    ParameterProxy,
    qcodes_parameter_attr_getter,
)
from qcodes.utils import deprecate


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

        self.my_param = Parameter(
            name="my_param", instrument=self, get_cmd=None, set_cmd=None
        )
        """A QCoDeS Parameter"""


class DummyNoInitClass(DummyTestClass):
    myattr: str = "ClassAttribute"
    """
    A class attribute
    """

    def somefunction(self) -> str:
        return self.myattr


class DummyDecoratedInitTestClass(InstrumentBase):
    myattr: str = "ClassAttribute"
    """
    A class attribute
    """

    @deprecate("Deprecate to test that decorated init is handled correctly")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.other_attr = "InstanceAttribute"
        """
        An instance attribute
        """


@deprecate("Deprecate to test that decorated class is handled correctly")
class DummyDecoratedClassTestClass(InstrumentBase):
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


def test_extract_class_attr() -> None:
    a = safe_getattr(DummyTestClass, "myattr")
    assert a == "ClassAttribute"

    b = qcodes_parameter_attr_getter(DummyTestClass, "myattr")
    assert b == "ClassAttribute"


def test_extract_instance_attr() -> None:
    with pytest.raises(AttributeError):
        safe_getattr(DummyTestClass, "other_attr")

    b = qcodes_parameter_attr_getter(DummyTestClass, "other_attr")
    assert isinstance(b, ParameterProxy)
    assert repr(b) == '"InstanceAttribute"'


def test_parameter_get_attr():
    parameters = qcodes_parameter_attr_getter(DummyTestClass, "my_param")
    assert isinstance(parameters, ParameterProxy)
    assert (
        repr(parameters)
        == 'Parameter( name="my_param", instrument=self, get_cmd=None, set_cmd=None )'
    )


def test_decorated_init_func() -> None:
    attr = qcodes_parameter_attr_getter(DummyDecoratedInitTestClass, "other_attr")
    assert isinstance(attr, ParameterProxy)
    assert repr(attr) == '"InstanceAttribute"'


def test_decorated_class() -> None:
    attr = qcodes_parameter_attr_getter(DummyDecoratedClassTestClass, "other_attr")
    assert isinstance(attr, ParameterProxy)
    assert repr(attr) == '"InstanceAttribute"'


def test_no_init() -> None:
    """Test that attribute can be found from a class without an init function."""
    attr = qcodes_parameter_attr_getter(DummyNoInitClass, "other_attr")
    assert isinstance(attr, ParameterProxy)
    assert repr(attr) == '"InstanceAttribute"'
