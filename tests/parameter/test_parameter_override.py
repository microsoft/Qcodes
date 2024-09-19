import logging
from typing import TYPE_CHECKING

import pytest

from qcodes.instrument import Instrument

if TYPE_CHECKING:
    from qcodes.parameters import Parameter


class DummyOverrideInstr(Instrument):
    def __init__(self, name, **kwargs):
        """
        This instrument errors because it tries to override an attribute with a parameter.
        Removing the parameter using `self.parameters.pop("voltage")` is not safe but
        would work if the parameter was not assigned to an attribute.
        """
        super().__init__(name, **kwargs)
        self.voltage = self.add_parameter("voltage", set_cmd=None, get_cmd=None)

        self.parameters.pop("voltage")

        self.voltage = self.add_parameter("voltage", set_cmd=None, get_cmd=None)


class DummyParameterIsAttrInstr(Instrument):
    def __init__(self, name, **kwargs):
        """
        This instrument errors because it tries to override an attribute with a parameter.
        """
        super().__init__(name, **kwargs)
        self.voltage = self.add_parameter("voltage", set_cmd=None, get_cmd=None)

    def voltage(self):
        return 0


class DummyParameterIsPropertyInstr(Instrument):
    def __init__(self, name, **kwargs):
        """
        We allow overriding a property since this pattern has been seen in the wild
        to define an interface for the instrument.
        """
        super().__init__(name, **kwargs)
        self.add_parameter("voltage", set_cmd=None, get_cmd=None)  # type: ignore

    @property
    def voltage(self):
        return self.parameters["voltage"]


class DummyClassAttrInstr(Instrument):
    some_attr = 1
    current: "Parameter"
    voltage: "Parameter | None" = None
    frequency: "Parameter | None" = None

    def __init__(self, name, **kwargs):
        """
        We allow overriding a property since this pattern has been seen in the wild
        to define an interface for the instrument.
        """
        super().__init__(name, **kwargs)
        self.voltage = self.add_parameter("voltage", set_cmd=None, get_cmd=None)
        self.current = self.add_parameter("current", set_cmd=None, get_cmd=None)
        self.add_parameter("frequency", set_cmd=None, get_cmd=None)


class DummyInstr(Instrument):
    def __init__(self, name, **kwargs):
        """
        We allow overriding a property since this pattern has been seen in the wild
        to define an interface for the instrument.
        """
        super().__init__(name, **kwargs)
        self.voltage = self.add_parameter("voltage", set_cmd=None, get_cmd=None)
        self.add_parameter("current", set_cmd=None, get_cmd=None)
        self.some_attr = 1


def test_overriding_parameter_attribute_with_parameter_raises():
    with pytest.raises(
        KeyError,
        match="Duplicate parameter name voltage on instrument",
    ):
        DummyOverrideInstr("my_instr")


def test_overriding_attribute_with_parameter_raises():
    with pytest.warns(
        match="Parameter voltage overrides an attribute of the same name on instrument",
    ):
        DummyParameterIsAttrInstr("my_instr")


def test_overriding_property_with_parameter_works(request):
    request.addfinalizer(DummyParameterIsPropertyInstr.close_all)
    DummyParameterIsPropertyInstr("my_instr")


def test_removing_parameter_works(request):
    request.addfinalizer(DummyInstr.close_all)
    a = DummyInstr("my_instr")

    a.remove_parameter("voltage")

    a.remove_parameter("current")

    with pytest.raises(KeyError, match="some_attr"):
        a.remove_parameter("some_attr")

    assert a.some_attr == 1


def test_removed_parameter_from_prop_instrument_works(request):
    request.addfinalizer(DummyParameterIsPropertyInstr.close_all)
    a = DummyParameterIsPropertyInstr("my_instr")
    a.remove_parameter("voltage")
    a.add_parameter("voltage", set_cmd=None, get_cmd=None)
    a.voltage.set(1)


def test_remove_parameter_from_class_attr_works(request, caplog):
    request.addfinalizer(DummyClassAttrInstr.close_all)
    a = DummyClassAttrInstr("my_instr")

    # removing a parameter that is assigned as an attribute
    # with a class level type works
    assert hasattr(a, "current")
    a.remove_parameter("current")
    assert not hasattr(a, "current")

    # when we remove a parameter with an attr that shadows a class attr
    # we get the class attr after the removal
    assert hasattr(a, "voltage")
    assert a.voltage is not None
    a.remove_parameter("voltage")
    assert hasattr(a, "voltage")
    assert a.voltage is None

    # modifying a parameter that is not assigned as an attribute
    # does not alter the class attribute
    assert hasattr(a, "frequency")
    assert a.frequency is None
    with caplog.at_level(logging.WARNING):
        a.remove_parameter("frequency")
    assert (
        "Could not remove attribute frequency from my_instr"
        in caplog.records[0].message
    )
    assert a.frequency is None

    # removing a classattr raises since it is not a parameter
    assert a.some_attr == 1
    with pytest.raises(KeyError, match="some_attr"):
        a.remove_parameter("some_attr")

    assert a.some_attr == 1
