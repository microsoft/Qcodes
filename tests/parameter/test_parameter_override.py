import pytest

from qcodes.instrument import Instrument


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
        self.voltage = self.add_parameter("voltage", set_cmd=None, get_cmd=None)  # type: ignore

    @property
    def voltage(self):
        return self.parameters["voltage"]


class DummyInstr(Instrument):
    def __init__(self, name, **kwargs):
        """
        We allow overriding a property since this pattern has been seen in the wild
        to define an interface for the instrument.
        """
        super().__init__(name, **kwargs)
        self.voltage = self.add_parameter("voltage", set_cmd=None, get_cmd=None)


def test_overriding_parameter_attribute_with_parameter_raises():
    with pytest.raises(
        KeyError,
        match="Parameter voltage overrides an attribute of the same name on instrument",
    ):
        DummyOverrideInstr("my_instr")


def test_overriding_attribute_with_parameter_raises():
    with pytest.raises(
        KeyError,
        match="Parameter voltage overrides an attribute of the same name on instrument",
    ):
        DummyParameterIsAttrInstr("my_instr")


def test_overriding_property_with_parameter_works():
    DummyParameterIsPropertyInstr("my_instr")
