import pytest

from qcodes import Instrument
from qcodes.instrument.abstract import (
    AbstractParameter, abstract_instrument, AbstractParameterException
)


@abstract_instrument
class BaseVoltageSource(Instrument):
    """
    
    We demand that all voltage sources implement a standard interface
    defined by this base class
    """

    def __init__(self, name: str):
        """
        A test voltage source
        """
        super().__init__(name)

        self.add_parameter(
            "voltage",
            unit="V",
            parameter_class=AbstractParameter
        )

        self.add_parameter(
            "current",
            unit="A",
            get_cmd=None,
            set_cmd=None
        )


class MyVoltageSource(BaseVoltageSource):
    """
    A voltage source driver for a particular instrument
    make and model.
    """
    def __init__(self, name: str):
        super().__init__(name)

        self.add_parameter(
            "voltage",
            unit="V",
            set_cmd=None,
            get_cmd=None
        )


class WrongSource(BaseVoltageSource):
    """
    Let's 'forget' to implement the voltage parameter
    """


class WrongSource2(BaseVoltageSource):
    """
    We implement the voltage parameter with the wrong unit
    """
    def __init__(self, name: str):
        super().__init__(name)

        self.add_parameter(
            "voltage",
            unit="mV"
        )


def test_sanity():
    source = MyVoltageSource("name1")
    source.voltage(0.1)
    assert pytest.approx(source.voltage(), 0.1)

    source.current(-0.1)
    assert pytest.approx(source.current(), -0.1)

    source.close()


def test_unimplemented_parameter():

    with pytest.raises(
        AbstractParameterException,
        match=r"Class 'WrongSource' has un\-implemented Abstract Parameter\(s\):"
    ):
        WrongSource("name2")


def test_inconsistent_units():

    with pytest.raises(
        AbstractParameterException,
        match=r"The unit of the parameter"
    ):
        WrongSource2("name3")
