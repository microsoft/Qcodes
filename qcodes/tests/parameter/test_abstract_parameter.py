import pytest

from qcodes import Instrument
from qcodes.instrument.parameter import AbstractParameter


class BaseVoltageSource(Instrument):
    def __init__(self, name):
        super().__init__(name)

        self.add_parameter(
            "voltage",
            parameter_class=AbstractParameter,
            unit="V"
        )


class MyVoltageSource(BaseVoltageSource):

    def __init__(self, name):
        super().__init__(name)

        self.add_parameter(
            "voltage",
            set_cmd=None,
            get_cmd=None,
            unit="V"
        )


class MyVoltageSource2(BaseVoltageSource):

    def __init__(self, name):
        super().__init__(name)

        self.add_parameter(
            "voltage",
            set_cmd=None,
            get_cmd=None,
            unit="mV"
        )


def test_basic():
    """
    We should be able to use the instrument like any other
    """
    source = MyVoltageSource("name")
    source.voltage.set(2.3)
    assert source.voltage.get() == 2.3


def test_unit_mismatch_exception():

    with pytest.raises(
            ValueError,
            match="inconsistent with the unit"
    ):
        MyVoltageSource2("name2")
