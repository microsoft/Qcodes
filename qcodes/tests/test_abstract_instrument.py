from typing import Any

import pytest

from qcodes import Instrument, InstrumentChannel
from qcodes.instrument.base import InstrumentBase


class ExampleBaseVoltageSource(Instrument):
    """
    All abstract parameters *must* be implemented
    before this class can be initialized. This
    allows us to enforce an interface.
    """

    def __init__(self, name: str):
        super().__init__(name)

        self.add_parameter(
            "voltage", unit="V", abstract=True, get_cmd=None, set_cmd=None
        )

        self.add_parameter("current", unit="A", get_cmd=None, set_cmd=None)


class VoltageSource(ExampleBaseVoltageSource):
    """
    Make a specific implementation of the interface
    """

    def __init__(self, name: str):
        super().__init__(name)

        self.add_parameter("voltage", unit="V", get_cmd=None, set_cmd=None)


class VoltageSourceNotImplemented(ExampleBaseVoltageSource):
    """
    We 'forget' to implement the voltage parameter
    """


class VoltageSourceBadUnit(ExampleBaseVoltageSource):
    """
    The units must match between sub and base classes
    """

    def __init__(self, name: str):
        super().__init__(name)

        self.add_parameter(
            "voltage", unit="mV", get_cmd=None, set_cmd=None  # This should be 'V'
        )


class VoltageSourceInitException(Instrument):
    """
    We conditionally raise an assertion error in the init.
    The instrument should not be registered when one
    is raised
    """

    def __init__(self, name: str, do_raise=True):
        super().__init__(name)

        if do_raise:
            assert False


class VoltageSourceSubSub(VoltageSource):
    """
    This is a sub-sub class of the example base voltage
    source. The post init function should be called
    only once
    """

    call_count = 0

    def __init__(self, name: str):
        super().__init__(name)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__()  # type: ignore[misc]
        self.call_count += 1


class VoltageChannelBase(InstrumentChannel):
    """
    Create a channel with an abstract parameter
    """

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any):
        super().__init__(parent, name, **kwargs)

        self.add_parameter("voltage", unit="V", abstract=True)


class VoltageChannel(VoltageChannelBase):
    """
    Create an implementation of the abstract channel
    """

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any):
        super().__init__(parent, name, **kwargs)

        self.add_parameter("voltage", unit="V", get_cmd=None, set_cmd=None)


@pytest.fixture(name="driver", scope="module")
def _driver():
    drvr = VoltageSource("driver")
    yield drvr
    drvr.close()


def test_sanity(driver):
    """
    If all abstract parameters are implemented, we should be able
    to instantiate the instrument
    """
    driver.voltage(0.1)
    assert driver.voltage() == 0.1


@pytest.mark.skip("tests unimplemented feature")
def test_not_implemented_error():
    """
    If not all abstract parameters are implemented, we should see
    an exception
    """
    with pytest.raises(
        NotImplementedError, match="has un-implemented Abstract Parameter"
    ):
        VoltageSourceNotImplemented("driver2")
    assert not VoltageSourceNotImplemented.instances()


def test_get_set_raises():
    """
    If not all abstract parameters are implemented, we should see
    an exception
    """
    vs = VoltageSourceNotImplemented("driver2")

    with pytest.raises(
        NotImplementedError, match="Trying to get an abstract parameter"
    ):
        vs.voltage.get()

    with pytest.raises(
        NotImplementedError, match="Trying to set an abstract parameter"
    ):
        vs.voltage.set(0)


def test_unit_value_error():
    """
    Units should match between subclasses and base classes
    """
    with pytest.raises(ValueError, match="This is inconsistent with the unit defined"):
        VoltageSourceBadUnit("driver3")


@pytest.mark.xfail()
def test_unit_value_error_does_not_register_instrument():
    """
    Units should match between subclasses and base classes
    """
    with pytest.raises(ValueError, match="This is inconsistent with the unit defined"):
        VoltageSourceBadUnit("driver3a")
    assert not VoltageSourceBadUnit.instances()


@pytest.mark.skip("tests unimplemented feature")
def test_exception_in_init():
    """
    In previous versions of QCoDeS, if an error occurred in
    the instrument init method, we could not attempt to retry
    the instantiation with the same instrument name. This is
    because the driver instance was recorded at the end of
    the init method in the base class. In current versions
    the instance is recorded in the __post_init__ method,
    which should eliminate this problem
    """
    name = "driver4"
    try:
        VoltageSourceInitException(name)
    except AssertionError:
        pass
    assert not VoltageSourceInitException.instances()
    instance = VoltageSourceInitException(name, do_raise=False)
    assert name in [ins.name for ins in VoltageSourceInitException.instances()]
    instance.close()


@pytest.mark.skip("tests unimplemented feature")
def test_subsub():
    """
    Verify that the post init method is only called once, even
    for sub-sub classes. This should work for arbitrary levels
    of subclassing.
    """
    instance = VoltageSourceSubSub("driver5")
    assert instance.call_count == 1


@pytest.mark.skip("tests unimplemented feature")
def test_channel(driver):
    """
    This should work without exceptions
    """
    VoltageChannel(driver, "driver6")

    with pytest.raises(
        NotImplementedError, match="has un-implemented Abstract Parameter"
    ):
        VoltageChannelBase(driver, "driver6")
