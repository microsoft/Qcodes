from typing import Any

import pytest
from pytest import FixtureRequest

from qcodes.instrument import ChannelList, Instrument, InstrumentBase, InstrumentChannel


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
            "voltage",
            unit="mV",
            get_cmd=None,
            set_cmd=None,  # This should be 'V'
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


class VoltageAbstractChannelSource(Instrument):
    """
    A channel instrument with an abstract parameter on the channel.
    This should raise.
    """

    def __init__(
        self,
        name: str,
    ):
        super().__init__(name)
        channel = VoltageChannelBase(self, "voltage")
        self.add_submodule("voltage", channel)


class VoltageChannelSource(Instrument):
    """
    A channel instrument with an implementation of the
    abstract parameter on the channel.
    This should not raise.
    """

    def __init__(
        self,
        name: str,
    ):
        super().__init__(name)
        channel = VoltageChannel(self, "voltage")
        self.add_submodule("voltage", channel)


class VoltageAbstractChannelListSource(Instrument):
    """
    A channel instrument with an abstract parameter on the channellist.
    This should raise.
    """

    def __init__(
        self,
        name: str,
    ):
        super().__init__(name)
        channel = VoltageChannelBase(self, "voltage")
        channellist = ChannelList(self, "cl", VoltageChannelBase, chan_list=[channel])
        self.add_submodule("voltage", channellist)


class VoltageChannelListSource(Instrument):
    """
    A channel instrument with an implementation of the
    abstract parameter on the channellist.
    This should not raise.
    """

    def __init__(
        self,
        name: str,
    ):
        super().__init__(name)
        channel = VoltageChannel(self, "voltage")
        channellist = ChannelList(self, "cl", VoltageChannel, chan_list=[channel])
        self.add_submodule("voltage", channellist)


@pytest.fixture(name="driver", scope="module")
def _driver():
    drvr = VoltageSource("abstract_instrument_driver")
    yield drvr
    drvr.close()


def test_sanity(driver) -> None:
    """
    If all abstract parameters are implemented, we should be able
    to instantiate the instrument
    """
    driver.voltage(0.1)
    assert driver.voltage() == 0.1


def test_not_implemented_error() -> None:
    """
    If not all abstract parameters are implemented, we should see
    an exception
    """
    with pytest.raises(
        NotImplementedError, match="has un-implemented Abstract Parameter"
    ):
        VoltageSourceNotImplemented("abstract_instrument_driver_2")
    assert not VoltageSourceNotImplemented.instances()


def test_unit_value_error() -> None:
    """
    Units should match between subclasses and base classes
    """
    with pytest.raises(ValueError, match="This is inconsistent with the unit defined"):
        VoltageSourceBadUnit("abstract_instrument_driver_3")


def test_unit_value_error_does_not_register_instrument() -> None:
    """
    Units should match between subclasses and base classes
    """
    with pytest.raises(ValueError, match="This is inconsistent with the unit defined"):
        VoltageSourceBadUnit("abstract_instrument_driver_4")
    assert not VoltageSourceBadUnit.instances()


def test_exception_in_init() -> None:
    """
    Verify that if the instrument raises in init it is not registered as an instance
    """
    name = "abstract_instrument_driver_6"
    try:
        VoltageSourceInitException(name)
    except AssertionError:
        pass
    assert not VoltageSourceInitException.instances()
    instance = VoltageSourceInitException(name, do_raise=False)
    assert name in [ins.name for ins in VoltageSourceInitException.instances()]
    instance.close()


def test_abstract_channel_raises() -> None:
    """
    Creating an instrument with a channel with abstract parameters should raise
    """
    with pytest.raises(
        NotImplementedError, match="has un-implemented Abstract Parameter"
    ):
        VoltageAbstractChannelSource("abstract_instrument_driver_7")


def test_non_abstract_channel_does_not_raises(request: FixtureRequest) -> None:
    """
    Creating an instrument with a channel that implements the interface.
    This should not raise
    """
    source = VoltageChannelSource("abstract_instrument_driver_8")
    request.addfinalizer(source.close)


def test_abstract_channellist_raises() -> None:
    """
    Creating an instrument with a channel (in a ChannelList)
    with abstract parameters should raise
    """
    with pytest.raises(
        NotImplementedError, match="has un-implemented Abstract Parameter"
    ):
        VoltageAbstractChannelListSource("abstract_instrument_driver_9")


def test_non_abstract_channellist_does_not_raises(request: FixtureRequest) -> None:
    """
    Creating an instrument with a ChannelList that contains a
    channel that implements the interface. This should not raise
    """
    source = VoltageChannelListSource("abstract_instrument_driver_10")
    request.addfinalizer(source.close)
