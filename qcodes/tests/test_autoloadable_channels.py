"""
Test the auto-loadable channels and channels list. These channels are helpful
when dealing with channel types which can be added or deleted from an
instrument. Please note that `channel` in this context does not necessarily
mean a physical instrument channel, but rather an instrument sub-module.
"""

from typing import List, Union, Any
import pytest
import re

from qcodes import Instrument
from qcodes.instrument.channel import (
    AutoLoadableInstrumentChannel, AutoLoadableChannelList, InstrumentChannel
)


class MockBackendBase:
    """
    A very simple mock backend that contains a dictionary with string keys and callable
    values. The keys are matched to input commands with regular expressions and on
    match the corresponding callable is called.
    """
    def __init__(self)->None:
        self._command_dict = {}

    def send(self, cmd: str)->Any:
        """
        Instead of sending a string to the Visa backend, we use this function as a mock
        """
        keys = self._command_dict.keys()
        ans = None
        key = ""
        for key in keys:
            ans = re.match(key, cmd)
            if ans is not None:
                break

        if ans is None:
            raise ValueError(f"Command {cmd} unknown")

        args = ans.groups()
        return self._command_dict[key](*args)


class MockBackend(MockBackendBase):
    """
    A mock backend for our test instrument. It defines the following SCPI (like) commands:

    INST:CHN<n>:HLO
    Return the string "Hello from channel <n>" where n is a channel number

    INST:CHN:ADD <n>
    Add a channel with channel number n

    INST:CHN:DEL <n>
    Delete a channel with channel number n

    INST:CHN:CAT
    Return a catalog of currently defined channels
    """
    def __init__(self)->None:
        super().__init__()
        self._channel_catalog = ["1", "2", "4"]  # Pre-existing channels

        self._command_dict = {
            r":INST:CHN(\d):HLO": lambda chn: f"Hello from channel {chn}",
            r":INST:CHN:ADD (\d)": self._channel_catalog.append,
            r":INST:CHN:DEL (\d)": self._channel_catalog.remove,
            r":INST:CHN:CAT": lambda: ",".join([str(i) for i in self._channel_catalog])
        }


class SimpleTestChannel(AutoLoadableInstrumentChannel):
    """
    A channel to test if we can create and delete channel instances
    """

    @classmethod
    def _discover_from_instrument(
            cls, parent: Instrument, **kwargs) -> List[dict]:
        """
        New channels need `name` and `channel` keyword arguments.
        """
        channels_str = parent.channel_catalog()
        kwarg_list = [
            {"name": f"channel{i}", "channel": int(i)}
            for i in channels_str.split(",")
        ]

        return kwarg_list

    @classmethod
    def _get_new_instance_kwargs(cls, parent: Instrument=None, **kwargs) -> dict:
        """
        Find the smallest channel number not yet occupied
        """
        channels_str = parent.channel_catalog()
        existing_channels = [int(i) for i in channels_str.split(",")]

        new_channel = 1
        while new_channel in existing_channels:
            new_channel += 1

        kwargs = {
            "name": f"channel{new_channel}",
            "channel": new_channel
        }

        return kwargs

    def __init__(
            self,
            parent: Union[Instrument, 'InstrumentChannel'],
            name: str,
            channel: int,
            existence: bool = False,
            channel_list: 'AutoLoadableChannelList' = None,
    ) -> None:

        super().__init__(parent, name, existence, channel_list)
        self._channel = channel

        self.add_parameter(
            "hello",
            get_cmd=f":INST:CHN{self._channel}:HLO"
        )

    def _create(self)->None:
        """Create the channel on the instrument"""
        self.parent.root_instrument.write(f":INST:CHN:ADD {self._channel}")

    def _remove(self)->None:
        """Remove the channel from the instrument"""
        self.write(f":INST:CHN:DEL {self._channel}")


class DummyInstrument(Instrument):
    """
    This dummy instrument allows the creation and deletion of
    channels
    """

    def __init__(self, name: str)->None:
        super().__init__(name)

        self._backend = MockBackend()

        self.add_parameter(
            "channel_catalog",
            get_cmd=":INST:CHN:CAT",
        )

        channels = AutoLoadableChannelList(self, "channels", SimpleTestChannel)
        self.add_submodule("channels", channels)

    def write_raw(self, cmd: str)->None:
        self._backend.send(cmd)

    def ask_raw(self, cmd: str)->str:
        return self._backend.send(cmd)


@pytest.fixture(scope='function')
def dummy_instrument():
    instrument = DummyInstrument("instrument")
    yield instrument
    instrument.close()


def test_sanity(dummy_instrument):
    """
    Test the basic functionality of the auto-loadable channels, without using
    the auto-loadable channels list
    """
    # Test that we are able to discover instruments automatically
    channels = SimpleTestChannel.load_from_instrument(dummy_instrument)
    assert len(channels) == 3
    assert channels[0].hello() == "Hello from channel 1"
    assert channels[1].hello() == "Hello from channel 2"
    assert channels[2].hello() == "Hello from channel 4"
    # Test that we can generate a new instance of the channels without conflicting names
    new_channel_kwargs = SimpleTestChannel._get_new_instance_kwargs(dummy_instrument)
    new_channel = SimpleTestChannel(dummy_instrument, **new_channel_kwargs)
    # Instrument IO through the newly instantiated channel should raise an
    # exception before actually creating the channel on the instrument
    with pytest.raises(RuntimeError) as e1:
        new_channel.hello()
    assert e1.value.args[0] == "Object does not exist (anymore) on the instrument"
    # After creating the channel we should be able to talk to the instrument.
    new_channel.create()
    assert new_channel.hello() == "Hello from channel 3"
    # Once we remove the channel we should no longer be able to talk to the instrument
    new_channel.remove()
    with pytest.raises(RuntimeError) as e2:
        new_channel.hello()  # We have deleted the channel and it should no longer be
        # available
    assert e2.value.args[0] == "Object does not exist (anymore) on the instrument"


def test_channels_list(dummy_instrument):
    """
    Test the auto-loadable channels list
    """
    assert dummy_instrument.channels[0].hello() == "Hello from channel 1"
    assert dummy_instrument.channels[1].hello() == "Hello from channel 2"
    assert dummy_instrument.channels[2].hello() == "Hello from channel 4"
    assert len(dummy_instrument.channels) == 3
    # Test that we can add channels
    new_channel = dummy_instrument.channels.add()
    assert len(dummy_instrument.channels) == 4
    assert new_channel is dummy_instrument.channels[-1]
    assert new_channel.hello() == "Hello from channel 3"
    # Test that we can remove them
    new_channel.remove()
    assert len(dummy_instrument.channels) == 3
    assert new_channel not in dummy_instrument.channels
    # Once removed we should no longer be able to talk to the channel
    with pytest.raises(RuntimeError) as e:
        new_channel.hello()
    assert e.value.args[0] == "Object does not exist (anymore) on the instrument"
