"""
Test the auto-loadable channels and channels list. These channels are helpful
when dealing with channel types which can be added or deleted from an
instrument. Please note that `channel` in this context does not necessarily
mean a physical instrument channel, but rather an instrument sub-module.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Union

import pytest

from qcodes import Instrument
from qcodes.instrument.channel import (
    AutoLoadableChannelList,
    AutoLoadableInstrumentChannel,
    InstrumentChannel,
)


class MockBackendBase:
    """
    A very simple mock backend that contains a dictionary with string keys and
    callable values. The keys are matched to input commands with regular
    expressions and on match the corresponding callable is called.
    """
    def __init__(self) -> None:
        self._command_dict: Dict[str, Callable[..., Any]] = {}

    def send(self, cmd: str) -> Any:
        """
        Instead of sending a string to the Visa backend, we use this function
        as a mock
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
    A mock backend for our test instrument. It defines the following SCPI (like)
    commands:

    INST:CHN<n>:HLO
    Return the string "Hello from channel <n>" where n is a channel number

    INST:CHN:ADD <n>, <greeting>
    Add a channel with channel number n. We use the greeting when the hello
    parameter is called.

    INST:CHN:DEL <n>
    Delete a channel with channel number n

    INST:CHN:CAT
    Return a catalog of currently defined channels

    INST:CHN<n>:GRT
    Return the greeting of this channel
    """
    def __init__(self) -> None:
        super().__init__()
        self._channel_catalog: List[str] = ["1", "2", "4", "5"]  # Pre-existing
        # channels
        self._greetings = {chn: "Hello" for chn in self._channel_catalog}

        self._command_dict = {
            r":INST:CHN(\d):HLO":
                lambda chn: self._greetings[chn] + " from channel " + str(chn),
            r":INST:CHN:ADD (\d), (.+)": self._add_channel,
            r":INST:CHN:DEL (\d)": self._channel_catalog.remove,
            r":INST:CHN:CAT": lambda: ",".join(str(i) for i in self._channel_catalog),
            r":INST:CHN(\d):GRT": self._greetings.get,
        }

    def _add_channel(self, chn: int, greeting: str)->None:
        """
        Add a channel on the mock instrument
        """
        self._channel_catalog.append(str(chn))
        self._greetings[str(chn)] = greeting


class SimpleTestChannel(AutoLoadableInstrumentChannel):
    """
    A channel to test if we can create and delete channel instances
    """

    @classmethod
    def _discover_from_instrument(
            cls, parent: Instrument, **kwargs) -> List[Dict[Any, Any]]:
        """
        New channels need `name` and `channel` keyword arguments.
        """
        channels_str = parent.channel_catalog()
        channels_to_skip = kwargs.get("channels_to_skip", [])  # Note that
        # `channels_to_skip` is an optional kwarg for loading from instrument.
        # We test this by giving this keyword during the initialization of the
        # AutoLoadableChannelList.
        kwarg_list = []
        for channel_str in channels_str.split(","):

            if channel_str in channels_to_skip:
                continue

            channel = int(channel_str)
            greeting = parent.ask(f":INST:CHN{channel}:GRT")
            new_kwargs = {
                "name": f"channel{channel}",
                "channel": channel,
                "greeting": greeting
            }
            kwarg_list.append(new_kwargs)

        return kwarg_list

    @classmethod
    def _get_new_instance_kwargs(
            cls, parent: Optional[Instrument] = None, **kwargs
    ) -> Dict[Any, Any]:
        """
        Find the smallest channel number not yet occupied. An optional keyword
        `greeting` is extracted from the kwargs. The default is "Hello"
        """
        if parent is None:
            raise RuntimeError("SimpleTestChannel needs a parent instrument")
        channels_str = parent.channel_catalog()
        existing_channels = [int(i) for i in channels_str.split(",")]

        new_channel = 1
        while new_channel in existing_channels:
            new_channel += 1

        new_kwargs = {
            "name": f"channel{new_channel}",
            "channel": new_channel,
            "greeting": kwargs.get("greeting", "Hello")
        }

        return new_kwargs

    def __init__(
            self,
            parent: Union[Instrument, InstrumentChannel],
            name: str,
            channel: int,
            greeting: str,
            existence: bool = False,
            channel_list: Optional[AutoLoadableChannelList] = None,
    ) -> None:

        super().__init__(parent, name, existence, channel_list)
        self._channel = channel
        self._greeting = greeting

        self.add_parameter(
            "hello",
            get_cmd=f":INST:CHN{self._channel}:HLO"
        )

    def _create(self) -> None:
        """Create the channel on the instrument"""
        self.parent.root_instrument.write(
            f":INST:CHN:ADD {self._channel}, {self._greeting}")

    def _remove(self) -> None:
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

        channels = AutoLoadableChannelList(
            self, "channels", SimpleTestChannel, channels_to_skip=["5"])
        self.add_submodule("channels", channels)

    def write_raw(self, cmd: str) -> None:
        self._backend.send(cmd)

    def ask_raw(self, cmd: str) -> str:
        return self._backend.send(cmd)


@pytest.fixture(scope='function')
def dummy_instrument():
    instrument = DummyInstrument("instrument")
    yield instrument
    instrument.close()


def test_sanity(dummy_instrument):
    """
    Test the basic functionality of the auto-loadable channels, without using
    the auto-loadable channels list. Please note that the `channels_to_skip`
    argument in the dummy instrument applies when accessing channels
    from the channels list. Since we are calling `load_from_instrument` directly
    in this test without this keyword argument, we will see all channels.
    """
    # Test that we are able to discover instruments automatically
    channels = SimpleTestChannel.load_from_instrument(dummy_instrument)
    assert len(channels) == 4
    assert channels[0].hello() == "Hello from channel 1"
    assert channels[1].hello() == "Hello from channel 2"
    assert channels[2].hello() == "Hello from channel 4"
    assert channels[3].hello() == "Hello from channel 5"
    # Test that we can generate a new instance of the channels without
    # conflicting names
    new_channel_kwargs = SimpleTestChannel._get_new_instance_kwargs(
        dummy_instrument)

    new_channel = SimpleTestChannel(dummy_instrument, **new_channel_kwargs)
    # Instrument IO through the newly instantiated channel should raise an
    # exception before actually creating the channel on the instrument
    with pytest.raises(
            RuntimeError,
            match=r"Object does not exist \(anymore\) on the instrument"):

        new_channel.hello()

    # After creating the channel we should be able to talk to the instrument.
    new_channel.create()
    assert new_channel.hello() == "Hello from channel 3"

    # Once we remove the channel we should no longer be able to talk to the
    # instrument
    new_channel.remove()
    with pytest.raises(
            RuntimeError,
            match=r"Object does not exist \(anymore\) on the instrument"):

        new_channel.hello()  # We have deleted the channel and it should no
        # longer be available


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
    with pytest.raises(
            RuntimeError,
            match=r"Object does not exist \(anymore\) on the instrument"):

        new_channel.hello()
    # Remove a channel that was pre-existing on the instrument.
    dummy_instrument.channels[-1].remove()
    assert len(dummy_instrument.channels) == 2


def test_with_kwargs(dummy_instrument):
    """
    Test keyword arguments given to the add method
    """
    new_channel = dummy_instrument.channels.add(greeting="Guden tag")
    assert new_channel.hello() == "Guden tag from channel 3"
