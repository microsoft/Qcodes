"""
Test the auto-loadable channels and channels list. These channels are helpful
when dealing with channel types which can be added or deleted from an
instrument. Please note that `channel` in this context does not necessarily
mean a physical instrument channel, but rather an instrument sub-module.
"""

from typing import List, Union
import pytest
import re

from qcodes import Instrument
from qcodes.instrument.channel import (
    AutoLoadableInstrumentChannel, AutoLoadableChannelList, InstrumentChannel
)


class DummyBackendBase:
    def __init__(self):
        self._command_dict = {}

    def send(self, cmd):
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


class TestInstrumentBackend(DummyBackendBase):
    def __init__(self):
        super().__init__()
        self._channel_catalog = ["1", "2", "4"]

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
    def get_new_instance_kwargs(cls, parent: Instrument, **kwargs) -> dict:
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
        self.parent.write(f":INST:CHN:ADD {self._channel}")

    def _delete(self)->None:
        """Remove the channel from the instrument"""
        self.write(f":INST:CHN:DEL {self._channel}")


class TestInstrument(Instrument):
    """
    This dummy instrument allows the creation and deletion of
    channels
    """

    def __init__(self, name):
        super().__init__(name)

        self._backend = TestInstrumentBackend()

        self.add_parameter(
            "channel_catalog",
            get_cmd=":INST:CHN:CAT",
        )

    def write_raw(self, cmd: str):
        return self._backend.send(cmd)

    def ask_raw(self, cmd: str):
        return self._backend.send(cmd)


class DummyInstrumentWithChannelList(TestInstrument):
    def __init__(self, name):
        super().__init__(name)

        channels = AutoLoadableChannelList(self, "channels", SimpleTestChannel)
        self.add_submodule("channels", channels)


def test_sanity():

    instrument = TestInstrument("instrument")
    channels = SimpleTestChannel.load_from_instrument(instrument)
    assert len(channels) == 3
    assert channels[0].hello() == "Hello from channel 1"
    assert channels[1].hello() == "Hello from channel 2"
    assert channels[2].hello() == "Hello from channel 4"

    new_channel_kwargs = SimpleTestChannel.get_new_instance_kwargs(instrument)
    new_channel = SimpleTestChannel(instrument, **new_channel_kwargs)

    with pytest.raises(RuntimeError):
        new_channel.hello()  # We have only instantiated a channel class, we
        # have yet to create it on the instrument

    new_channel.create()
    assert new_channel.hello() == "Hello from channel 3"
    new_channel.delete()

    with pytest.raises(RuntimeError):
        new_channel.hello()  # We have deleted the channel and it should no longer be
        # available


def test_channels_list():
    instrument = DummyInstrumentWithChannelList("instrument2")
    assert instrument.channels[0].hello() == "Hello from channel 1"
    assert instrument.channels[1].hello() == "Hello from channel 2"
    assert instrument.channels[2].hello() == "Hello from channel 4"

    new_channel = instrument.channels.add()
    assert len(instrument.channels) == 4
    assert new_channel is instrument.channels[-1]
    assert new_channel.hello() == "Hello from channel 3"

    new_channel.delete()
    assert len(instrument.channels) == 3
    with pytest.raises(RuntimeError):
        new_channel.hello()
