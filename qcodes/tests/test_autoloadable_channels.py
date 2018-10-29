from typing import List, Union
import pytest
import re

from qcodes import Instrument
from qcodes.instrument.channel import (
    AutoLoadableInstrumentChannel, AutoLoadableChannelList, InstrumentChannel
)


class SimpleTestChannel(AutoLoadableInstrumentChannel):

    @classmethod
    def _discover_from_instrument(
            cls, parent: Instrument, **kwargs) -> List[dict]:

        channels_str = parent.channel_catalog()
        kwarg_list = [
            {"name": f"channel{i}", "channel": int(i)}
            for i in channels_str.split(",")
        ]

        return kwarg_list

    @classmethod
    def get_new_instance_kwargs(cls, parent: Instrument, **kwargs) -> dict:
        channels_str = parent.channel_catalog()
        channels = [int(i) for i in channels_str.split(",")]

        new_channel = 1
        while new_channel in channels:
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

    def _create(self):
        self.parent.add_channel(self._channel)

    def _delete(self):
        self.parent.remove_channel(self._channel)


class DummyInstrument(Instrument):
    def __init__(self, name):
        super().__init__(name)

        self._channel_catalog = [1, 2, 4]
        self.add_channel = self._channel_catalog.append
        self.remove_channel = self._channel_catalog.remove

        self.add_parameter(
            "channel_catalog",
            get_cmd=lambda: ",".join([
                str(i) for i in self._channel_catalog
            ])
        )

    def write_raw(self, cmd: str):
        pass

    def ask_raw(self, cmd: str):
        ans = re.match(r":INST:CHN(\d):HLO", cmd)
        if ans is not None:
            channel = ans.groups()[0]
            return f"Hello from channel {channel}"
        else:
            return ""


def test_sanity():

    instrument = DummyInstrument("instrument")
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
