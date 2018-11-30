import numpy as np
import re

from typing import cast
from qcodes import VisaInstrument, InstrumentChannel, ChannelList
from qcodes.utils.validators import Enum


class LockAcquisitionError(Exception):
    pass


class S46Channel(InstrumentChannel):
    def __init__(self, parent, name, channel_number, relay):
        super().__init__(parent, name)

        self._channel_number = channel_number
        self._relay = relay
        if self._get_state() == "close":
            try:
                self._relay.acquire_lock(self._channel_number)
            except LockAcquisitionError as e:
                raise RuntimeError(
                    "The driver is initialized from an undesirable instrument "
                    "state where more then one channel on a single relay is "
                    "closed. It is advised to power cycle the instrument. "
                    "Refusing to initialize driver!"
                ) from e

        self.add_parameter(
            "state",
            get_cmd=self._get_state,
            set_cmd=self._set_state,
            vals=Enum("open", "close")
        )

    def _get_state(self):
        is_closed = self._channel_number in \
                    self.root_instrument.get_closed_channel_numbers()

        return {True: "close", False: "open"}[is_closed]

    def _set_state(self, new_state: str):

        if new_state == "close":
            self._relay.acquire_lock(self._channel_number)
        elif new_state == "open":
            self._relay.release_lock(self._channel_number)

        self.write(f":{new_state} (@{self._channel_number})")

    @property
    def channel_number(self):
        return self._channel_number


class S46Relay(InstrumentChannel):
    def __init__(self, parent, name, channel_offset, channel_count):
        super().__init__(parent, name)

        self._channel_offset = channel_offset
        self._locked_by = None

        channels = ChannelList(
            cast(VisaInstrument, self),
            "channel",
            S46Channel,
            snapshotable=False
        )

        for count, channel_number in enumerate(range(
                channel_offset, channel_offset + channel_count)):

            if channel_count == 1:
                channel_name = self.short_name
            else:
                channel_name = f"{self.short_name}{count + 1}"

            channel = S46Channel(
                cast(VisaInstrument, self.parent),
                channel_name,
                channel_number,
                self
            )
            channels.append(channel)

        self.add_submodule("channels", channels)

    def acquire_lock(self, channel_number):

        if self._locked_by is not None and self._locked_by != channel_number:
            raise LockAcquisitionError(
                f"Relay {self.short_name} is already in use by channel "
                f"{self._locked_by}"
            )
        else:
            self._locked_by = channel_number

    def release_lock(self, channel_number):

        if self._locked_by == channel_number:
            self._locked_by = None


class S46(VisaInstrument):
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator="\n", **kwargs)

        relay_layout = [
            int(i) for i in self.ask(":CONF:CPOL?").split(",")
        ]
        relay_names = (["A", "B", "C", "D"] + [f"R{i}" for i in range(1, 9)])
        # Channel offsets are independent of pole configuration. See page
        # 2-5 of the manual
        channel_offsets = np.cumsum([0] + 4 * [6] + 7 * [1]) + 1

        channels = ChannelList(
            self,
            "channel",
            S46Channel,
            snapshotable=False
        )

        for name, channel_offset, channel_count in zip(
                relay_names, channel_offsets, relay_layout):

            relay = S46Relay(self, name, channel_offset, channel_count)
            for channel in relay.channels:
                channels.append(channel)
                self.add_submodule(channel.short_name, channel)

        self.add_submodule("channels", channels)
        self.connect_message()

    def get_closed_channel_numbers(self):
        closed_channels_str = re.findall(r"\d+", self.ask(":CLOS?"))
        return [int(i) for i in closed_channels_str]

    def get_closed_channels(self):
        return [
            channel for channel in self.channels if
            channel.channel_number in self.get_closed_channel_numbers()
        ]

    def open_all_channels(self):
        for channel in self.get_closed_channels():
            cast(S46Channel, channel).state("open")
