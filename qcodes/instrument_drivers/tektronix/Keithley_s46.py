import re
from itertools import product

from typing import cast
from qcodes import VisaInstrument, InstrumentChannel, ChannelList
from qcodes.utils.validators import Enum


class LockAcquisitionError(Exception):
    pass


def cached_method(method):
    def inner(self, *args, get_cached=False, **kwargs):
        if not hasattr(self, "__cached_values__"):
            self.__cached_values__ = {}

        if method not in self.__cached_values__ or not get_cached:
            self.__cached_values__[method] = method(self, *args, **kwargs)

        return self.__cached_values__[method]

    return inner


class S46Channel(InstrumentChannel):
    def __init__(self, parent, name, channel_number, relay_lock):
        super().__init__(parent, name)

        self._channel_number = channel_number
        self._relay_lock = relay_lock

        if self._get_state(get_cached=True) == "close":
            try:
                self._relay_lock.acquire(self._channel_number)
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

    def _get_state(self, get_cached=False):
        is_closed = self._channel_number in \
                    self.parent.get_closed_channel_numbers(get_cached=get_cached)
        return {True: "close", False: "open"}[is_closed]

    def _set_state(self, new_state: str):

        if new_state == "close":
            self._relay_lock.acquire(self._channel_number)
        elif new_state == "open":
            self._relay_lock.release(self._channel_number)

        self.write(f":{new_state} (@{self._channel_number})")

    @property
    def channel_number(self):
        return self._channel_number


class RelayLock:
    def __init__(self, relay_name):
        self.relay_name = relay_name
        self._locked_by = None

    def acquire(self, channel_number):

        if self._locked_by is not None and self._locked_by != channel_number:
            raise LockAcquisitionError(
                f"Relay {self.relay_name} is already in use by channel "
                f"{self._locked_by}"
            )
        else:
            self._locked_by = channel_number

    def release(self, channel_number):

        if self._locked_by == channel_number:
            self._locked_by = None


class S46(VisaInstrument):

    relay_names = ["A", "B", "C", "D"] + [f"R{j}" for j in range(1, 9)]

    _aliases_list = [
        f"{a}{b}" for a, b in product(
            ["A", "B", "C", "D"],
            list(range(1, 7))
        )
    ]

    _aliases_list += [f"R{i}" for i in range(1, 9)]
    aliases = dict(zip(_aliases_list, range(1, 33)))

    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator="\n", **kwargs)

        channels = ChannelList(
            self,
            "channel",
            S46Channel,
            snapshotable=False
        )

        for relay_name, channel_count in zip(S46.relay_names, self.relay_layout):
            relay_lock = RelayLock(relay_name)

            for channel_index in range(1, channel_count + 1):

                if channel_count > 1:
                    alias = f"{relay_name}{channel_index}"
                else:
                    alias = relay_name

                channel_number = S46.aliases[alias]
                channel = S46Channel(self, alias, channel_number, relay_lock)
                channels.append(channel)
                self.add_submodule(alias, channel)

        self.add_submodule("channels", channels)
        self.connect_message()

    @cached_method
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

    @property
    def relay_layout(self):
        return [int(i) for i in self.ask(":CONF:CPOL?").split(",")]
