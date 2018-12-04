"""
Driver for the Tekronix S46 RF switch
"""
import re
from itertools import product

from typing import Callable, Any, Union, Dict, List
from qcodes import Instrument, VisaInstrument, InstrumentChannel, ChannelList
from qcodes.utils.validators import Enum


class LockAcquisitionError(Exception):
    pass


def cached_method(method: Callable) -> Callable:
    """
    A decorator which adds a keyword 'get_cached' to a method. When
    'get_cached=True', the decorated method returns a cached return value. If
    the method has not been called before, or 'get_cached=False' the original
    method is called.
    """
    def inner(self, *args: Any, get_cached: bool=False, **kwargs: Any) -> Any:
        if not hasattr(self, "__cached_values__"):
            self.__cached_values__ = {}

        if method not in self.__cached_values__ or not get_cached:
            self.__cached_values__[method] = method(self, *args, **kwargs)

        return self.__cached_values__[method]
    return inner


class RelayLock:
    """
    The S46 either has six pole or a four pole relays. For example, channels
    'A1' to 'A6' are all on relay 'A'. However, channels 'R1' to 'R8' are all on
    individual relays.

    Only one channel per relay may be closed at any given time to prevent
    degradation of RF performance and even switch damage. See page 2-11
    of the manual. To enforce this, a lock mechanism has been implemented.
    """
    def __init__(self, relay_name: str):
        self.relay_name = relay_name
        self._locked_by: int = None

    def acquire(self, channel_number: int):
        """
        Request a lock acquisition
        """
        if self._locked_by is not None and self._locked_by != channel_number:
            raise LockAcquisitionError(
                f"Relay {self.relay_name} is already in use by channel "
                f"{self._locked_by}"
            )
        else:
            self._locked_by = channel_number

    def release(self, channel_number: int):
        """
        Release a lock.
        """
        if self._locked_by == channel_number:
            self._locked_by = None


class S46Channel(InstrumentChannel):
    """
    A channel class for the S46

    Args:
        parent
        name
        channel_number: unlike other instruments, channel numbers on the
            S46 will not be contiguous. That is, we may have channels 1, 2 and 4
            but channel 3 may be missing.
        relay_lock: When closing the channel, request a lock acquisition.
            Release the lock when opening
    """
    def __init__(
            self,
            parent: Union[Instrument, 'InstrumentChannel'],
            name: str,
            channel_number: int,
            relay_lock: RelayLock
    ):
        super().__init__(parent, name)

        self._channel_number = channel_number
        self._relay_lock = relay_lock
        # Acquire the lock if upon init we find the channel closed.
        if self._get_state(init=True) == "close":
            try:
                self._relay_lock.acquire(self._channel_number)
            except LockAcquisitionError as e:
                # another channel has already acquired the lock
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

    def _get_state(self, init: bool=False) -> str:
        """
        Args:
            init: When calling this method from self.__init__, make this
                value 'True'. This will prevent the instrument being queried
                ':CLOS?' for each channel.
        """
        closed_channels = self.parent.get_closed_channel_numbers(
            get_cached=init)

        is_closed = self._channel_number in closed_channels
        return {True: "close", False: "open"}[is_closed]

    def _set_state(self, new_state: str) -> None:
        """
        Open/Close the channel
        """
        if new_state == "close":
            self._relay_lock.acquire(self._channel_number)
        elif new_state == "open":
            self._relay_lock.release(self._channel_number)

        self.write(f":{new_state} (@{self._channel_number})")

    @property
    def channel_number(self) -> int:
        return self._channel_number


class S46(VisaInstrument):

    relay_names: list = ["A", "B", "C", "D"] + [f"R{j}" for j in range(1, 9)]

    # Make a dictionary where keys are channel aliases (e.g. 'A1', 'B3', etc)
    # and values are corresponding channel numbers.
    aliases: Dict[str, int] = {
        f"{a}{b}": count + 1 for count, (a, b) in enumerate(product(
            ["A", "B", "C", "D"],
            range(1, 7)
        ))
    }
    aliases.update({f"R{i}": i + 24 for i in range(1, 9)})

    def __init__(
            self,
            name: str,
            address: str,
            **kwargs: Any
    ):

        super().__init__(name, address, terminator="\n", **kwargs)

        channels = ChannelList(
            self,
            "channel",
            S46Channel,
            snapshotable=False
        )

        for relay_name, channel_count in zip(
                S46.relay_names, self.relay_layout):

            relay_lock = RelayLock(relay_name)

            for channel_index in range(1, channel_count + 1):
                # E.g. For channel 'B2', channel_index is 2
                if channel_count > 1:
                    alias = f"{relay_name}{channel_index}"
                else:
                    alias = relay_name  # For channels R1 to R8, we have one
                    # channel per relay. Channel alias = relay name

                channel_number = S46.aliases[alias]
                channel = S46Channel(self, alias, channel_number, relay_lock)
                channels.append(channel)
                self.add_submodule(alias, channel)

        self.add_submodule("channels", channels)
        self.connect_message()

    @cached_method
    def get_closed_channel_numbers(self) -> List[int]:
        """
        Query the instrument for closed channels. Add an option to return
        a cached response so we can prevent this method from being called
        repeatedly for each channel during initialization of the driver.
        """
        closed_channels_str = re.findall(r"\d+", self.ask(":CLOS?"))
        return [int(i) for i in closed_channels_str]

    def get_closed_channels(self) -> List[S46Channel]:
        """
        Return a list of closed channels as a list of channel objects
        """
        return [
            channel for channel in self.channels if
            channel.channel_number in self.get_closed_channel_numbers()
        ]

    def open_all_channels(self) -> None:
        """
        Please do not write ':OPEN ALL' to the instrument as this will
        circumvent the lock.
        """
        for channel in self.get_closed_channels():
            channel.state("open")

    @property
    def relay_layout(self) -> List[int]:
        """
        The relay layout tells us how many channels we have per relay. Note
        that we can have zero channels per relay.
        """
        return [int(i) for i in self.ask(":CONF:CPOL?").split(",")]
