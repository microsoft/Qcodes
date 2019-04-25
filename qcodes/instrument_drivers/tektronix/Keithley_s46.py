"""
Driver for the Tekronix S46 RF switch
"""
import re
from itertools import product
from functools import partial

from typing import Any, Dict, List, Optional
from qcodes import Instrument, VisaInstrument, Parameter


class LockAcquisitionError(Exception):
    pass


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
        self._locked_by: Optional[int] = None

    def acquire(self, channel_number: int) -> None:
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

    def release(self, channel_number: int) -> None:
        """
        Release a lock.
        """
        if self._locked_by == channel_number:
            self._locked_by = None


class S46Parameter(Parameter):
    """
    A parameter class for S46 channels. We do not use the QCoDeS
    InstrumentChannel class because our channel has one state parameter,
    which can either be "open" or "close".

    Args:
        name
        instrument
        channel_number
        lock: Acquire the lock when closing and release when opening
    """
    def __init__(
            self,
            name: str,
            instrument: Optional[Instrument],
            channel_number: int,
            lock: RelayLock
    ):
        super().__init__(name, instrument)

        self._lock = lock
        self._channel_number = channel_number
        self.get = partial(self._get, get_cached=False)

        if self._get(get_cached=True) == "close":
            try:
                self._lock.acquire(self._channel_number)
            except LockAcquisitionError as e:
                raise RuntimeError(
                    "The driver is initialized from an undesirable instrument "
                    "state where more then one channel on a single relay is "
                    "closed. It is advised to power cycle the instrument. "
                    "Refusing to initialize driver!"
                ) from e

        self.set = self._set

    def _get(self, get_cached):

        closed_channels = self._instrument.closed_channels.get_latest()

        if not get_cached or closed_channels is None:
            closed_channels = self._instrument.closed_channels.get()

        return "close" if self.name in closed_channels else "open"

    def _set(self, new_state) -> None:

        if new_state == "close":
            self._lock.acquire(self._channel_number)
        elif new_state == "open":
            self._lock.release(self._channel_number)

        if self._instrument is None:
            raise RuntimeError("Cannot set the value on a parameter "
                               "that is not attached to an instrument.")
        self._instrument.write(f":{new_state} (@{self._channel_number})")

    @property
    def channel_number(self) -> int:
        return self._channel_number


class S46(VisaInstrument):

    relay_names: list = ["A", "B", "C", "D"] + [f"R{j}" for j in range(1, 9)]

    # Make a dictionary where keys are channel aliases (e.g. 'A1', 'B3', etc)
    # and values are corresponding channel numbers.
    channel_numbers: Dict[str, int] = {
        f"{a}{b}": count + 1 for count, (a, b) in enumerate(product(
            ["A", "B", "C", "D"],
            range(1, 7)
        ))
    }
    channel_numbers.update({f"R{i}": i + 24 for i in range(1, 9)})
    # Make a reverse dict for efficient alias lookup given a channel number
    aliases = dict((v, k) for k, v in channel_numbers.items())

    def __init__(
            self,
            name: str,
            address: str,
            **kwargs: Any
    ):

        super().__init__(name, address, terminator="\n", **kwargs)

        self.add_parameter(
            "closed_channels",
            get_cmd=":CLOS?",
            get_parser=self._get_closed_channels_parser
        )

        self._available_channels: List[str] = []

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

                self.add_parameter(
                    alias,
                    channel_number=S46.channel_numbers[alias],
                    lock=relay_lock,
                    parameter_class=S46Parameter
                )

                self._available_channels.append(alias)

    @staticmethod
    def _get_closed_channels_parser(reply: str) -> List[str]:
        """
        The SCPI command ":CLOS ?" returns a reply in the form
        "(@1,9)", if channels 1 and 9 are closed. Return a list of
        strings, representing the aliases of the closed channels
        """
        closed_channels_str = re.findall(r"\d+", reply)
        return [S46.aliases[int(i)] for i in closed_channels_str]

    def open_all_channels(self) -> None:
        for channel_name in self.closed_channels():
            self.parameters[channel_name].set("open")

    @property
    def relay_layout(self) -> List[int]:
        """
        The relay layout tells us how many channels we have per relay. Note
        that we can have zero channels per relay.
        """
        return [int(i) for i in self.ask(":CONF:CPOL?").split(",")]

    @property
    def available_channels(self) -> List[str]:
        return self._available_channels
