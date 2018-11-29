import numpy as np
import itertools
from collections import defaultdict
import re

from qcodes import VisaInstrument, InstrumentChannel, ChannelList
from qcodes.utils.validators import Enum


class ChannelNotPresentError(Exception):
    pass


class LockAcquisitionError(Exception):
    pass


class RelayLock:
    def __init__(self):
        self._acquired = False
        self._acquired_by = None

    def acquire(self, requester_id):
        if self._acquired and self._acquired_by != requester_id:
            raise LockAcquisitionError(
                f"Relay already in use by channel {self.acquired_by}"
            )

        self._acquired = True
        self._acquired_by = requester_id

    def release(self, requester_id):
        if self._acquired_by != requester_id:
            # It should be impossible to get here. There is a driver bug
            # if we do.
            raise RuntimeError(
                f"Relay can only be freed by channel {self.acquired_by} "
                f"that acquired the lock. Please get in touch with a QCoDeS "
                f"developer to get to the root cause of this error"
            )

        self._acquired = False
        self._acquired_by = None

    @property
    def acquired_by(self):
        return self._acquired_by


class RFSwitchChannel(InstrumentChannel):
    def __init__(self, parent, name, channel_number, relay_locks):
        super().__init__(parent, name)
        self._channel_number = channel_number

        self._relay_id = self._get_relay()
        self._relay_lock = relay_locks[self._relay_id]

        if self._get_state() == "close":
            try:
                self._relay_lock.acquire(self._channel_number)
            except LockAcquisitionError as e:
                raise RuntimeError(
                    "The driver is initialized from an undesirable instrument "
                    "state where more then one channel on a single relay is "
                    "closed. It is advised to power cycle the instrument or to "
                    "manually send the 'OPEN:ALL' SCPI command to get the "
                    "instrument back into a normal state. Refusing to "
                    "initialize driver!"
                ) from e

        self.add_parameter(
            "state",
            get_cmd=self._get_state,
            set_cmd=f":{{}} (@{self._channel_number})",
            set_parser=self._set_state_parser,
            vals=Enum("open", "close")
        )

    def _get_relay(self):
        relay_layout = self.parent.relay_layout
        found = False

        count = 0
        for count, total_count in enumerate(np.cumsum(relay_layout)):
            if total_count >= self._channel_number:
                found = True
                break

        if not found:
            raise ChannelNotPresentError()

        return (["A", "B", "C", "D"] + [str(i) for i in range(1, 9)])[count]

    def _get_state(self):
        closed_channels = re.findall(r"(\d+)[,)]",  self.ask(":CLOS?"))
        return "close" \
            if str(self._channel_number) in closed_channels \
            else "open"

    def _set_state_parser(self, new_state: str):

        if new_state == "close":
            self._relay_lock.acquire(self._channel_number)
        elif new_state == "open" \
                and self._relay_lock.acquired_by == self._channel_number:

            self._relay_lock.release(self._channel_number)

        return new_state.upper()

    @property
    def relay_id(self):
        return self._relay_id


class S46(VisaInstrument):
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator="\n", **kwargs)

        self._relay_layout = [
            int(i) for i in self.ask(":CONF:CPOL?").split(",")
        ]

        channels = ChannelList(
            self, "channel", RFSwitchChannel, snapshotable=False
        )

        relay_locks = defaultdict(RelayLock)

        for chn in itertools.count(1):
            chn_name = f"channel{chn}"

            try:
                channel = RFSwitchChannel(self, chn_name, chn, relay_locks)
            except ChannelNotPresentError:
                break

            self.add_submodule(chn_name, channel)
            channels.append(channel)

        self.add_submodule("channels", channels)
        self.connect_message()

    @property
    def relay_layout(self):
        return self._relay_layout
