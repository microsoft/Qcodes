import numpy as np
import itertools
from collections import defaultdict

from qcodes import VisaInstrument, InstrumentChannel, ChannelList


class ChannelNotPresentError(Exception):
    pass


class RelayLock:
    def __init__(self):
        self.acquired = False
        self.acquired_by = None

    def acquire(self, requester_id):
        if self.acquired and self.acquired_by != requester_id:
            raise RuntimeError(
                f"Relay already in use by channel {self.acquired_by}"
            )

        self.acquired = True
        self.acquired_by = requester_id

    def release(self, requester_id):
        if self.acquired_by != requester_id:
            raise RuntimeError(
                f"Relay can only be freed by channel {self.acquired_by} "
                f"that acquired the lock"
            )

        self.acquired = False
        self.acquired_by = None


class RFSwitchChannel(InstrumentChannel):
    def __init__(self, parent, name, channel_number, relay_locks):
        super().__init__(parent, name)
        self._channel_number = channel_number

        self._relay_id = self._get_relay()
        self._relay_lock = relay_locks[self._relay_id]

        self.add_parameter(
            "close",
            get_cmd=lambda: str(self._channel_number) in self.ask(":CONF:CPOL?"),
            set_cmd=lambda state: {
                True: self._close_channel,
                False: self._open_channel
            }[state]()
        )

    def _get_relay(self):
        relay_layout = self.parent.relay_layout()
        found = False

        count = 0
        for count, total_count in enumerate(np.cumsum(relay_layout)):
            if total_count >= self._channel_number:
                found = True
                break

        if not found:
            raise ChannelNotPresentError()

        return (["A", "B", "C", "D"] + [chr(i) for i in range(1, 9)])[count]

    def _close_channel(self):
        self._relay_lock.acquire(self._channel_number)
        self.write(f":CLOSE (@{self._channel_number})")

    def _open_channel(self):
        self._relay_lock.release(self._channel_number)
        self.write(f":OPEN (@{self._channel_number})")

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
