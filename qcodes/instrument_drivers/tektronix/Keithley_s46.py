import numpy as np
import itertools
from collections import defaultdict

from qcodes import VisaInstrument, InstrumentChannel, ChannelList


class RelayLock:
    def __init__(self):
        self.acquired = False
        self.acquired_by = None

    def acquire(self, requester_id):
        if self.acquired and self.acquired_by != requester_id:
            raise RuntimeError(
                "Relay {self.name} already in use by another channel on the "
                "same relay"
            )

        self.acquired = True
        self.acquired_by = requester_id

    def release(self, requester_id):
        if self.acquired_by != requester_id:
            raise RuntimeError(
                "Relay can only be freed by the channel that acquired the lock")

        self.acquired = False
        self.acquired_by = None


class ChannelNotPresentError(Exception):
    pass


class RFSwitchChannel(InstrumentChannel):
    def __init__(self, parent, name, channel_number, relay_locks):
        super().__init__(parent, name)
        self._channel_number = channel_number

        relay_id = self._get_relay()
        self._relay_lock = relay_locks[relay_id]

        self.add_parameter(
            "is_closed",
            get_cmd=lambda: self._channel_number in
                            self.parent.query_closed_channels()
        )

    def _get_relay(self):
        relay_layout = self.parent.get_relay_layout()
        found = False

        count = 0
        for count, total_count in enumerate(np.cumsum(relay_layout)):
            if total_count >= self._channel_number:
                found = True
                break

        if not found:
            raise ChannelNotPresentError()

        return ["A", "B", "C", "D", "1", "2", "3", "4", "5", "6", "7", "8"][
            count]

    def close_channel(self):
        self._relay_lock.acquire(self._channel_number)
        self.write(f":CLOSE (@{self._channel_number})")

    def open_channel(self):
        self._relay_lock.release(self._channel_number)
        self.write(f":OPEN (@{self._channel_number})")


class S46(VisaInstrument):
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator="\n", **kwargs)

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

    def query_closed_channels(self):
        response = self.ask(":CLOS?")
        channels_list_string = response.lstrip("(@").rstrip(")")
        return [int(i) for i in channels_list_string.split(",")]

    def get_relay_layout(self):
        return [int(i) for i in self.ask(":CONF:CPOL?").split(",")]