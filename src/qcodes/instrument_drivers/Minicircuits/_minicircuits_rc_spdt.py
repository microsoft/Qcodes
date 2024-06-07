from typing import TYPE_CHECKING, Optional

from qcodes import validators as vals
from qcodes.instrument import (
    ChannelList,
    InstrumentBaseKWArgs,
    InstrumentChannel,
    IPInstrument,
)

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.parameters import Parameter


class MiniCircuitsRCSPDTChannel(InstrumentChannel):
    def __init__(
        self,
        parent: "MiniCircuitsRCSPDT",
        name: str,
        channel_letter: str,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ):
        """
        Args:
            parent: The instrument the channel is a part of
            name: the name of the channel
            channel_letter: channel letter ['a', 'b', 'c' or 'd'])
            **kwargs: Forwarded to the baseclass
        """

        super().__init__(parent, name, **kwargs)
        self.channel_letter = channel_letter.upper()
        _chanlist = ["a", "b", "c", "d", "e", "f", "g", "h"]
        self.channel_number = _chanlist.index(channel_letter)

        self.switch: Parameter = self.add_parameter(
            "switch",
            label="switch",
            set_cmd=self._set_switch,
            get_cmd=self._get_switch,
            vals=vals.Ints(1, 2),
        )
        """Parameter switch"""

    def _set_switch(self, switch: int) -> None:
        self.write(f"SET{self.channel_letter}={switch-1}")

    def _get_switch(self) -> int:
        val = int(self.ask("SWPORT?"))
        # select out bit in return number
        # corisponding to channel switch configuration
        # LSB corrisponds to Chan A etc
        ret = (val >> self.channel_number) & 1
        return ret + 1


class MiniCircuitsRCSPDT(IPInstrument):
    """
    Mini-Circuits RC-SPDT RF switch connected
    via ethernet.

    Args:
        name: the name of the instrument
        address: ip address ie "10.0.0.1"
        port: port to connect to default Telnet:23
    """

    def __init__(
        self,
        name: str,
        address: str,
        port: int = 23,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ):
        super().__init__(name, address, port, **kwargs)
        self.flush_connection()

        channels = ChannelList(
            self, "Channels", MiniCircuitsRCSPDTChannel, snapshotable=False
        )

        _chanlist = ["a", "b", "c", "d", "e", "f", "g", "h"]
        _max_channel_number = int(self.IDN()["model"][3])
        _chanlist = _chanlist[0:_max_channel_number]

        for c in _chanlist:
            channel = MiniCircuitsRCSPDTChannel(self, f"channel_{c}", c)
            channels.append(channel)
            self.add_submodule(f"channel_{c}", channel)
        self.add_submodule("channels", channels.to_channel_tuple())

        self.connect_message()

    def ask(self, cmd: str) -> str:
        ret = self.ask_raw(cmd)
        ret = ret.strip()
        return ret

    def get_idn(self) -> dict[str, Optional[str]]:
        fw = self.ask("FIRMWARE?")
        MN = self.ask("MN?")
        SN = self.ask("SN?")

        id_dict: dict[str, Optional[str]] = {
            "firmware": fw,
            "model": MN[3:],
            "serial": SN[3:],
            "vendor": "Mini-Circuits",
        }
        return id_dict
