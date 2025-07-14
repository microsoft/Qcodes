from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from qcodes.instrument import (
    ChannelList,
    Instrument,
    InstrumentBaseKWArgs,
    InstrumentChannel,
)
from qcodes.validators import Ints

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.parameters import Parameter

log = logging.getLogger(__name__)


class MiniCircuitsSPDTSwitchChannelBase(InstrumentChannel):
    def __init__(
        self,
        parent: Instrument,
        name: str,
        channel_letter: str,
        **kwargs: Unpack[InstrumentBaseKWArgs],
    ):
        """
        Base class for MiniCircuits SPDT Switch channels.
        Should not be instantiated directly.

        Args:
            parent: The instrument the channel is a part of
            name: the name of the channel
            channel_letter: channel letter ['a', 'b', 'c' or 'd'])
            **kwargs: Forwarded to base class.

        """

        super().__init__(parent, name, **kwargs)
        self.channel_letter = channel_letter.upper()
        _chanlist = ["a", "b", "c", "d", "e", "f", "g", "h"]
        self.channel_number = _chanlist.index(channel_letter)

        self.switch: Parameter = self.add_parameter(
            "switch",
            label=f"switch {self.channel_letter}",
            set_cmd=self._set_switch,
            get_cmd=self._get_switch,
            vals=Ints(1, 2),
        )
        """Parameter switch"""

    def __call__(self, *args: int) -> int | None:
        if len(args) == 1:
            self.switch(args[0])
            return None
        elif len(args) == 0:
            return self.switch()
        else:
            raise RuntimeError("Call channel with either one or zero arguments")

    def _set_switch(self, switch: int) -> None:
        raise NotImplementedError()

    def _get_switch(self) -> int:
        raise NotImplementedError()


class MiniCircuitsSPDTBase(Instrument):
    """
    Base class for MiniCircuits SPDT Switch instruments.
    Should not be instantiated directly.
    """

    CHANNEL_CLASS: type[MiniCircuitsSPDTSwitchChannelBase]

    def add_channels(self) -> None:
        channels = ChannelList(self, "Channels", self.CHANNEL_CLASS, snapshotable=False)

        _chanlist = ["a", "b", "c", "d", "e", "f", "g", "h"]
        _max_channel_number = self.get_number_of_channels()
        _chanlist = _chanlist[0:_max_channel_number]

        for c in _chanlist:
            channel = self.CHANNEL_CLASS(self, f"channel_{c}", c)
            channels.append(channel)
            self.add_submodule(c, channel)
        self.add_submodule("channels", channels.to_channel_tuple())

    def all(self, switch_to: int) -> None:
        for c in self.channels:
            c.switch(switch_to)

    def get_number_of_channels(self) -> int:
        model = self.get_idn()["model"]
        if model is None:
            raise RuntimeError(
                "The driver could not get model information for the device, "
                "it might not be supported."
            )
        model_parts = model.split("-")
        if len(model_parts) < 2:
            raise RuntimeError(
                "The driver could not determine the number of channels of "
                f"the model '{model}', it might not be supported"
            )
        if model_parts[0] not in ("RC", "USB"):
            log.warning(
                f"The model with the name '{model}' might not be supported by"
                " the driver"
            )
        match = re.match("^[0-9]*", model_parts[1])
        if match is None:
            raise RuntimeError(
                "The driver could not determine the number of channels of"
                f" the model '{model}', it might not be supported"
            )
        channels = match[0]
        if not channels:
            raise RuntimeError(
                "The driver could not determine the number of channels of"
                f" the model '{model}', it might not be supported"
            )
        return int(channels)
