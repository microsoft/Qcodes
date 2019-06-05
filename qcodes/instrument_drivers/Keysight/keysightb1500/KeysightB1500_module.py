import re
from typing import Optional, Tuple

from qcodes import InstrumentChannel
from .message_builder import MessageBuilder
from . import constants
from .constants import InstrClass, SlotNr


# TODO notes:
# - [ ] Instead of generating a Qcodes InstrumentChannel for each **module**,
#   it might make more sense to generate one for each **channel**


class B1500Module(InstrumentChannel):
    INSTRUMENT_CLASS: InstrClass

    def __init__(self, parent: 'KeysightB1500', name: Optional[str], slot_nr,
                 **kwargs):
        self.channels: Tuple  # self.channels will be populated in the concrete
        # module subclasses because channel count is module specific
        self.slot_nr = SlotNr(slot_nr)

        if name is None:
            number = len(parent.by_class[self.INSTRUMENT_CLASS]) + 1
            name = self.INSTRUMENT_CLASS.lower() + str(number)
        super().__init__(parent=parent, name=name, **kwargs)

    def enable_outputs(self):
        """
        Enables all outputs of this module by closing the output relays of its
        channels.
        """
        # TODO This always enables all outputs of a module, which is maybe not
        # desirable. (Also check the TODO item at the top about
        # InstrumentChannel per Channel instead of per Module.
        msg = MessageBuilder().cn(self.channels).message
        self.write(msg)

    def disable_outputs(self):
        """
        Disables all outputs of this module by opening the output relays of its
         channels.
        """
        # TODO See enable_output TODO item
        msg = MessageBuilder().cl(self.channels).message
        self.write(msg)

    def is_enabled(self) -> bool:
        """
        Check if channels of this module are enabled.

        :return: `True` if *all* channels of this module are enabled. `False`
         otherwise.
        """
        # TODO If a module has multiple channels, and only one is enabled, then
        # this will return false, which is probably not desirable.
        # Also check the TODO item at the top about InstrumentChannel per
        # Channel instead of per Module.
        msg = (MessageBuilder()
               .lrn_query(constants.LRN.Type.OUTPUT_SWITCH)
               .message
               )
        response = self.ask(msg)
        activated_channels = re.sub(r"[^,\d]", "", response).split(",")

        is_enabled = set(self.channels).issubset(
            int(x) for x in activated_channels
        )
        return is_enabled
