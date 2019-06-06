import re
from typing import Optional, TYPE_CHECKING

from .KeysightB1500_module import B1500Module
from .message_builder import MessageBuilder
from . import constants
from .constants import ModuleKind, ChNr
if TYPE_CHECKING:
    from .KeysightB1500 import KeysightB1500


class B1520A(B1500Module):
    INSTRUMENT_CLASS = ModuleKind.CMU

    def __init__(self, parent: 'KeysightB1500', name: Optional[str], slot_nr,
                 **kwargs):
        super().__init__(parent, name, slot_nr, **kwargs)

        self.channels = (ChNr(slot_nr),)

        self.add_parameter(
            name="voltage_dc", set_cmd=self._set_voltage_dc, get_cmd=None
        )

        self.add_parameter(
            name="voltage_ac", set_cmd=self._set_voltage_ac, get_cmd=None
        )

        self.add_parameter(
            name="frequency", set_cmd=self._set_frequency, get_cmd=None
        )

        self.add_parameter(name="capacitance", get_cmd=self._get_capacitance)

    def _set_voltage_dc(self, value):
        msg = MessageBuilder().dcv(self.channels[0], value)

        self.write(msg.message)

    def _set_voltage_ac(self, value):
        msg = MessageBuilder().acv(self.channels[0], value)

        self.write(msg.message)

    def _set_frequency(self, value):
        msg = MessageBuilder().fc(self.channels[0], value)

        self.write(msg.message)

    def _set_mode(self, mode):
        """
        mode is capacitance (CpG) or impedance (RX, Z complex)

        :param mode:
        :return:
        """
        pass

    def _get_capacitance(self):
        pattern = re.compile(
            r"((?P<status>\w)(?P<chnr>\w)(?P<dtype>\w))?"
            r"(?P<value>[+-]\d{1,3}\.\d{3,6}E[+-]\d{2})"
        )
        msg = MessageBuilder().tc(
            chnum=self.channels[0], mode=constants.RangingMode.AUTO
        )

        response = self.ask(msg.message)

        parsed = [item for item in re.finditer(pattern, response)]

        if (
                len(parsed) != 2
                or parsed[0]["dtype"] != "C"
                or parsed[1]["dtype"] != "Y"
        ):
            raise ValueError("Result format not supported.")

        return float(parsed[0]["value"]), float(parsed[1]["value"])
