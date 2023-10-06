from typing import Any, Optional

from qcodes.instrument import Instrument

from .USBHIDMixin import MiniCircuitsHIDMixin


class MiniCircuitsRudat13G90Base(Instrument):
    """
    Args:
        name (str)
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        super().__init__(name, **kwargs)

        self.add_parameter("model_name", get_cmd=":MN?")

        self.add_parameter("serial_number", get_cmd=":SN?")

        self.add_parameter("firmware", get_cmd=":FIRMWARE?")

        self.add_parameter(
            "attenuation", set_cmd=":SETATT={}", get_cmd=":ATT?", get_parser=float
        )

        self.add_parameter(
            "startup_attenuation",
            set_cmd=":STARTUPATT:VALUE:{}",
            get_cmd=":STARTUPATT:VALUE?",
            get_parser=float,
        )

        self.add_parameter("hop_points", get_cmd="HOP:POINTS?", get_parser=int)

        self.connect_message()

    def get_idn(self) -> dict[str, Optional[str]]:
        model = self.model_name()
        serial = self.serial_number()
        firmware = self.firmware()

        return {
            "vendor": "Mini-Circuits",
            "model": model,
            "serial": serial,
            "firmware": firmware,
        }


class MiniCircuitsRudat13G90Usb(MiniCircuitsHIDMixin, MiniCircuitsRudat13G90Base):
    """
    Driver for the Minicircuits RUDAT-13G-90
    90 dB Programmable Attenuator
    """

    vendor_id = 0x20CE
    product_id = 0x0023
