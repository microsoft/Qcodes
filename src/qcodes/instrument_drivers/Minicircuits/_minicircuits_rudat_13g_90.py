from typing import TYPE_CHECKING

from qcodes.instrument import Instrument, InstrumentBaseKWArgs

from .USBHIDMixin import MiniCircuitsHIDMixin

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.parameters import Parameter


class MiniCircuitsRudat13G90Base(Instrument):
    def __init__(self, name: str, **kwargs: "Unpack[InstrumentBaseKWArgs]") -> None:
        """
        Base class for drivers for MiniCircuits RUDAT-13G-90
        Should not be instantiated directly.

        Args:
            name: Name of the instrument
            **kwargs: Forwarded to base class.
        """
        super().__init__(name, **kwargs)

        self.model_name: Parameter = self.add_parameter("model_name", get_cmd=":MN?")
        """Parameter model_name"""

        self.serial_number: Parameter = self.add_parameter(
            "serial_number", get_cmd=":SN?"
        )
        """Parameter serial_number"""

        self.firmware: Parameter = self.add_parameter("firmware", get_cmd=":FIRMWARE?")
        """Parameter firmware"""

        self.attenuation: Parameter = self.add_parameter(
            "attenuation", set_cmd=":SETATT={}", get_cmd=":ATT?", get_parser=float
        )
        """Parameter attenuation"""

        self.startup_attenuation: Parameter = self.add_parameter(
            "startup_attenuation",
            set_cmd=":STARTUPATT:VALUE:{}",
            get_cmd=":STARTUPATT:VALUE?",
            get_parser=float,
        )
        """Parameter startup_attenuation"""

        self.hop_points: Parameter = self.add_parameter(
            "hop_points", get_cmd="HOP:POINTS?", get_parser=int
        )
        """Parameter hop_points"""

        self.connect_message()

    def get_idn(self) -> dict[str, str | None]:
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
