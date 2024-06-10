from typing import TYPE_CHECKING

from qcodes.instrument import Instrument, InstrumentBaseKWArgs
from qcodes.parameters import DelegateParameter, Parameter
from qcodes.validators import Enum, Numbers

if TYPE_CHECKING:
    from typing_extensions import Unpack


class BaselSP983(Instrument):
    """
    A virtual driver for the Basel SP 983 current to voltage converter.

    This driver supports both the SP 983 and SP 983c models. These differ only
    in their handling of input offset voltage. It is the responsibility of the
    user to capture the input offset, (from the voltage supply) and compensate
    that as needed for SP 983. For SP 983c model, 'input_offset_voltage'
    argument can be used to set up offset (This doesn't work for SP 983c01
    model).

    Note that, as this is a purely virtual driver, there is no support
    for the remote control interface (SP 983a). It is the responsibility of
    the user to ensure that values set here are in accordance with the values
    set on the instrument.

    Args:
        name
        input_offset_voltage: (Optional) A source input offset voltage
            parameter. The range for input is -10 to 10 Volts and it is
            user's responsibility to ensure this. This source parameter is
            used to set offset voltage parameter of the preamp and the
            source parameter should represent a voltage source that is
            connected to the "Offset Input Voltage" connector of the SP983C.
        **kwargs: Forwarded to base class.
    """

    def __init__(
        self,
        name: str,
        input_offset_voltage: Parameter | None = None,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ):
        super().__init__(name, **kwargs)

        self.gain: Parameter = self.add_parameter(
            "gain",
            initial_value=1e8,
            label="Gain",
            unit="V/A",
            get_cmd=None,
            set_cmd=None,
            vals=Enum(1e09, 1e08, 1e07, 1e06, 1e05),
        )
        """Parameter gain"""

        self.fcut: Parameter = self.add_parameter(
            "fcut",
            initial_value=1e3,
            label="Cutoff frequency",
            unit="Hz",
            get_cmd=None,
            set_cmd=None,
            vals=Enum(30.0, 100.0, 300.0, 1e3, 3e3, 10e3, 30e3, 100e3, 1e6),
        )
        """Parameter fcut"""

        self.offset_voltage: DelegateParameter = self.add_parameter(
            "offset_voltage",
            label="Offset Voltage",
            unit="V",
            vals=Numbers(-0.1, 0.1),
            scale=100,
            source=input_offset_voltage,
            parameter_class=DelegateParameter,
        )
        """Parameter offset_voltage"""

    def get_idn(self) -> dict[str, str | None]:
        vendor = "Physics Basel"
        model = "SP 983"
        serial = None
        firmware = None
        return {
            "vendor": vendor,
            "model": model,
            "serial": serial,
            "firmware": firmware,
        }
