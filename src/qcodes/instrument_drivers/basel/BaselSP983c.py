from .BaselSP983 import BaselSP983


class BaselSP983c(BaselSP983):
    """
    A virtual driver for the Basel SP 983c current to voltage converter.

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
    """

    def get_idn(self) -> dict[str, str | None]:
        vendor = "Physics Basel"
        model = "SP 983c"
        serial = None
        firmware = None
        return {
            "vendor": vendor,
            "model": model,
            "serial": serial,
            "firmware": firmware,
        }
