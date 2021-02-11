from typing import Any, Optional, Dict

from qcodes import Parameter, DelegateParameter
from qcodes.instrument.base import Instrument
from qcodes.utils.validators import Enum, Numbers


class SP983C(Instrument):
    """
    A virtual driver for the Basel SP 983 and SP 983C current to voltage
    converter.

    This driver supports both the SP 983 and SP 983C models. These differ only
    in their handling of input offset voltage. It is the responsibility of the
    user to capture the input offset, (from the voltage supply) and compensate
    that as needed for SP 983. For SP 983C model, 'input_offset_voltage'
    argument can be used to set up offset (This doesn't work for SP 983c01
    model).

    Note that, as this is a purely virtual driver, there is no support
    for the the remote control interface (SP 983a). It is the responsibility of
    the user to ensure that values set here are in accordance with the values
    set on the instrument.

    Args:
        name
        input_offset_voltage: (Optional) A source input offset voltage
            parameter. The range for input is -10 to 10 Volts and it is
            user's responsibility to ensure this. This source parameter is
            used to set offset voltage parameter of the preamp and the
            source parameter should represent a voltage source that is
            connected to the "Offset Input Volgate" connector of the SP983C.
    """

    def __init__(self, name: str,
                 input_offset_voltage: Optional[Parameter] = None,
                 **kwargs: Any):
        super().__init__(name, **kwargs)

        self.add_parameter('gain',
                           initial_value=1e8,
                           label='Gain',
                           unit='V/A',
                           get_cmd=None, set_cmd=None,
                           vals=Enum(1e09, 1e08, 1e07, 1e06, 1e05))

        self.add_parameter('fcut',
                           initial_value=1e3,
                           label='Cutoff frequency',
                           unit='Hz',
                           get_cmd=None, set_cmd=None,
                           vals=Enum(30., 100., 300., 1e3, 3e3, 10e3, 30e3,
                                     100e3, 1e6))

        self.add_parameter('offset_voltage',
                           label="Offset Voltage",
                           unit='V',
                           vals=Numbers(-0.1, 0.1),
                           scale=100,
                           source=input_offset_voltage,
                           parameter_class=DelegateParameter)

    def get_idn(self) -> Dict[str, Optional[str]]:
        vendor = 'Physics Basel'
        model = 'SP 983(c)'
        serial = None
        firmware = None
        return {'vendor': vendor, 'model': model,
                'serial': serial, 'firmware': firmware}
