from typing import Any

from numpy import pi

from qcodes import VisaInstrument, validators as vals
from qcodes.utils.validators import numbertypes
from qcodes.utils.helpers import create_on_off_val_mapping


class Agilent_E8527D(VisaInstrument):
    """
    This is the QCoDeS driver for the Agilent_E8527D signal generator.

    This driver will most likely work for multiple Agilent sources.

    This driver does not contain all commands available for the E8527D but
    only the ones most commonly used.
    """
    def __init__(self, name: str, address: str,
                 step_attenuator: bool = False,
                 terminator: str = '\n',
                 **kwargs: Any) -> None:
        super().__init__(name, address, terminator=terminator, **kwargs)

        self.add_parameter(name='frequency',
                           label='Frequency',
                           unit='Hz',
                           get_cmd='FREQ:CW?',
                           set_cmd='FREQ:CW' + ' {:.4f}',
                           get_parser=float,
                           set_parser=float,
                           vals=vals.Numbers(1e5, 20e9))
        self.add_parameter(name='phase',
                           label='Phase',
                           unit='deg',
                           get_cmd='PHASE?',
                           set_cmd='PHASE' + ' {:.8f}',
                           get_parser=self.rad_to_deg,
                           set_parser=self.deg_to_rad,
                           vals=vals.Numbers(-180, 180))
        min_power = -135 if step_attenuator else -20
        self.add_parameter(name='power',
                           label='Power',
                           unit='dBm',
                           get_cmd='POW:AMPL?',
                           set_cmd='POW:AMPL' + ' {:.4f}',
                           get_parser=float,
                           set_parser=float,
                           vals=vals.Numbers(min_power, 16))
        self.add_parameter('status',
                           get_cmd=':OUTP?',
                           set_cmd='OUTP {}',
                           val_mapping=create_on_off_val_mapping(on_val='1',
                                                                 off_val='0'))

        self.connect_message()

    # Note it would be useful to have functions like this in some module instead
    # of repeated in every instrument driver.
    @staticmethod
    def rad_to_deg(angle_rad: numbertypes) -> float:
        angle_deg = float(angle_rad)/(2*pi)*360
        return angle_deg

    @staticmethod
    def deg_to_rad(angle_deg: numbertypes) -> float:
        angle_rad = float(angle_deg)/360 * 2 * pi
        return angle_rad

    def on(self) -> None:
        self.set('status', 'on')

    def off(self) -> None:
        self.set('status', 'off')
