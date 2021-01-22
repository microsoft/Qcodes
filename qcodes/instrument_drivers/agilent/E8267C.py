from typing import Any

from qcodes import VisaInstrument, validators as vals
import numpy as np
from qcodes.utils.validators import Numbers, numbertypes


class E8267(VisaInstrument):
    """
    This is the code for Agilent E8267 Signal Generator.
    """

    def __init__(self, name: str, address: str, **kwargs: Any) -> None:
        super().__init__(name, address,  terminator='\n', **kwargs)
        # general commands
        self.add_parameter(name='frequency',
                           label='Frequency',
                           unit='Hz',
                           get_cmd='FREQ?',
                           set_cmd='FREQ {}',
                           get_parser=float,
                           vals=Numbers(min_value=100e3,
                                        max_value=40e9))
        self.add_parameter(name='freq_offset',
                           label='Frequency offset',
                           unit='Hz',
                           get_cmd='FREQ:OFFS?',
                           set_cmd='FREQ:OFFS {}',
                           get_parser=float,
                           vals=Numbers(min_value=-200e9,
                                        max_value=200e9))
        self.add_parameter('freq_mode',
                           label='Frequency mode',
                           set_cmd='FREQ:MODE {}',
                           get_cmd='FREQ:MODE?',
                           vals=vals.Enum('FIX', 'CW', 'SWE', 'LIST'))
        self.add_parameter('pulse_width',
                           label='Pulse width',
                           unit='ns',
                           set_cmd='PULM:INT:PWID {}',
                           get_cmd='PULM:INT:PWID?',
                           vals=Numbers(min_value=10e-9,
                                        max_value=20e-9))
        self.add_parameter(name='phase',
                           label='Phase',
                           unit='deg',
                           get_cmd='PHAS?',
                           set_cmd='PHAS {}',
                           get_parser=self.rad_to_deg,
                           set_parser=self.deg_to_rad,
                           vals=Numbers(min_value=-180,
                                        max_value=179))
        self.add_parameter(name='power',
                           label='Power',
                           unit='dBm',
                           get_cmd='POW?',
                           set_cmd='POW {}',
                           get_parser=float,
                           vals=Numbers(min_value=-135,
                                        max_value=25))
        self.add_parameter(name='power_offset',
                           label='Power offset',
                           unit='dBm',
                           get_cmd='POW:OFFS?',
                           set_cmd='POW:OFFS {}',
                           get_parser=float,
                           vals=Numbers(min_value=-200,
                                        max_value=200))
        self.add_parameter(name='output_rf',
                           get_cmd='OUTP?',
                           set_cmd='OUTP {}',
                           val_mapping={'OFF': 0,
                                        'ON': 1})
        self.add_parameter(name='modulation_rf',
                           get_cmd='OUTP:MOD?',
                           set_cmd='OUTP:MOD {}',
                           val_mapping={'OFF': 0,
                                        'ON': 1})
        # reset values after each reconnect
        self.power(0)
        self.power_offset(0)
        self.connect_message()
        self.add_function('reset', call_cmd='*RST')

    # functions to convert between rad and deg
    @staticmethod
    def deg_to_rad(angle_deg: numbertypes) -> "np.floating[Any]":
        return np.deg2rad(float(angle_deg))

    @staticmethod
    def rad_to_deg(angle_rad: numbertypes) -> "np.floating[Any]":
        return np.rad2deg(float(angle_rad))
