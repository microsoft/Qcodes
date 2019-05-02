import numpy as np
from qcodes import VisaInstrument, validators as vals
from qcodes.utils.validators import Numbers


def parse_on_off(stat):
    if stat.startswith('0'):
        stat = 'Off'
    elif stat.startswith('1'):
        stat = 'On'
    return stat


class Keysight_E8267D(VisaInstrument):
    """
    This is the qcodes driver for the Keysight_E8267D signal generator

    Status: beta-version.
        TODO:
        - Add all parameters that are in the manual

    This driver will most likely work for multiple Agilent sources.

    This driver does not contain all commands available for the E8527D but
    only the ones most commonly used.
    """

    def __init__(self, name, address, step_attenuator=False, **kwargs):
        super().__init__(name, address, **kwargs)

        # Only listed most common spellings idealy want a
        # .upper val for Enum or string
        on_off_validator = vals.Enum('on', 'On', 'ON',
                                     'off', 'Off', 'OFF')
        self.add_parameter(name='frequency',
                           label='Frequency',
                           unit='Hz',
                           get_cmd='FREQ:CW?',
                           set_cmd='FREQ:CW' + ' {:.4f}',
                           get_parser=float,
                           set_parser=float,
                           vals=vals.Numbers(1e5, 20e9),
                           docstring='Adjust the RF output frequency')
        self.add_parameter(name='frequency_offset',
                           label='Frequency offset',
                           unit='Hz',
                           get_cmd='FREQ:OFFS?',
                           set_cmd='FREQ:OFFS {}',
                           get_parser=float,
                           vals=Numbers(min_value=-200e9,
                                        max_value=200e9))
        self.add_parameter('frequency_mode',
                           label='Frequency mode',
                           set_cmd='FREQ:MODE {}',
                           get_cmd='FREQ:MODE?',
                           vals=vals.Enum('FIX', 'CW', 'SWE', 'LIST'))
        self.add_parameter(name='phase',
                           label='Phase',
                           unit='deg',
                           get_cmd='PHASE?',
                           set_cmd='PHASE' + ' {:.8f}',
                           get_parser=self.rad_to_deg,
                           set_parser=self.deg_to_rad,
                           vals=vals.Numbers(-180, 180))
        self.add_parameter(name='power',
                           label='Power',
                           unit='dBm',
                           get_cmd='POW:AMPL?',
                           set_cmd='POW:AMPL' + ' {:.4f}',
                           get_parser=float,
                           set_parser=float,
                           vals=vals.Numbers(-130, 25))
        self.add_parameter('status',
                           get_cmd=':OUTP?',
                           set_cmd='OUTP {}',
                           get_parser=parse_on_off,
                           vals=on_off_validator)
        self.add_parameter(name='modulation_rf',
                           get_cmd='OUTP:MOD?',
                           set_cmd='OUTP:MOD {}',
                           get_parser=parse_on_off,
                           vals=on_off_validator)
        self.add_parameter('IQmodulator',
                           get_cmd='DM:STATe?',
                           set_cmd='DM:STATe {}',
                           get_parser=parse_on_off,
                           vals=on_off_validator,
                           docstring='Enables or disables the internal I/Q modulator. Source can be external or internal.')

        for source in [1, 2]:
            self.add_parameter(f'IQsource{source}',
                               get_cmd=f'DM:SOUR{source}?',
                               set_cmd=f'DM:SOUR{source} {{}}',
                               vals=vals.Enum('OFF', 'EXT', 'EXT600', 'INT'),
                               docstring=f'Source {source} for I/Q modulation.')

        self.connect_message()

    def on(self):
        self.set('status', 'on')

    def off(self):
        self.set('status', 'off')

    @staticmethod
    def deg_to_rad(angle_deg):
        return np.deg2rad(float(angle_deg))

    @staticmethod
    def rad_to_deg(angle_rad):
        return np.rad2deg(float(angle_rad))
