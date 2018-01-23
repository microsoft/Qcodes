from numpy import rad2deg, deg2rad
from qcodes import VisaInstrument, validators as vals


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
                           get_parser=rad2deg,
                           set_parser=deg2rad,
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
                           # Only listed most common spellings idealy want a
                           # .upper val for Enum or string
                           vals=vals.Enum('on', 'On', 'ON',
                                          'off', 'Off', 'OFF'))

        self.connect_message()

    def on(self):
        self.set('status', 'on')

    def off(self):
        self.set('status', 'off')
