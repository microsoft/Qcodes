from numpy import pi

from qcodes import VisaInstrument, validators as vals


class Agilent_E8527D(VisaInstrument):
    '''
    This is the qcodes driver for the Agilent_E8527D signal generator

    Status: beta-version.
        TODO:
        - Add all parameters that are in the manual

    This driver will most likely work for multiple Agilent sources.

    This driver does not contain all commands available for the E8527D but
    only the ones most commonly used.
    '''
    def __init__(self, name, address, step_attenuator=False, **kwargs):
        super().__init__(name, address, **kwargs)

        self.add_parameter(name='frequency',
                           label='Frequency',
                           units='Hz',
                           get_cmd='FREQ:CW?',
                           set_cmd='FREQ:CW' + ' {:.4f}',
                           get_parser=float,
                           set_parser=float,
                           vals=vals.Numbers(1e5, 20e9))
        self.add_parameter(name='phase',
                           label='Phase',
                           units='deg',
                           get_cmd='PHASE?',
                           set_cmd='PHASE' + ' {:.8f}',
                           get_parser=self.rad_to_deg,
                           set_parser=self.deg_to_rad,
                           vals=vals.Numbers(-180, 180))
        min_power = -135 if step_attenuator else -20
        self.add_parameter(name='power',
                           label='Power',
                           units='dBm',
                           get_cmd='POW:AMPL?',
                           set_cmd='POW:AMPL' + ' {:.4f}',
                           get_parser=float,
                           set_parser=float,
                           vals=vals.Numbers(min_power, 16))
        self.add_parameter('status',
                           get_cmd=':OUTP?',
                           set_cmd='OUTP {}',
                           get_parser=self.parse_on_off,
                           # Only listed most common spellings idealy want a
                           # .upper val for Enum or string
                           vals=vals.Enum('on', 'On', 'ON',
                                          'off', 'Off', 'OFF'))

        self.connect_message()

    # Note it would be useful to have functions like this in some module instad
    # of repeated in every instrument driver
    def rad_to_deg(self, angle_rad):
        angle_deg = float(angle_rad)/(2*pi)*360
        return angle_deg

    def deg_to_rad(self, angle_deg):
        angle_rad = float(angle_deg)/360 * 2 * pi
        return angle_rad

    def parse_on_off(self, stat):
        if stat.startswith('0'):
            stat = 'Off'
        elif stat.startswith('1'):
            stat = 'On'
        return stat

    def on(self):
        self.set('status', 'on')

    def off(self):
        self.set('status', 'off')
