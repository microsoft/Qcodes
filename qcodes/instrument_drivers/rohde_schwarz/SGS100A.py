from qcodes import VisaInstrument, validators as vals


class RohdeSchwarz_SGS100A(VisaInstrument):
    '''
    This is the qcodes driver for the Rohde & Schwarz SGS100A signal generator

    Status: beta-version.
        TODO:
        - Add all parameters that are in the manual
        - Add test suite
        - See if there can be a common driver for RS mw sources from which
          different models inherit
    This driver will most likely work for multiple Rohde & Schwarz sources.
    it would be a good idea to group all similar RS drivers together in one
    module.
        Tested working with
            - RS_SGS100A
            - RS_SMB100A
    This driver does not contain all commands available for the RS_SGS100A but
    only the ones most commonly used.
    '''
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, **kwargs)

        self.add_parameter(name='frequency',
                           label='Frequency',
                           units='Hz',
                           get_cmd='SOUR:FREQ' + '?',
                           set_cmd='SOUR:FREQ' + ' {:.2f}',
                           get_parser=float,
                           vals=vals.Numbers(1e9, 20e9))
        self.add_parameter(name='phase',
                           label='Phase',
                           units='deg',
                           get_cmd='SOUR:PHAS' + '?',
                           set_cmd='SOUR:PHAS' + ' {:.2f}',
                           get_parser=float,
                           vals=vals.Numbers(0, 360))
        self.add_parameter(name='power',
                           label='Power',
                           units='dBm',
                           get_cmd='SOUR:POW' + '?',
                           set_cmd='SOUR:POW' + ' {:.2f}',
                           get_parser=float,
                           vals=vals.Numbers(-120, 25))
        self.add_parameter('status',
                           get_cmd=':OUTP:STAT?',
                           set_cmd=self.set_status,
                           get_parser=self.parse_on_off,
                           vals=vals.Strings())
        self.add_parameter('pulsemod_state',
                           get_cmd=':SOUR:PULM:STAT?',
                           set_cmd=self.set_pulsemod_state,
                           get_parser=self.parse_on_off,
                           vals=vals.Strings())
        self.add_parameter('pulsemod_source',
                           get_cmd='SOUR:PULM:SOUR?',
                           set_cmd=self.set_pulsemod_source,
                           vals=vals.Strings())
        self.add_function('reset', call_cmd='*RST')
        self.add_function('run_self_tests', call_cmd='*TST?')

        self.connect_message()

    def parse_on_off(self, stat):
        if stat.startswith('0'):
            stat = 'Off'
        elif stat.startswith('1'):
            stat = 'On'
        return stat

    def set_status(self, stat):
        if stat.upper() in ('ON', 'OFF'):
            self.write(':OUTP:STAT %s' % stat)
        else:
            raise ValueError('Unable to set status to %s, ' % stat +
                             'expected "ON" or "OFF"')

    def set_pulsemod_state(self, stat):
        if stat.upper() in ('ON', 'OFF'):
            self.write(':PULM:SOUR EXT')
            self.write(':SOUR:PULM:STAT %s' % stat)
        else:
            raise ValueError('Unable to set status to %s,' % stat +
                             'expected "ON" or "OFF"')

    def set_pulsemod_source(self, source):
        if source.upper() in ('INT', 'EXT'):
            self.write(':SOUR:PULM:SOUR %s' % source)
        else:
            raise ValueError('Unable to set source to %s,' % source +
                             'expected "INT" or "EXT"')

    def on(self):
        self.set('status', 'on')

    def off(self):
        self.set('status', 'off')
