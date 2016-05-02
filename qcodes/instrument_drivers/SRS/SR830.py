from numpy import pi

from qcodes import VisaInstrument, validators as vals


class SRS_SR830(VisaInstrument):
    '''
    This is the qcodes driver for the Stanford Research Systems SR830
    Lock-in Amplifier

    Status: beta-version.
        TODO: Add all parameters that are in the manual
        TODO: Complete getters, setters and validators for ranges and time constants
        TODO: Complete auto-offset operation
    '''
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, **kwargs)

        # Identify
        self.add_parameter('IDN', get_cmd='*IDN?')

        # Source Parameters
        self.add_parameter(name='frequency',
                           label='Frequency',
                           units='Hz',
                           get_cmd='FREQ?',
                           set_cmd='FREQ {:.4f}',
                           get_parser=float,
                           set_parser=float,
                           vals=vals.Numbers(0.001, 102e3))
        self.add_parameter(name='phase',
                           label='Phase',
                           units='deg',
                           get_cmd='PHAS?',
                           set_cmd='PHAS {:.2f}',
                           get_parser=float,
                           set_parser=float,
                           vals=vals.Numbers(-180.0, 180.0))
        self.add_parameter(name='amplitude',
                           label='Amplitude',
                           units='V',
                           get_cmd='SLVL?',
                           set_cmd='SLVL {:.3f}',
                           get_parser=float,
                           set_parser=float,
                           vals=vals.Numbers(0.004,5.000))
        self.add_parameter(name='harmonic',
                           label='Harmonic',
                           get_cmd='HARM?',
                           set_cmd='HARM {:d}',
                           get_parser=int,
                           set_parser=int,
                           vals=vals.Ints(min_value=0))

        # Input Parameters
        ISRC = {'A': 0, 'A-B': 1, 'I': 2, 'I100': 3}
        self.add_parameter(name='input_source',
                           label='Input Source',
                           get_cmd='ISRC?',
                           set_cmd='ISRC {:d}',
                           val_mapping=ISRC)
        IGND = {'Float': 0, 'Ground': 1}
        self.add_parameter(name='input_shield',
                           label='Input Shield',
                           get_cmd='IGND?',
                           set_cmd='IGND {:d}',
                           val_mapping=IGND)
        ICPL = {'AC': 0, 'DC': 1}
        self.add_parameter(name='input_couple',
                   label='Input Couple',
                   get_cmd='ICPL?',
                   set_cmd='ICPL {:d}',
                   val_mapping=ICPL)
        ILIN = {'Off': 0, 'Line': 1, '2xLine': 2, 'Both': 3}
        self.add_parameter(name='input_filter',
                   label='Input Filter',
                   get_cmd='ILIN?',
                   set_cmd='ILIN {:d}',
                   val_mapping=ILIN)

        self.add_parameter(name='sensitivity',
                   label='Sensitivity',
                   get_cmd='SENS?',
                   set_cmd='SENS {:d}',
                   get_parser=self.get_sensitivity,
                   set_parser=self.set_sensitivity,
                   vals=self.ValSensitivity())
        RMOD = {'High_Reserve': 0, 'Normal': 1, 'Low_Noise': 2}
        self.add_parameter(name='Reserve',
                   label='Reserve',
                   get_cmd='RMOD?',
                   set_cmd='RMOD {:d}',
                   val_mapping=RMOD)
        self.add_parameter(name='time_constant',
                   label='Time Constant',
                   get_cmd='OFLT?',
                   set_cmd='OFLT {:d}',
                   get_parser=self.get_tc,
                   set_parser=self.set_tc,
                   vals=self.ValTC())
        OFSL = {6: 0, 12: 1, 18: 2, 24: 3}
        self.add_parameter(name='filter_slope',
                   label='Filter Slope',
                   get_cmd='OFSL?',
                   set_cmd='OFSL {:d}',
                   val_mapping=OFSL)
        self.add_parameter(name="sync_filter",
                           label="Sync Filter",
                           get_cmd="SYNC?",
                           set_cmd="SYNC {:d}",
                           get_parser=self.get_on_off,
                           set_parser=self.set_on_off,
                           vals=vals.OnOff())

        # TODO: Add auxilliary output channels
        # TODO: Add auxilliary input channels
        # TODO: Add display settings

        # Measurement Parameters
        self.add_parameter(name='X',
                           label='X',
                           get_cmd='OUTP ? 1',
                           get_parser=float)
        self.add_parameter(name='Y',
                           label='Y',
                           get_cmd='OUTP ? 2',
                           get_parser=float)
        self.add_parameter(name='R',
                           label='R',
                           get_cmd='OUTP ? 3',
                           get_parser=float)
        self.add_parameter(name='theta',
                           label='Phase',
                           get_cmd='OUTP ? 4',
                           get_parser=float)

        print(self.connect_message('IDN'))

    def get_on_off(self, state):
        '''
        Map instrument returned state to On/Off
        '''
        if state.startswith('0'):
            state = 'Off'
        elif state.startswith('1'):
            state = 'On'
        return state

    def set_on_off(self, state):
        '''
        Set boolean values. All true-like mapped to 1, all false-like mapped to 0
        '''
        if isinstance(state, str):
            state = state.lower()
        if state in [0, "off", "false"]:
            state = 0
        elif state in [1, "on", "true"]:
            state = 1
        else:
            raise ValueError("Invalid state")
        return state

    # TODO: Write sensitivity getters and setters and validators
    def get_sensitivity(self, val):
        return 0

    def set_sensitivity(self, val):
        return 0

    class ValSensitivity(vals.Validator):
        '''
        validates on off type values
        valid responses are: (0, 1), (off, on)
        '''

        def __init__(self):
            pass

        def validate(self, value, context=''):
            return True

    # TODO: Write time constant getters and setters and validators
    def get_tc(self, val):
        return 0

    def set_tc(self, val):
        return 0
    
    class ValTC(vals.Validator):
        '''
        validates on off type values
        valid responses are: (0, 1), (off, on)
        '''

        def __init__(self):
            pass

        def validate(self, value, context=''):
            return True

    '''
    Perform an auto-gain operation"
    '''
    def auto_gain(self):
        self.write("AGAN")
    '''
    Perform an auto-reserve operation
    '''
    def auto_reserve(self):
        self.write("ARSV")
    '''
    Perform an auto-phase operation
    '''
    def auto_phase(self):
        self.write("APHS")
    # TODO: Figure out syntax for auto-offset command