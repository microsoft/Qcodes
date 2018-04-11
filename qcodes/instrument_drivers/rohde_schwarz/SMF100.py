from qcodes import VisaInstrument, validators as vals

class SMF100(VisaInstrument):
    """
    qcodes driver for the Rohde & Schwarz SMF100 Signal Generator.

    Args:
        name: instrument name
        address: Address of instrument probably in format
            'TCPIP0::192.168.15.100::inst0::INSTR'
        init_s_params: Automatically setup channels matching S parameters
        **kwargs: passed to base class

    TODO:
    - check initialisation settings and test functions
    """

    def __init__(self, name: str, address: str, init_s_params: bool=True, **kwargs):

        super().__init__(name=name, address=address, **kwargs)
        model = self.get_idn()['model'].split('-')[0]

        self.add_parameter('frequency', unit='Hz',
                            set_cmd='SOUR:FREQ {:.6f}',
                            get_cmd='SOUR:FREQ?',
                            vals=vals.Numbers(1e9,22e9),
                            get_parser=float)

        self.add_parameter('power', unit='dBm',
                            set_cmd='SOUR:POW {:.1f} dBm',
                            get_cmd='SOUR:POW?',
                            vals=vals.Numbers(-60,30),
                            get_parser=float)

        self.add_parameter('rf',
                            set_cmd='OUTP {}',
                            get_cmd='OUTP?',
                            vals=vals.OnOff(),
                            get_parser=self.parse_on_off)

        self.add_parameter('mod',
                            set_cmd='PULM:STAT {}',
                            get_cmd='MOD?',
                            vals=vals.OnOff(),
                            get_parser=self.parse_on_off)

        self.add_parameter('pulm_polarity',
                            set_cmd='PULM:POL {}',
                            get_cmd='PULM:POL?',
                            val_mapping={'Normal':'NORM', 'Inverted':'INV'},
                            get_parser=self.parse_str)

        self.add_parameter('pulm_video_polarity',
                            set_cmd='PULM:OUTP:VID:POL {}',
                            get_cmd='PULM:OUTP:VID:POL?',
                            val_mapping={'Normal':'NORM', 'Inverted':'INV'},
                            get_parser=self.parse_str)

        self.add_parameter('pulm_sync',
                            set_cmd='PULN:SYNC {}',
                            get_cmd='PULN:SYNC?',
                            vals=vals.OnOff(),
                            get_parser=self.parse_on_off)

        self.add_parameter('trigger_level',
                            set_cmd='PULM:TRIG:EXT:LEV {}',
                            get_cmd='PULM:TRIG:EXT:LEV?',
                            val_mapping={"Transistor-Transistor Logic" : "TTL",
                                         "-2.5V" : "M2V5", "0.5V" : "P0V5"},
                            get_parser=self.parse_str)

        self.add_parameter('external_impedance',
                            set_cmd='PULM:TRIG:EXT:LEV {}',
                            get_cmd='PULM:TRIG:EXT:LEV?',
                            val_mapping={'50 Ohm' : 'G50', '10 kOhm' : 'G10K'},
                            get_parser=self.parse_str)

    def parse_on_off(self, stat):
        if stat.startswith('0'):
            stat = 'off'
        elif stat.startswith('1'):
            stat = 'on'
        return stat

    def parse_str(self, string):
        return string.strip().upper()

    def toggle_pulm_polarity(self):
        '''
        Toggles the PUlse Modulation Polarity.
        '''
        if self.get_pulm_polarity() == 'Normal':
            self.set_pulm_polarity('Inverted')
        else:
            self.set_pulm_polarity('Normal')

    def pulse_mod_on(self, polarity = 'Normal',
                     sync = 'off',
                     video_polarity = 'Normal',
                     trigger_level = 'Transistor-Transistor Logic',
                     external_impedance = '10 kOhm'):
        '''
        Sets pulse modulation on with given settings.
        '''
        self.pulm_polarity(polarity)
        self.pulm_sync(sync)
        self.pulm_video_polarity(video_polarity)
        self.trigger_level(trigger_level)
        self.external_impedance(external_impedance)
        self.mod('on')