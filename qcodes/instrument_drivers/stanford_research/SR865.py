from qcodes import VisaInstrument
from qcodes.utils.validators import Numbers, Ints, Enum


class SR865(VisaInstrument):
    """
    This is the code for Stanford_SR865 Lock-in Amplifier
    """
    _VOLT_TO_N = {1: 0, 500e-3: 1, 200e-3: 2,
                  100e-3: 3, 50e-3: 4, 20e-3: 5,
                  10e-3: 6, 5e-3: 7, 2e-3: 8,
                  1e-3: 9, 500e-6: 10, 200e-6: 11,
                  100e-6: 12, 50e-6: 13, 20e-6: 14,
                  10e-6: 15, 5e-6: 16, 2e-6: 17,
                  1e-6: 18, 500e-9: 19, 200e-9: 20,
                  100e-9: 21, 50e-9: 22, 20e-9: 23,
                  10e-9: 24, 5e-9: 25, 2e-9: 26,
                  1e-9: 27}
    _N_TO_VOLT = {v: k for k, v in _VOLT_TO_N.items()}

    _CURR_TO_N = {1e-6: 0, 500e-9: 1, 200e-9: 2,
                  100e-9: 3, 50e-9: 4, 20e-9: 5,
                  10e-9: 6, 5e-9: 7, 2e-9: 8,
                  1e-9: 9, 500e-12: 10, 200e-12: 11,
                  100e-12: 12, 50e-12: 13, 20e-12: 14,
                  10e-12: 15, 5e-12: 16, 2e-12: 17,
                  1e-12: 18, 500e-15: 19, 200e-15: 20,
                  100e-15: 21, 50e-15: 22, 20e-15: 23,
                  10e-15: 24, 5e-15: 25, 2e-15: 26,
                  1e-15: 27}
    _N_TO_CURR = {v: k for k, v in _CURR_TO_N.items()}

    _VOLT_ENUM = Enum(*_VOLT_TO_N.keys())
    _CURR_ENUM = Enum(*_CURR_TO_N.keys())

    _INPUT_SIGNAL_TO_N = {
        'voltage': 0,
        'current': 1,
    }
    _N_TO_INPUT_SIGNAL = {v: k for k, v in _INPUT_SIGNAL_TO_N.items()}

    def __init__(self, name, address, reset=False,  **kwargs):
        super().__init__(name, address,  terminator='\n', **kwargs)
        # Reference commands
        self.add_parameter(name='frequency',
                           label='Frequency',
                           units='Hz',
                           get_cmd='FREQ?',
                           set_cmd='FREQ {}',
                           get_parser=float,
                           vals=Numbers(min_value=1e-3, max_value=2.5e6))
        self.add_parameter(name='sine_outdc',
                           label='Sine out dc level',
                           units='V',
                           get_cmd='SOFF?',
                           set_cmd='SOFF {:.3f}',
                           get_parser=float,
                           vals=Numbers(min_value=-5, max_value=5))
        self.add_parameter(name='amplitude',
                           label='Amplitude',
                           units='V',
                           get_cmd='SLVL?',
                           set_cmd='SLVL {}',
                           get_parser=float,
                           vals=Numbers(min_value=10e-9, max_value=2))
        self.add_parameter(name='harmonic',
                           label='Harmonic',
                           get_cmd='HARM?',
                           get_parser=int,
                           set_cmd='HARM {:d}',
                           vals=Ints(min_value=1, max_value=99))
        self.add_parameter(name='phase',
                           label='Phase',
                           units='deg',
                           get_cmd='PHAS?',
                           set_cmd='PHAS {}',
                           get_parser=float,
                           vals=Numbers(min_value=-3.6e5, max_value=3.6e5))
        # Signal commands
        self.add_parameter(name='sensitivity',
                           label='Sensitivity',
                           get_cmd='SCAL?',
                           set_cmd='SCAL {:d}',
                           get_parser=self._get_sensitivity,
                           set_parser=self._set_sensitivity
                           )
        self.add_parameter(name='filter_slope',
                           label='Filter slope',
                           units='dB/oct',
                           get_cmd='OFSL?',
                           set_cmd='OFSL {}',
                           val_mapping={6: 0,
                                        12: 1,
                                        18: 2,
                                        24: 3})
        self.add_parameter(name='sync_filter',
                           label='Sync filter',
                           get_cmd='SYNC?',
                           set_cmd='SYNC {}',
                           val_mapping={'OFF': 0,
                                        'ON': 1})
        self.add_parameter(name='noise_bandwidth',
                           label='Noise bandwidth',
                           units='Hz',
                           get_cmd='ENBW?',
                           get_parser=float)
        self.add_parameter(name='signal_strength',
                           label='Signal strength indicator',
                           get_cmd='ILVL?',
                           get_parser=int)
        self.add_parameter(name='signal_input',
                           label='Signal input',
                           get_cmd='IVMD?',
                           get_parser=self._get_input_config,
                           set_cmd='IVMD {}',
                           set_parser=self._set_input_config,
                           vals=Enum(*self._INPUT_SIGNAL_TO_N.keys()))
        self.add_parameter(name='input_range',
                           label='Input range',
                           units='V',
                           get_cmd='IRNG?',
                           set_cmd='IRNG {}',
                           val_mapping={1: 0,
                                        300e-3: 1,
                                        100e-3: 2,
                                        30e-3: 3,
                                        10e-3: 4})
        self.add_parameter(name='input_config',
                           label='Input configuration',
                           get_cmd='ISRC?',
                           set_cmd='ISRC {}',
                           val_mapping={'a': 0,
                                        'a-b': 1})
        self.add_parameter(name='input_shield',
                           label='Input shield',
                           get_cmd='IGND?',
                           set_cmd='IGND {}',
                           val_mapping={'float': 0,
                                        'ground': 1})
        self.add_parameter(name='input_gain',
                           label='Input gain',
                           units='ohm',
                           get_cmd='ICUR?',
                           set_cmd='ICUR {}',
                           val_mapping={1e6:   0,
                                        100e6: 1})
        self.add_parameter(name='adv_filter',
                           label='Advanced filter',
                           get_cmd='ADVFILT?',
                           set_cmd='ADVFILT {}',
                           val_mapping={'OFF': 0,
                                        'ON': 1})
        self.add_parameter(name='input_coupling',
                           label='Input coupling',
                           get_cmd='ICPL?',
                           set_cmd='ICPL {}',
                           val_mapping={'ac': 0, 'dc': 1})
        self.add_parameter(name='time_constant',
                           label='Time constant',
                           units='s',
                           get_cmd='OFLT?',
                           set_cmd='OFLT {}',
                           val_mapping={1e-6: 0, 3e-6: 1,
                                        10e-6: 2, 30e-6: 3,
                                        100e-6: 4, 300e-6: 5,
                                        1e-3: 6, 3e-3: 7,
                                        10e-3: 8, 30e-3: 9,
                                        100e-3: 10, 300e-3: 11,
                                        1: 12, 3: 13,
                                        10: 14, 30: 15,
                                        100: 16, 300: 17,
                                        1e3: 18, 3e3: 19,
                                        10e3: 20, 30e3: 21})
        # Auto functions
        self.add_function('auto_range', call_cmd='ARNG')
        self.add_function('auto_scale', call_cmd='ASCL')
        self.add_function('auto_phase', call_cmd='APHS')

        # Data transfer
        # first 4 parameters from a list of 16 below.
        self.add_parameter('X',
                           label='In-phase Magnitude',
                           get_cmd='OUTP? 0',
                           get_parser=float,
                           units='V')
        self.add_parameter('Y',
                           label='Out-phase Magnitude',
                           get_cmd='OUTP? 1',
                           get_parser=float,
                           units='V')
        self.add_parameter('R',
                           label='Magnitude',
                           get_cmd='OUTP? 2',
                           get_parser=float,
                           units='V')
        self.add_parameter('P',
                           label='Phase',
                           get_cmd='OUTP? 3',
                           get_parser=float,
                           units='deg')

        # CH1/CH2 Output Commands
        self.add_parameter('X_offset',
                           label='X offset ',
                           units='%',
                           get_cmd='COFP? 0',
                           set_cmd='COFP 0, {}',
                           get_parser=float,
                           vals=Numbers(min_value=-999.99, max_value=999.99))
        self.add_parameter('Y_offset',
                           label='Y offset',
                           units='%',
                           get_cmd='COFP? 1',
                           set_cmd='COFP 1, {}',
                           get_parser=float,
                           vals=Numbers(min_value=-999.99, max_value=999.99))
        self.add_parameter('R_offset',
                           label='R offset',
                           units='%',
                           get_cmd='COFP? 2',
                           set_cmd='COFP 2, {}',
                           get_parser=float,
                           vals=Numbers(min_value=-999.99, max_value=999.99))
        self.add_parameter('X_expand',
                           label='X expand multiplier',
                           get_cmd='CEXP? 0',
                           set_cmd='CEXP 0, {}',
                           val_mapping={'OFF': '0',
                                        'X10': '1',
                                        'X100': '2'})
        self.add_parameter('Y_expand',
                           label='Y expand multiplier',
                           get_cmd='CEXP? 1',
                           set_cmd='CEXP 1, {}',
                           val_mapping={'OFF': 0,
                                        'X10': 1,
                                        'X100': 2})
        self.add_parameter('R_expand',
                           label='R expand multiplier',
                           get_cmd='CEXP? 2',
                           set_cmd='CEXP 2, {}',
                           val_mapping={'OFF': 0,
                                        'X10': 1,
                                        'X100': 2})
        # Aux input/output
        for i in [0, 1, 2, 3]:
            self.add_parameter('aux_in{}'.format(i),
                               label='Aux input {}'.format(i),
                               get_cmd='OAUX? {}'.format(i),
                               get_parser=float,
                               units='V')

            self.add_parameter('aux_out{}'.format(i),
                               label='Aux output {}'.format(i),
                               get_cmd='AUXV? {}'.format(i),
                               get_parser=float,
                               set_cmd='AUXV {0}, {{}}'.format(i),
                               units='V')

        # Interface
        self.add_function('reset', call_cmd='*RST')

        self.add_function('disable_front_panel', call_cmd='OVRM 0')
        self.add_function('enable_front_panel', call_cmd='OVRM 1')

        self.input_config()
        self.connect_message()

    def _set_units(self, units):
        for param in [self.X, self.Y, self.R, self.sensitivity]:
            param.units = units

    def _get_input_config(self, s):
        mode = self._N_TO_INPUT_SIGNAL[int(s)]

        if mode == 'voltage':
            self.sensitivity.set_validator(self._VOLT_ENUM)
            self._set_units('V')
        else:
            self.sensitivity.set_validator(self._CURR_ENUM)
            self._set_units('A')

        return mode

    def _set_input_config(self, s):
        if s == 'voltage':
            self.sensitivity.set_validator(self._VOLT_ENUM)
            self._set_units('V')
        else:
            self.sensitivity.set_validator(self._CURR_ENUM)
            self._set_units('A')

        return self._INPUT_SIGNAL_TO_N[s]

    def _get_sensitivity(self, s):
        if self.signal_input() == 'voltage':
            return self._N_TO_VOLT[int(s)]
        else:
            return self._N_TO_CURR[int(s)]

    def _set_sensitivity(self, s):
        if self.signal_input() == 'voltage':
            return self._VOLT_TO_N[s]
        else:
            return self._CURR_TO_N[s]
