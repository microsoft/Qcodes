from qcodes import VisaInstrument
from qcodes.utils.validators import Numbers, Ints, Enum, MultiType

class SR830(VisaInstrument):
    """
    This is the qcodes driver for the Stanford Research Systems SR830
    Lock-in Amplifier
    """

    _VOLT_TO_N = {2e-9:    0, 5e-9:    1, 10e-9:  2,
                  20e-9:   3, 50e-9:   4, 100e-9: 5,
                  200e-9:  6, 500e-9:  7, 1e-6:   8,
                  2e-6:    9, 5e-6:   10, 10e-6:  11,
                  20e-6:  12, 50e-6:  13, 100e-6: 14,
                  200e-6: 15, 500e-6: 16, 1e-3:   17,
                  2e-3:   18, 5e-3:   19, 10e-3:  20,
                  20e-3:  21, 50e-3:  22, 100e-3: 23,
                  200e-3: 24, 500e-3: 25, 1:      26}
    _N_TO_VOLT = {v: k for k, v in _VOLT_TO_N.items()}

    _CURR_TO_N = {2e-15:    0, 5e-15:    1, 10e-15:  2,
                  20e-15:   3, 50e-15:   4, 100e-15: 5,
                  200e-15:  6, 500e-15:  7, 1e-12:   8,
                  2e-12:    9, 5e-12:   10, 10e-12:  11,
                  20e-12:  12, 50e-12:  13, 100e-12: 14,
                  200e-12: 15, 500e-12: 16, 1e-9:    17,
                  2e-9:    18, 5e-9:    19, 10e-9:   20,
                  20e-9:   21, 50e-9:   22, 100e-9:  23,
                  200e-9:  24, 500e-9:  25, 1e-6:    26}
    _N_TO_CURR = {v: k for k, v in _CURR_TO_N.items()}

    _VOLT_ENUM = Enum(*_VOLT_TO_N.keys())
    _CURR_ENUM = Enum(*_CURR_TO_N.keys())

    _INPUT_CONFIG_TO_N = {
        'a': 0,
        'a-b': 1,
        'I 1M': 2,
        'I 100M': 3,
    }

    _N_TO_INPUT_CONFIG = {v: k for k, v in _INPUT_CONFIG_TO_N.items()}

    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, **kwargs)

        # Reference and phase
        self.add_parameter('phase',
                           label='Phase',
                           get_cmd='PHAS?',
                           get_parser=float,
                           set_cmd='PHAS {:.2f}',
                           units='deg',
                           vals=Numbers(min_value=-360, max_value=729.99))

        self.add_parameter('reference_source',
                           label='Reference source',
                           get_cmd='FMOD?',
                           set_cmd='FMOD {}',
                           val_mapping={
                               'external': 0,
                               'internal': 1,
                           })

        self.add_parameter('frequency',
                           label='Frequency',
                           get_cmd='FREQ?',
                           get_parser=float,
                           set_cmd='FREQ {:.4f}',
                           units='Hz',
                           vals=Numbers(min_value=1e-3, max_value=102e3))

        self.add_parameter('ext_trigger',
                           label='External trigger',
                           get_cmd='RSLP?',
                           set_cmd='RSLP {}',
                           val_mapping={
                               'sine': 0,
                               'TTL rising': 1,
                               'TTL falling': 2,
                           })

        self.add_parameter('harmonic',
                           label='Harmonic',
                           get_cmd='HARM?',
                           get_parser=int,
                           set_cmd='HARM {:d}',
                           vals=Ints(min_value=1, max_value=19999))

        self.add_parameter('amplitude',
                           label='Amplitude',
                           get_cmd='SLVL?',
                           get_parser=float,
                           set_cmd='SLVL {:.3f}',
                           units='V',
                           vals=Numbers(min_value=0.004, max_value=5.000))

        # Input and filter
        self.add_parameter('input_config',
                           label='Input configuration',
                           get_cmd='ISRC?',
                           get_parser=self._get_input_config,
                           set_cmd='ISRC {}',
                           set_parser=self._set_input_config,
                           vals=Enum(*self._INPUT_CONFIG_TO_N.keys()))

        self.add_parameter('input_shield',
                           label='Input shield',
                           get_cmd='IGND?',
                           set_cmd='IGND {}',
                           val_mapping={
                               'float': 0,
                               'ground': 1,
                           })

        self.add_parameter('input_coupling',
                           label='Input coupling',
                           get_cmd='ICPL?',
                           set_cmd='ICPL {}',
                           val_mapping={
                               'AC': 0,
                               'DC': 1,
                           })

        self.add_parameter('notch_filter',
                           label='Notch filter',
                           get_cmd='ILIN?',
                           set_cmd='ILIN {}',
                           val_mapping={
                               'off': 0,
                               'line in': 1,
                               '2x line in': 2,
                               'both': 3,
                           })

        # Gain and time constant
        self.add_parameter(name='sensitivity',
                           label='Sensitivity',
                           get_cmd='SENS?',
                           set_cmd='SENS {:d}',
                           get_parser=self._get_sensitivity,
                           set_parser=self._set_sensitivity
                           )

        self.add_parameter('reserve',
                           label='Reserve',
                           get_cmd='RMOD?',
                           set_cmd='RMOD {}',
                           val_mapping={
                               'high': 0,
                               'normal': 1,
                               'low noise': 2,
                           })

        self.add_parameter('time_constant',
                           label='Time constant',
                           get_cmd='OFLT?',
                           set_cmd='OFLT {}',
                           units='s',
                           val_mapping={
                               10e-6:  0, 30e-6:  1,
                               100e-6: 2, 300e-6: 3,
                               1e-3:   4, 3e-3:   5,
                               10e-3:  6, 30e-3:  7,
                               100e-3: 8, 300e-3: 9,
                               1:     10, 3:     11,
                               10:    12, 30:    13,
                               100:   14, 300:   15,
                               1e3:   16, 3e3:   17,
                               10e3:  18, 30e3:  19,
                           })

        self.add_parameter('filter_slope',
                           label='Filter slope',
                           get_cmd='OFSL?',
                           set_cmd='OFSL {}',
                           units='dB/oct',
                           val_mapping={
                               6: 0,
                               12: 1,
                               18: 2,
                               24: 3,
                           })

        self.add_parameter('sync_filter',
                           label='Sync filter',
                           get_cmd='SYNC?',
                           set_cmd='SYNC {}',
                           val_mapping={
                               'off': 0,
                               'on': 1,
                           })

        def parse_offset_get(s):
            parts = s.split(',')

            return float(parts[0]), int(parts[1])

        # TODO: Parameters that can be set with multiple arguments
        # For the OEXP command for example two arguments are needed
        self.add_parameter('X_offset',
                           get_cmd='OEXP? 1',
                           get_parser=parse_offset_get)

        self.add_parameter('Y_offset',
                           get_cmd='OEXP? 2',
                           get_parser=parse_offset_get)

        self.add_parameter('R_offset',
                           get_cmd='OEXP? 3',
                           get_parser=parse_offset_get)

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

        # Setup
        self.add_parameter('output_interface',
                           label='Output interface',
                           get_cmd='OUTX?',
                           set_cmd='OUTX {}',
                           val_mapping={
                               'RS232': '0\n',
                               'GPIB': '1\n',
                           })

        # Auto functions
        self.add_function('auto_gain', call_cmd='AGAN')
        self.add_function('auto_reserve', call_cmd='ARSV')
        self.add_function('auto_phase', call_cmd='APHS')
        self.add_function('auto_offset', call_cmd='AOFF {0}', args=[Enum(1, 2, 3, 4)])

        # Data transfer
        self.add_parameter('X',
                           get_cmd='OUTP? 1',
                           get_parser=float,
                           units='V')

        self.add_parameter('Y',
                           get_cmd='OUTP? 2',
                           get_parser=float,
                           units='V')

        self.add_parameter('R',
                           get_cmd='OUTP? 3',
                           get_parser=float,
                           units='V')

        self.add_parameter('P',
                           get_cmd='OUTP? 4',
                           get_parser=float,
                           units='deg')

        # Interface
        self.add_function('reset', call_cmd='*RST')

        self.add_function('disable_front_panel', call_cmd='OVRM 0')
        self.add_function('enable_front_panel', call_cmd='OVRM 1')

        # Initialize the proper units of the outputs and sensitivities
        self.input_config()

        self.connect_message()

    def _set_units(self, units):
        # TODO:
        # make a public parameter function that allows to change the units
        for param in [self.X, self.Y, self.R, self.sensitivity]:
            param.units = units

    def _get_input_config(self, s):
        mode = self._N_TO_INPUT_CONFIG[int(s)]

        if mode in ['a', 'a-b']:
            self.sensitivity.set_validator(self._VOLT_ENUM)
            self._set_units('V')
        else:
            self.sensitivity.set_validator(self._CURR_ENUM)
            self._set_units('A')

        return mode

    def _set_input_config(self, s):
        if s in ['a', 'a-b']:
            self.sensitivity.set_validator(self._VOLT_ENUM)
            self._set_units('V')
        else:
            self.sensitivity.set_validator(self._CURR_ENUM)
            self._set_units('A')

        return self._INPUT_CONFIG_TO_N[s]

    def _get_sensitivity(self, s):
        if self.input_config() in ['a', 'a-b']:
            return self._N_TO_VOLT[int(s)]
        else:
            return self._N_TO_CURR[int(s)]

    def _set_sensitivity(self, s):
        if self.input_config() in ['a', 'a-b']:
            return self._VOLT_TO_N[s]
        else:
            return self._CURR_TO_N[s]
