from qcodes import VisaInstrument
from qcodes.utils.validators import Numbers, Ints, Enum, MultiType, Anything

from functools import partial


def is_number(s):
    """ Test whether a given string can be parsed as a float """
    try:
        float(s)
        return True
    except ValueError:
        return False


def clean_string(s):
    """ Clean string outputs of a VISA instrument for further parsing """
    # Remove surrounding whitespace and newline characters
    s = s.strip()

    # Remove surrounding quotes
    if (s[0] == s[-1]) and s.startswith(("'", '"')):
        s = s[1:-1]

    s = s.lower()

    return s


def parse_string_output(s):
    """ Parse an output of the VISA instrument into either text of a number """
    s = clean_string(s)

    # prevent float() from parsing 'infinity' into a float
    if s == 'infinity':
        return s

    # If it is a number; parse it
    if is_number(s):
        return float(s)

    return s


def parse_single_output(i, s):
    """ Used as a partial function to parse output i in string s """
    parts = clean_string(s).split(',')

    return parse_string_output(parts[i])


def parse_multiple_outputs(s):
    """ Parse an output such as 'sin,1.5,0,2' and return a parsed array """
    parts = clean_string(s).split(',')

    return [parse_string_output(part) for part in parts]


class Rigol_DG4000(VisaInstrument):
    """
    Driver for the Rigol DG4000 series arbitrary waveform generator.

    This driver works for all four models (DG4202, DG4162, DG4102, DG4062).
    """
    def __init__(self, name, address, reset=False, **kwargs):
        super().__init__(name, address, terminator='\n', **kwargs)

        model = self.get_idn()['model']

        models = ['DG4202',  'DG4162',   'DG4102',   'DG4062']

        if model in models:
            i = models.index(model)

            self.sine_freq = [200e6, 160e6, 100e6, 60e6][i]
            self.square_freq = [60e6, 50e6, 40e6, 25e6][i]
            self.ramp_freq = [5e6, 4e6, 3e6, 1e6][i]
            self.pulse_freq = [50e6, 40e6, 25e6, 15e6][i]
            self.harmonic_freq = [100e6, 80e6, 50e6, 30e6][i]
            self.arb_freq = [50e6, 40e6, 25e6, 15e6][i]
        else:
            raise KeyError('Model code ' + model + ' is not recognized')

        on_off_map = {True: 'ON', False: 'OFF'}

        # Counter
        self.add_parameter('counter_attenuation',
                           get_cmd='COUN:ATT?',
                           set_cmd='COUN:ATT {}',
                           val_mapping={1: '1X', 10: '10X'})

        self.add_parameter('counter_coupling',
                           get_cmd='COUN:COUP?',
                           set_cmd='COUN:COUP {}',
                           vals=Enum('AC', 'DC'))

        self.add_parameter('counter_gate_time',
                           get_cmd='COUN:GATE?',
                           set_cmd='COUN:GATE {}',
                           unit='s',
                           val_mapping={
                               'auto': 'AUTO',
                               0.001:  'USER1',
                               0.01:   'USER2',
                               0.1:    'USER3',
                               1:      'USER4',
                               10:     'USER5',
                               '>10':  'USER6',
                           })

        self.add_parameter('counter_hf_reject_enabled',
                           get_cmd='COUN:HF?',
                           set_cmd='COUN:HF {}',
                           val_mapping=on_off_map)

        self.add_parameter('counter_impedance',
                           get_cmd='COUN:IMP?',
                           set_cmd='COUN:IMP {}',
                           unit='Ohm',
                           val_mapping={50: '50', 1e6: '1M'})

        self.add_parameter('counter_trigger_level',
                           get_cmd='COUN:LEVE?',
                           get_parser=float,
                           set_cmd='COUN:LEVE {}',
                           unit='V',
                           vals=Numbers(min_value=-2.5, max_value=2.5))

        self.add_parameter('counter_enabled',
                           get_cmd='COUN:STAT?',
                           set_cmd='COUN:STAT {}',
                           val_mapping=on_off_map)

        measure_params = ['frequency', 'period', 'duty_cycle',
                          'positive_width', 'negative_width']

        # TODO: Check units of outputs
        for i, param in enumerate(measure_params):
            self.add_parameter('counter_{}'.format(param),
                               get_cmd='COUN:MEAS?',
                               get_parser=partial(parse_single_output, i))

        self.add_parameter('counter_trigger_sensitivity',
                           get_cmd='COUN:SENS?',
                           get_parser=float,
                           set_cmd='COUN:SENS {}',
                           unit='%',
                           vals=Numbers(min_value=0, max_value=100))

        # Output and Source parameters for both channel 1 and 2
        for i in [1, 2]:
            ch = 'ch{}_'.format(i)
            output = 'OUTP{}:'.format(i)
            source = 'SOUR{}:'.format(i)

            self.add_parameter(ch + 'output_impedance',
                               get_cmd=output + 'IMP?',
                               get_parser=parse_string_output,
                               set_cmd=output + 'IMP {}',
                               unit='Ohm',
                               vals=MultiType(Numbers(min_value=1,
                                                      max_value=10e3),
                                              Enum('infinity',
                                                   'minimum',
                                                   'maximum')))

            self.add_parameter(ch + 'add_noise_scale',
                               get_cmd=output + 'NOIS:SCAL?',
                               get_parser=float,
                               set_cmd=output + 'NOIS:SCAL',
                               unit='%',
                               vals=Numbers(min_value=0, max_value=50))

            self.add_parameter(ch + 'add_noise_enabled',
                               get_cmd=output + 'NOIS?',
                               set_cmd=output + 'NOIS {}',
                               val_mapping=on_off_map)

            self.add_parameter(ch + 'output_polarity',
                               get_cmd=output + 'POL?',
                               set_cmd=output + 'POL {}',
                               val_mapping={'normal': 'NORM',
                                            'inverted': 'INV'})

            self.add_parameter(ch + 'output_enabled',
                               get_cmd=output + 'STAT?',
                               set_cmd=output + 'STAT {}',
                               val_mapping=on_off_map)

            self.add_parameter(ch + 'sync_polarity',
                               get_cmd=output + 'SYNC:POL?',
                               set_cmd=output + 'SYNC:POL {}',
                               val_mapping={'positive': 'POS',
                                            'negative': 'NEG'})

            self.add_parameter(ch + 'sync_enabled',
                               get_cmd=output + 'SYNC?',
                               set_cmd=output + 'SYNC {}',
                               val_mapping=on_off_map)

            self.add_parameter(ch + 'configuration',
                               get_cmd=source + 'APPL?',
                               get_parser=parse_multiple_outputs)

            # Source Burst
            self.add_parameter(ch + 'burst_mode',
                               get_cmd=source + 'BURS:MODE?',
                               set_cmd=source + 'BURS:MODE {}',
                               val_mapping={'triggered': 'TRIG',
                                            'gated': 'GAT',
                                            'infinity': 'INF'})

            self.add_parameter(ch + 'burst_cycles',
                               get_cmd=source + 'BURS:NCYC?',
                               get_parser=float,
                               set_cmd=source + 'BURS:NCYC {}',
                               vals=Ints(1, 1000000))

            self.add_parameter(ch + 'burst_period',
                               get_cmd=source + 'BURS:INT:PER?',
                               get_parser=float,
                               set_cmd=source + 'BURS:INT:PER {}',
                               unit='s',
                               vals=Numbers(1e-6))

            self.add_parameter(ch + 'burst_phase',
                               get_cmd=source + 'BURS:PHAS?',
                               get_parser=float,
                               set_cmd=source + 'BURS:PHAS {}',
                               unit='deg',
                               vals=Numbers(0, 360))

            self.add_parameter(ch + 'burst_trigger_edge',
                               get_cmd=source + 'BURS:TRIG:SLOP?',
                               set_cmd=source + 'BURS:TRIG:SLOP {}',
                               val_mapping={'positive': 'POS',
                                            'negative': 'NEG'})

            self.add_parameter(ch + 'burst_trigger_source',
                               get_cmd=source + 'BURS:TRIG:SOUR?',
                               set_cmd=source + 'BURS:TRIG:SOUR {}',
                               val_mapping={'internal': 'INT',
                                            'external': 'EXT',
                                            'manual': 'MAN'})

            self.add_parameter(ch + 'burst_trigger_out',
                               get_cmd=source + 'BURS:TRIG:TRIGO?',
                               set_cmd=source + 'BURS:TRIG:TRIGO {}',
                               val_mapping={'off': 'OFF',
                                            'positive': 'POS',
                                            'negative': 'NEG'})

            # Source Frequency
            # TODO: The upper bounds of these parameters also depend on the
            # current waveform
            self.add_parameter(ch + 'frequency_center',
                               get_cmd=source + 'FREQ:CENT?',
                               get_parser=float,
                               set_cmd=source + 'FREQ:CENT {}',
                               unit='Hz',
                               vals=Numbers(1e-6))

            self.add_parameter(ch + 'frequency',
                               get_cmd=source + 'FREQ?',
                               get_parser=float,
                               set_cmd=source + 'FREQ {}',
                               unit='Hz',
                               vals=Numbers(1e-6))

            self.add_parameter(ch + 'frequency_start',
                               get_cmd=source + 'FREQ:STAR?',
                               get_parser=float,
                               set_cmd=source + 'FREQ:STAR {}',
                               unit='Hz',
                               vals=Numbers(1e-6))

            self.add_parameter(ch + 'frequency_stop',
                               get_cmd=source + 'FREQ:STOP?',
                               get_parser=float,
                               set_cmd=source + 'FREQ:STOP {}',
                               unit='Hz',
                               vals=Numbers(1e-6))

            # Source Function
            self.add_parameter(ch + 'ramp_symmetry',
                               get_cmd=source + 'FUNC:RAMP:SYMM?',
                               get_parser=float,
                               set_cmd=source + 'FUNC:RAMP:SYMM {}',
                               unit='%',
                               vals=Numbers(0, 100))

            self.add_parameter(ch + 'square_duty_cycle',
                               get_cmd=source + 'FUNC:SQU:DCYC?',
                               get_parser=float,
                               set_cmd=source + 'FUNC:SQU:DCYC {}',
                               unit='%',
                               vals=Numbers(20, 80))

            # Source Harmonic

            self.add_parameter(ch + 'harmonic_order',
                               get_cmd=source + 'HARM:ORDE?',
                               get_parser=int,
                               set_cmd=source + 'HARM:ORDE {}',
                               vals=Ints(2, 16))

            self.add_parameter(ch + 'harmonic_type',
                               get_cmd=source + 'HARM:TYP?',
                               get_parser=str.lower,
                               set_cmd=source + 'HARM:TYP {}',
                               vals=Enum('even', 'odd', 'all', 'user'))

            # Source Marker
            self.add_parameter(ch + 'marker_frequency',
                               get_cmd=source + 'MARK:FREQ?',
                               get_parser=float,
                               set_cmd=source + 'HMARK:FREQ {}',
                               unit='Hz',
                               vals=Numbers(1e-6))

            self.add_parameter(ch + 'marker_enabled',
                               get_cmd=source + 'MARK?',
                               set_cmd=source + 'MARK {}',
                               val_mapping=on_off_map)

            # Source Modulation (not implemented yet)

            # Source Period (not implemented yet)

            # Source Phase
            self.add_parameter(ch + 'phase',
                               get_cmd=source + 'PHAS?',
                               get_parser=float,
                               set_cmd=source + 'PHAS {}',
                               unit='deg',
                               vals=Numbers(0, 360))

            # Source Pulse
            self.add_parameter(ch + 'pulse_duty_cycle',
                               get_cmd=source + 'PULS:DCYC?',
                               get_parser=float,
                               set_cmd=source + 'PULS:DCYC {}',
                               unit='%',
                               vals=Numbers(0, 100))

            self.add_parameter(ch + 'pulse_delay',
                               get_cmd=source + 'PULS:DEL?',
                               get_parser=float,
                               set_cmd=source + 'PULS:DEL {}',
                               unit='s',
                               vals=Numbers(0))

            self.add_parameter(ch + 'pulse_hold',
                               get_cmd=source + 'PULS:HOLD?',
                               set_cmd=source + 'PULS:HOLD {}',
                               unit='s',
                               val_mapping={'width': 'WIDT', 'duty': 'DUTY'})

            self.add_parameter(ch + 'pulse_leading_edge',
                               get_cmd=source + 'PULS:TRAN:LEAD?',
                               get_parser=float,
                               set_cmd=source + 'PULS:TRAN:LEAD {}',
                               unit='s',
                               vals=Numbers(0))

            self.add_parameter(ch + 'pulse_trailing_edge',
                               get_cmd=source + 'PULS:TRAN:TRA?',
                               get_parser=float,
                               set_cmd=source + 'PULS:TRAN:TRA {}',
                               unit='s',
                               vals=Numbers(0))

            self.add_parameter(ch + 'pulse_width',
                               get_cmd=source + 'PULS:WIDT?',
                               get_parser=float,
                               set_cmd=source + 'PULS:WIDT {}',
                               unit='s',
                               vals=Numbers(0))

            # Source Sweep
            self.add_parameter(ch + 'sweep_hold_start',
                               get_cmd=source + 'SWE:HTIM:STAR?',
                               get_parser=float,
                               set_cmd=source + 'SWE:HTIM:STAR {}',
                               unit='s',
                               vals=Numbers(0, 300))

            self.add_parameter(ch + 'sweep_hold_stop',
                               get_cmd=source + 'SWE:HTIM:STOP?',
                               get_parser=float,
                               set_cmd=source + 'SWE:HTIM:STOP {}',
                               unit='s',
                               vals=Numbers(0, 300))

            self.add_parameter(ch + 'sweep_return_time',
                               get_cmd=source + 'SWE:RTIM?',
                               get_parser=float,
                               set_cmd=source + 'SWE:RTIM {}',
                               unit='s',
                               vals=Numbers(0, 300))

            self.add_parameter(ch + 'sweep_spacing',
                               get_cmd=source + 'SWE:SPAC?',
                               set_cmd=source + 'SWE:SPAC {}',
                               val_mapping={'linear': 'LIN',
                                            'logarithmic': 'LOG',
                                            'step': 'STE'})

            self.add_parameter(ch + 'sweep_enabled',
                               get_cmd=source + 'SWE:STAT?',
                               set_cmd=source + 'SWE:STAT {}',
                               val_mapping=on_off_map)

            self.add_parameter(ch + 'sweep_step',
                               get_cmd=source + 'SWE:STEP?',
                               get_parser=int,
                               set_cmd=source + 'SWE:STEP {}',
                               vals=Ints(2, 2048))

            self.add_parameter(ch + 'sweep_time',
                               get_cmd=source + 'SWE:TIME?',
                               get_parser=float,
                               set_cmd=source + 'SWE:TIME {}',
                               unit='s',
                               vals=Numbers(1e-3, 300))

            # Source Voltage
            self.add_parameter(ch + 'amplitude',
                               get_cmd=source + 'VOLT?',
                               get_parser=float,
                               set_cmd=source + 'VOLT {}',
                               unit='V',
                               vals=Numbers())

            self.add_parameter(ch + 'offset',
                               get_cmd=source + 'VOLT:OFFS?',
                               get_parser=float,
                               set_cmd=source + 'VOLT:OFFS {}',
                               unit='V',
                               vals=Numbers())

            self.add_parameter(ch + 'unit',
                               get_cmd=source + 'VOLT:UNIT?',
                               get_parser=str.lower,
                               set_cmd=source + 'VOLT:UNIT {}',
                               vals=Enum('vpp', 'vrms', 'dbm'))

        # System
        self.add_parameter('beeper_enabled',
                           get_cmd='SYST:BEEP:STAT?',
                           set_cmd='SYST:BEEP:STAT {}',
                           val_mapping=on_off_map)

        self.add_parameter('keyboard_locked',
                           get_cmd='SYST:KLOCK?',
                           set_cmd='SYST:KLOCK {}',
                           val_mapping=on_off_map)

        self.add_parameter('startup_mode',
                           get_cmd='SYST:POWS?',
                           get_parser=str.lower,
                           set_cmd='SYST:POWS {}',
                           vals=Enum('user', 'auto'))

        self.add_parameter('reference_clock_source',
                           get_cmd='SYST:ROSC:SOUR?',
                           set_cmd='SYST:ROSC:SOUR {}',
                           val_mapping={'internal': 'INT', 'external': 'EXT'})

        self.add_parameter('scpi_version', get_cmd='SYST:VERS?')

        if reset:
            self.reset()

        self.connect_message()

    def get_error(self) -> str:
        """
        Query the error event queue
        """
        resp = self.ask('SYST:ERR?')
        return str(resp)

    def preset(self, system_state: str) -> None:
        """
        Restore the system to its default state or to a user-defined state

        Args:
            system_state: The state to which the system shall be restored
        """
        valid_states = Enum('default', 'user1', 'user2', 'user3',
                            'user4', 'user5', 'user6', 'user7',
                            'user8', 'user9', 'user10')
        valid_states.validate(system_state)
        self.write(f"SYST:PRES {system_state}")

    def copy_waveform_to_ch1(self) -> None:
        """
        Copy the arbitrary waveform data (not include the waveform parameters)
        of CH2 to CH1
        """
        self.write('SYST:CWC CH2,CH1')

    def copy_waveform_to_ch2(self) -> None:
        """
        Copy the arbitrary waveform data (not include the waveform parameters)
        of CH1 to CH2
        """
        self.write('SYST:CWC CH1,CH2')

    def copy_config_to_ch1(self) -> None:
        """
        Copy the configuration state of CH2 to CH1.
        """
        self.write('SYST:CSC CH2,CH1')

    def copy_config_to_ch2(self) -> None:
        """
        Copy the configuration state of CH1 to CH2.
        """
        self.write('SYST:CSC CH1,CH2')

    def beep(self) -> None:
        """
        The beeper generates a beep immediately (if the beeper is enabled)
        """
        self.write("SYST:BEEP")

    def restart(self) -> None:
        """
        Restart the instrument
        """
        self.write("SYST:RESTART")

    def shutdown(self) -> None:
        """
        Shut down the instrument
        """
        self.write("SYST:SHUTDOWN")

    def reset(self) -> None:
        """
        Restore the instrument to its default states
        """
        self.write('*RST')

    def auto_counter(self) -> None:
        """
        Send this command and the instrument will set the gate time of the
        counter automatically.
        """
        self.write('COUN:AUTO')

    def ch1_set_harmonic_amplitude(self, order: int, amplitude: float) -> None:
        """
        Set the amplitude of the specified order of harmonic
        """
        Ints(2, 16).validate(order)
        Numbers(0).validate(amplitude)
        self.write(f"SOUR1:HARM:AMPL {order},{amplitude:.6e}")

    def ch2_set_harmonic_amplitude(self, order: int, amplitude: float) -> None:
        """
        Set the amplitude of the specified order of harmonic
        """
        Ints(2, 16).validate(order)
        Numbers(0).validate(amplitude)
        self.write(f"SOUR2:HARM:AMPL {order},{amplitude:.6e}")

    def ch1_get_harmonic_amplitude(self, order: int) -> float:
        """
        Query the amplitude of the specified order of harmonic
        """
        Ints(2, 16).validate(order)
        resp = self.ask(f'SOUR1:HARM:AMPL? {order}')
        return float(resp)

    def ch2_get_harmonic_amplitude(self, order: int) -> float:
        """
        Query the amplitude of the specified order of harmonic
        """
        Ints(2, 16).validate(order)
        resp = self.ask(f'SOUR2:HARM:AMPL? {order}')
        return float(resp)

    def ch1_set_harmonic_phase(self, order: int, phase: float) -> None:
        """
        Set the phase of the specified order of harmonic
        """
        Ints(2, 16).validate(order)
        Numbers(0, 360).validate(phase)
        self.write(f"SOUR1:HARM:PHAS {order},{phase:.6e}")

    def ch2_set_harmonic_phase(self, order: int, phase: float) -> None:
        """
        Set the phase of the specified order of harmonic
        """
        Ints(2, 16).validate(order)
        Numbers(0, 360).validate(phase)
        self.write(f"SOUR2:HARM:PHAS {order},{phase:.6e}")

    def ch1_get_harmonic_phase(self, order: int) -> float:
        """
        Query the phase of the specified order of harmonic
        """
        Ints(2, 16).validate(order)
        resp = self.ask(f'SOUR1:HARM:PHAS? {order}')
        return float(resp)

    def ch2_get_harmonic_phase(self, order: int) -> float:
        """
        Query the phase of the specified order of harmonic
        """
        Ints(2, 16).validate(order)
        resp = self.ask(f'SOUR2:HARM:PHAS? {order}')
        return float(resp)

    # Source Apply
    # TODO: Various parameters are limited by
    # impedance/freq/period/amplitude settings, that ought to be implemented

    def _source_apply(self, source: int, form: str,
                      freq: float, amplitude: float,
                      offset: float, phase: float, max_freq: float) -> None:
        """
        Helper function to apply a waveform to the output
        """
        validators = [Numbers(1e-6, max_freq), Numbers(), Numbers(),
                      Numbers(0, 360)]
        values = [freq, amplitude, offset, phase]

        for vd, vl in zip(validators, values):
            vd.validate(vl)

        known_forms = ('CUST', 'HARM', 'RAMP', 'SIN', 'SQU', 'USER', 'PULS')

        if form not in known_forms:
            raise ValueError(f'Unknown form: {form}. Must be one of '
                             f'{known_forms}')

        cmd = (f'SOUR{source}:APPL:{form} {freq:.6e},{amplitude:.6e},'
               f'{offset:.6e},{phase:.6e}')

        self.write(cmd)

    def _source_apply_noise(self, source: int, amplitude: float,
                            offset: float) -> None:
        """
        Helper function for app;y noise
        """
        validators = [Numbers(0, 10), Numbers()]
        values = [amplitude, offset]
        for vd, vl in zip(validators, values):
            vd.validate(vl)

        cmd = (f'SOUR{source}:APPL:NOISE {amplitude:.6e},'
               f'{offset:.6e}')
        self.write(cmd)

    def ch1_custom(self, frequency: float, amplitude: float, offset: float,
                   phase: float) -> None:
        """
        Output the user-defined waveform with the specified parameters
        (frequency, amplitude, DC offset and start phase).
        """
        self._source_apply(source=1, form='CUST',
                           freq=frequency, amplitude=amplitude,
                           offset=offset, phase=phase,
                           max_freq=self.arb_freq)

    def ch2_custom(self, frequency: float, amplitude: float, offset: float,
                   phase: float) -> None:
        """
        Output the user-defined waveform with the specified parameters
        (frequency, amplitude, DC offset and start phase).
        """
        self._source_apply(source=2, form='CUST',
                           freq=frequency, amplitude=amplitude,
                           offset=offset, phase=phase,
                           max_freq=self.arb_freq)

    def ch1_harmonic(self, frequency: float, amplitude: float, offset: float,
                     phase: float) -> None:
        """
        Output a harmonic with specified frequency, amplitude, DC offset and
        start phase.
        """
        self._source_apply(source=1, form='HARM',
                           freq=frequency, amplitude=amplitude,
                           offset=offset, phase=phase,
                           max_freq=self.harmonic_freq)

    def ch2_harmonic(self, frequency: float, amplitude: float, offset: float,
                     phase: float) -> None:
        """
        Output a harmonic with specified frequency, amplitude, DC offset and
        start phase.
        """
        self._source_apply(source=2, form='HARM',
                           freq=frequency, amplitude=amplitude,
                           offset=offset, phase=phase,
                           max_freq=self.harmonic_freq)

    def ch1_ramp(self, frequency: float, amplitude: float, offset: float,
                 phase: float) -> None:
        """
        Output a ramp with specified frequency, amplitude, DC offset and start
        phase.
        """
        self._source_apply(source=1, form='RAMP',
                           freq=frequency, amplitude=amplitude,
                           offset=offset, phase=phase,
                           max_freq=self.ramp_freq)

    def ch2_ramp(self, frequency: float, amplitude: float, offset: float,
                    phase: float) -> None:
        """
        Output a ramp with specified frequency, amplitude, DC offset and start
        phase.
        """
        self._source_apply(source=2, form='RAMP',
                           freq=frequency, amplitude=amplitude,
                           offset=offset, phase=phase,
                           max_freq=self.ramp_freq)


    def ch1_sinusoid(self, frequency: float, amplitude: float, offset: float,
                    phase: float) -> None:
        """
        Output a sine waveform with specified frequency, amplitude, DC offset
        and start phase.
        """
        self._source_apply(source=1, form='SIN',
                           freq=frequency, amplitude=amplitude,
                           offset=offset, phase=phase,
                           max_freq=self.sine_freq)

    def ch2_sinusoid(self, frequency: float, amplitude: float, offset: float,
                    phase: float) -> None:
        """
        Output a sine waveform with specified frequency, amplitude, DC offset
        and start phase.
        """
        self._source_apply(source=2, form='SIN',
                           freq=frequency, amplitude=amplitude,
                           offset=offset, phase=phase,
                           max_freq=self.sine_freq)

    def ch1_square(self, frequency: float, amplitude: float, offset: float,
                    phase: float) -> None:
        """
        Output a square waveform with specified frequency, amplitude, DC offset
        and start phase.
        """
        self._source_apply(source=1, form='SQU',
                           freq=frequency, amplitude=amplitude,
                           offset=offset, phase=phase,
                           max_freq=self.square_freq)

    def ch2_square(self, frequency: float, amplitude: float, offset: float,
                    phase: float) -> None:
        """
        Output a square waveform with specified frequency, amplitude, DC offset
        and start phase.
        """
        self._source_apply(source=2, form='SQU',
                           freq=frequency, amplitude=amplitude,
                           offset=offset, phase=phase,
                           max_freq=self.square_freq)

    def ch1_user(self, frequency: float, amplitude: float, offset: float,
                 phase: float) -> None:
        """
        Output an arbitrary waveform with specified frequency, amplitude, DC
        offset and start phase.
        """
        self._source_apply(source=1, form='USER',
                           freq=frequency, amplitude=amplitude,
                           offset=offset, phase=phase,
                           max_freq=self.arb_freq)

    def ch2_user(self, frequency: float, amplitude: float, offset: float,
                 phase: float) -> None:
        """
        Output an arbitrary waveform with specified frequency, amplitude, DC
        offset and start phase.
        """
        self._source_apply(source=2, form='USER',
                           freq=frequency, amplitude=amplitude,
                           offset=offset, phase=phase,
                           max_freq=self.arb_freq)

    def ch1_noise(self, amplitude: float, offset: float) -> None:
        """
        Output a noise with specified amplitude and DC offset.
        """
        self._source_apply_noise(source=1, amplitude=amplitude,
                                 offset=offset)

    def ch2_noise(self, amplitude: float, offset: float) -> None:
        """
        Output a noise with specified amplitude and DC offset.
        """
        self._source_apply_noise(source=2, amplitude=amplitude,
                                 offset=offset)

    def ch1_pulse(self, frequency: float, amplitude: float, offset: float,
                  delay: float) -> None:
        """
        Output a pulse with specified frequency, amplitude, DC offset and
        delay.
        """
        # this function takes a delay rather than a phase, but the same SCPI
        # command-builder can be used
        self._source_apply(source=1, form='PULS',
                           freq=frequency, amplitude=amplitude,
                           offset=offset, phase=delay,
                           max_freq=self.arb_freq)

    def ch2_pulse(self, frequency: float, amplitude: float, offset: float,
                  delay: float) -> None:
        """
        Output a pulse with specified frequency, amplitude, DC offset and
        delay.
        """
        # this function takes a delay rather than a phase, but the same SCPI
        # command-builder can be used
        self._source_apply(source=2, form='PULS',
                           freq=frequency, amplitude=amplitude,
                           offset=offset, phase=delay,
                           max_freq=self.arb_freq)

    def ch1_align_phase(self) -> None:
        """
        Execute align phase

        This command is invalid when any of the two channels is in modulation
        mode
        """
        self.write("SOUR1:PHAS:INIT")

    def ch2_align_phase(self) -> None:
        """
        Execute align phase

        This command is invalid when any of the two channels is in modulation
        mode
        """
        self.write("SOUR2:PHAS:INIT")

    def upload_data(self, data):
        """
        Upload data to the AWG memory.

        data: list, tuple or numpy array containing the datapoints
        """
        if 1 <= len(data) <= 16384:
            # Convert the input to a comma-separated string
            string = ','.join(format(f, '.9f') for f in data)

            self.write('DATA VOLATILE,' + string)
        else:
            raise Exception('Data length of ' + str(len(data)) +
                            ' is not in the range of 1 to 16384')
