from functools import partial
from typing import Any, List, Sequence, Union

import numpy as np
from qcodes import VisaInstrument
from qcodes.utils.validators import Anything, Enum, Ints, MultiType, Numbers


def is_number(s: str) -> bool:
    """ Test whether a given string can be parsed as a float """
    try:
        float(s)
        return True
    except ValueError:
        return False


def clean_string(s: str) -> str:
    """ Clean string outputs of a VISA instrument for further parsing """
    # Remove surrounding whitespace and newline characters
    s = s.strip()

    # Remove surrounding quotes
    if (s[0] == s[-1]) and s.startswith(("'", '"')):
        s = s[1:-1]

    s = s.lower()

    return s


def parse_string_output(s: str) -> Union[float, str]:
    """ Parse an output of the VISA instrument into either text of a number """
    s = clean_string(s)

    # prevent float() from parsing 'infinity' into a float
    if s == 'infinity':
        return s

    # If it is a number; parse it
    if is_number(s):
        return float(s)

    return s


def parse_single_output(i: int, s: str) -> Union[float, str]:
    """ Used as a partial function to parse output i in string s """
    parts = clean_string(s).split(',')

    return parse_string_output(parts[i])


def parse_multiple_outputs(s: str) -> List[Union[float, str]]:
    """ Parse an output such as 'sin,1.5,0,2' and return a parsed array """
    parts = clean_string(s).split(',')

    return [parse_string_output(part) for part in parts]


class Rigol_DG4000(VisaInstrument):
    """
    Driver for the Rigol DG4000 series arbitrary waveform generator.

    This driver works for all four models (DG4202, DG4162, DG4102, DG4062).
    """
    def __init__(
            self,
            name: str,
            address: str,
            reset: bool = False,
            **kwargs: Any):
        super().__init__(name, address, terminator='\n', **kwargs)

        model = self.get_idn()['model']

        models = ['DG4202',  'DG4162',   'DG4102',   'DG4062']

        if model in models:
            i = models.index(model)

            sine_freq = [200e6, 160e6, 100e6, 60e6][i]
            square_freq = [60e6, 50e6, 40e6, 25e6][i]
            ramp_freq = [5e6, 4e6, 3e6, 1e6][i]
            pulse_freq = [50e6, 40e6, 25e6, 15e6][i]
            harmonic_freq = [100e6, 80e6, 50e6, 30e6][i]
            arb_freq = [50e6, 40e6, 25e6, 15e6][i]
        elif model is None:
            raise KeyError('Could not determine model')
        else:
            raise KeyError('Model code ' + model + ' is not recognized')

        on_off_map = {True: 'ON', False: 'OFF'}

        # Counter
        self.add_parameter('counter_attenuation',
                           get_cmd='COUN:ATT?',
                           set_cmd='COUN:ATT {}',
                           val_mapping={1: '1X', 10: '10X'})

        self.add_function('auto_counter', call_cmd='COUN:AUTO')

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
            self.add_parameter(f'counter_{param}',
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
            ch = f'ch{i}_'
            output = f'OUTP{i}:'
            source = f'SOUR{i}:'

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

            # Source Apply
            # TODO: Various parameters are limited by
            # impedance/freq/period/amplitude settings, this might be very hard
            # to implement in here
            self.add_function(ch + 'custom',
                              call_cmd=source + 'APPL:CUST '
                                                '{:.6e},{:.6e},{:.6e},{:.6e}',
                              args=[Numbers(1e-6, arb_freq),
                                    Numbers(), Numbers(), Numbers(0, 360)])

            self.add_function(ch + 'harmonic',
                              call_cmd=source + 'APPL:HARM '
                                                '{:.6e},{:.6e},{:.6e},{:.6e}',
                              args=[Numbers(1e-6, harmonic_freq),
                                    Numbers(), Numbers(), Numbers(0, 360)])

            self.add_function(ch + 'noise',
                              call_cmd=source + 'APPL:NOIS {:.6e},{:.6e}',
                              args=[Numbers(0, 10), Numbers()])

            self.add_function(ch + 'pulse',
                              call_cmd=source + 'APPL:PULS '
                                                '{:.6e},{:.6e},{:.6e},{:.6e}',
                              args=[Numbers(1e-6, pulse_freq),
                                    Numbers(), Numbers(), Numbers(0)])

            self.add_function(ch + 'ramp',
                              call_cmd=source + 'APPL:RAMP '
                                                '{:.6e},{:.6e},{:.6e},{:.6e}',
                              args=[Numbers(1e-6, ramp_freq),
                                    Numbers(), Numbers(), Numbers(0, 360)])

            self.add_function(ch + 'sinusoid',
                              call_cmd=source + 'APPL:SIN '
                                                '{:.6e},{:.6e},{:.6e},{:.6e}',
                              args=[Numbers(1e-6, sine_freq),
                                    Numbers(), Numbers(), Numbers(0, 360)])

            self.add_function(ch + 'square',
                              call_cmd=source + 'APPL:SQU '
                                                '{:.6e},{:.6e},{:.6e},{:.6e}',
                              args=[Numbers(1e-6, square_freq),
                                    Numbers(), Numbers(), Numbers(0, 360)])

            self.add_function(ch + 'user',
                              call_cmd=source + 'APPL:USER '
                                                '{:.6e},{:.6e},{:.6e},{:.6e}',
                              args=[Numbers(1e-6, arb_freq),
                                    Numbers(), Numbers(), Numbers(0, 360)])

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
            self.add_function(ch + 'set_harmonic_amplitude',
                              call_cmd=source + 'HARM:AMPL {},{:.6e}',
                              args=[Ints(2, 16), Numbers(0)])

            self.add_function(ch + 'get_harmonic_amplitude',
                              call_cmd=source + 'HARM:AMPL? {}',
                              args=[Ints(2, 16)],
                              return_parser=float)

            self.add_parameter(ch + 'harmonic_order',
                               get_cmd=source + 'HARM:ORDE?',
                               get_parser=int,
                               set_cmd=source + 'HARM:ORDE {}',
                               vals=Ints(2, 16))

            self.add_function(ch + 'set_harmonic_phase',
                              call_cmd=source + 'HARM:PHAS {},{:.6e}',
                              args=[Ints(2, 16), Numbers(0, 360)])

            self.add_function(ch + 'get_harmonic_phase',
                              call_cmd=source + 'HARM:PHAS? {}',
                              args=[Ints(2, 16)],
                              return_parser=float)

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

            self.add_function(ch + 'align_phase',
                              call_cmd=source + 'PHAS:INIT')

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
        self.add_function('beep', call_cmd='SYST:BEEP')

        self.add_parameter('beeper_enabled',
                           get_cmd='SYST:BEEP:STAT?',
                           set_cmd='SYST:BEEP:STAT {}',
                           val_mapping=on_off_map)

        self.add_function('copy_config_to_ch1', call_cmd='SYST:CSC CH2,CH1')
        self.add_function('copy_config_to_ch2', call_cmd='SYST:CSC CH1,CH2')

        self.add_function('copy_waveform_to_ch1', call_cmd='SYST:CWC CH2,CH1')
        self.add_function('copy_waveform_to_ch2', call_cmd='SYST:CWC CH1,CH2')

        self.add_function('get_error', call_cmd='SYST:ERR?', return_parser=str)

        self.add_parameter('keyboard_locked',
                           get_cmd='SYST:KLOCK?',
                           set_cmd='SYST:KLOCK {}',
                           val_mapping=on_off_map)

        self.add_parameter('startup_mode',
                           get_cmd='SYST:POWS?',
                           get_parser=str.lower,
                           set_cmd='SYST:POWS {}',
                           vals=Enum('user', 'auto'))

        system_states = Enum('default', 'user1', 'user2', 'user3',
                             'user4', 'user5', 'user6', 'user7',
                             'user8', 'user9', 'user10')

        self.add_function('preset',
                          call_cmd='SYST:PRES {}',
                          args=[system_states])

        self.add_function('restart', call_cmd='SYST:RESTART')

        self.add_parameter('reference_clock_source',
                           get_cmd='SYST:ROSC:SOUR?',
                           set_cmd='SYST:ROSC:SOUR {}',
                           val_mapping={'internal': 'INT', 'external': 'EXT'})

        self.add_function('shutdown', call_cmd='SYST:SHUTDOWN')

        self.add_parameter('scpi_version', get_cmd='SYST:VERS?')

        # Trace
        self.add_function('upload_data',
                          call_cmd=self._upload_data,
                          args=[Anything()])

        self.add_function('reset', call_cmd='*RST')

        if reset:
            self.reset()

        self.connect_message()

    def _upload_data(self, data: Union[Sequence[float], np.ndarray]) -> None:
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
