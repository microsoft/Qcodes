from qcodes import VisaInstrument
from qcodes.utils.validators import Numbers, Ints, Enum
from functools import partial

def boolean_parser(command, instrument_response):
    if instrument_response.startswith(command):
        return bool(instrument_response[2])
    else:
        raise ValueError(
            'Instrument responded with wrong command: {0}'.format(
                instrument_response))

def freq_parser(command, instrument_response):
    return float(instrument_response.lstrip(command).rstrip('HZ'))

def amplitude_parser(command, instrument_response):
    amp = float(instrument_response.lstrip(command)[:-2])
    if instrument_response[-2:] == 'VO':
        return amp
    elif instrument_response[-2:] == 'MV':
        return amp*1000

def offset_parser(instrument_response):
    offset = float(instrument_response[2:-2])
    if instrument_response[-2:] == 'VO':
        return offset
    elif instrument_response[-2:] == 'MV':
        return offset*1000

def int_parser(command, instrument_response):
    return int(instrument_response.lstrip(command))

class HP3325B(VisaInstrument):
    """
    This is the code for Hewlett Packard 3325B synthesizer
    """

    def __init__(self, name, address, reset=False,  **kwargs):
        super().__init__(name, address,  terminator='\n', **kwargs)
        self.visa_handle.baud_rate = 4800 # switch 3 and 4 down
        self.visa_handle.read_termination = '\r\n'

        options = self.visa_handle.query('OPT?')
        if options.lstrip('OPT')[0] == '1':
            self._instrument_options.append('oven')
        if options.lstrip('OPT')[2] == '1':
            self._instrument_options.append('high voltage')

        self.add_parameter(name='frequency',
                           label='Frequency',
                           unit='Hz',
                           get_cmd='IFR',
                           set_cmd='FR{:8.3f}HZ',
                           get_parser=partial(freq_parser, 'FR'),
                           vals=Numbers(min_value=0.0e6,
                                        max_value=60.999999e6),
                           post_delay=0.1)
        self.add_parameter(name='amplitude',
                           label='Amplitude',
                           unit='V',
                           get_cmd='IAM',
                           set_cmd='AM{:5.6f}VO',
                           get_parser=partial(amplitude_parser, 'AM'),
                           vals=Numbers(min_value=0.001, 
                                         max_value=40.0),
                           post_delay=0.1)
        self.add_parameter(name='offset',
                           label='Offset',
                           unit='V',
                           get_cmd='IOF',
                           set_cmd='OF{:5.6f}VO',
                           get_parser=offset_parser,
                           vals=Numbers(min_value=-5.0,
                                        max_value=5.0),
                           post_delay=0.1)
        self.add_parameter(name='connected_output',
                           label='Connected output',
                           get_cmd='IRF',
                           val_mapping={'front' : 'RF1',
                                        'rear'  : 'RF2'})
        self.add_parameter(name='output_function',
                           label='Output function',
                           get_cmd='IFU',
                           set_cmd='FU{0:d}',
                           get_parser=partial(int_parser, 'FU'),
                           val_mapping={'DC' : 0,
                                        'sine' : 1,
                                        'square' : 2,
                                        'triangle' : 3,
                                        'positive ramp' : 4,
                                        'negative ramp' : 5})

        # Modulation
        self.add_parameter(name='modulation_source_frequency',
                           label='Modulation frequency',
                           get_cmd='MOFR?',
                           set_cmd='MOFR{:8.6f}HZ',
                           get_parser=partial(freq_parser, 'MOFR'),
                           )
        self.add_parameter(name='modulation_source_amplitude',
                           label='Modulation amplitude',
                           get_cmd='MOAM?',
                           set_cmd='MOAM{:5.6f}VO',
                           get_parser=partial(amplitude_parser, 'MOAM'),
                           )
        self.add_parameter(name='amplitude_modulation_status',
                           label='Amplitude modulation status',
                           get_cmd='IMA',
#                           set_cmd='MA{0}',
                           get_parser=partial(boolean_parser, 'MA'),
                            )

        #resets amplitude and offset each time user connects
        self.add_function('reset', call_cmd='*RST')
