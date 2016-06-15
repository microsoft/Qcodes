from qcodes import VisaInstrument
from qcodes.utils.validators import Numbers, Ints, Strings, Anything, Enum

from functools import partial

def string_cleaner(s):
    # Remove surrounding whitespace and newline characters
    s = s.strip()

    # Remove surrounding quotes
    if (s[0] == s[-1]) and s.startswith(("'", '"')):
        s = s[1:-1]

    return s.lower()

def output_bool_parser(value):
    # '0' to False, and '1' to True
    return bool(int(value))

def input_bool_parser(value):
    # Convert True/1 to 1, and 'on' to 'on'
    # Convert False/0 to 0, and 'off' to 'off'
    parser = int

    if type(value) == str:
        parser = str

    return parser(value)

class Keithley_2000(VisaInstrument):
    '''
    Driver for the Keithley 2000 multimeter.
    '''
    def __init__(self, name, address, reset=False, **kwargs):
        super().__init__(name, address, **kwargs)

        self._modes = ['VOLT:AC', 'VOLT:DC', 'CURR:AC', 'CURR:DC', 'RES',
                       'FRES', 'TEMP', 'FREQ']

        self._modes += [s.lower() for s in self._modes]

        self.add_parameter('mode',
                           get_cmd='SENS:FUNC?',
                           get_parser=string_cleaner,
                           set_cmd="SENS:FUNC '{}'",
                           vals=Enum(*self._modes))

        # Mode specific parameters
        self.add_parameter('nplc',
                           get_cmd=partial(self._get_mode_param, 'NPLC'),
                           set_cmd=partial(self._set_mode_param, 'NPLC'),
                           vals=Numbers(min_value=0.01, max_value=10))

        self.add_parameter('range',
                           get_cmd=partial(self._get_mode_param, 'RANG'),
                           set_cmd=partial(self._set_mode_param, 'RANG'),
                           vals=Numbers())

        self.add_parameter('auto_range',
                           get_cmd=partial(self._get_mode_param, 'RANG:AUTO'),
                           set_cmd=partial(self._set_mode_param, 'RANG:AUTO'),
                           vals=Anything())

        self.add_parameter('digits',
                           get_cmd=partial(self._get_mode_param, 'DIG'),
                           set_cmd=partial(self._set_mode_param, 'DIG'),
                           vals=Ints(min_value=4, max_value=7))

        self.add_parameter('averaging_type',
                           get_cmd=partial(self._get_mode_param, 'AVER:TCON'),
                           set_cmd=partial(self._set_mode_param, 'AVER:TCON'),
                           vals=Strings())

        self.add_parameter('averaging_count',
                           get_cmd=partial(self._get_mode_param, 'AVER:COUN'),
                           set_cmd=partial(self._set_mode_param, 'AVER:COUN'),
                           vals=Numbers(min_value=1, max_value=100))

        self.add_parameter('averaging',
                           get_cmd=partial(self._get_mode_param, 'AVER:STAT'),
                           set_cmd=partial(self._set_mode_param, 'AVER:STAT'),
                           vals=Anything())

        # Global parameters
        self.add_parameter('display',
                           get_cmd='DISP:ENAB?',
                           get_parser=output_bool_parser,
                           set_cmd='DISP:ENAB {}',
                           set_parser=input_bool_parser,
                           vals=Anything())

        self.add_parameter('trigger_continuous',
                           get_cmd='INIT:CONT?',
                           get_parser=output_bool_parser,
                           set_cmd='INIT:CONT {}',
                           set_parser=input_bool_parser,
                           vals=Anything())

        self.add_parameter('trigger_count',
                           get_cmd='TRIG:COUN?',
                           set_cmd='TRIG:COUN {}',
                           vals=Ints(min_value=1, max_value=9999))

        self.add_parameter('trigger_delay',
                           get_cmd='TRIG:DEL?',
                           set_cmd='TRIG:DEL {}',
                           units='s',
                           vals=Numbers(min_value=0, max_value=999999.999))

        self.add_parameter('trigger_source',
                           get_cmd='TRIG:SOUR?',
                           set_cmd='TRIG:SOUR {}',
                           val_mapping={
                               'immediate': 'IMM\n',
                               'timer': 'TIM\n',
                               'manual': 'MAN\n',
                               'bus': 'BUS\n',
                               'external': 'EXT\n',
                           })

        self.add_parameter('trigger_timer',
                           get_cmd='TRIG:TIM?',
                           set_cmd='TRIG:TIM {}',
                           units='s',
                           vals=Numbers(min_value=0.001, max_value=999999.999))

        self.add_parameter('amplitude',
                           units='arb.unit',
                           get_cmd=':DATA:FRESH?',
                           get_parser=float)

        self.add_function('reset', call_cmd='*RST')

    def _get_mode_param(self, parameter):
        # Select the appropriate return value parser according to the parameter
        parser = float

        if parameter in ['RANG:AUTO', 'AVER:STAT']:
            parser = bool
        elif parameter in ['DIG', 'AVER:COUN']:
            parser = int
        elif parameter in ['AVER:TCON']:
            parser = str

        cmd = '{}:{}?'.format(self.mode(), parameter)

        result = parser(self.ask(cmd))

        if type(result) == str:
            result = string_cleaner(result)

        # Convert some results to a better readable version
        conversions = {
            'mov': 'moving',
            'rep': 'repeat',
        }

        if result in conversions.keys():
            result = conversions[result]

        return result

    def _set_mode_param(self, parameter, value):
        # Convert input bools to 1/0 as required by the Keithley
        if type(value) is bool:
            value = int(value)

        cmd = '{}:{} {}'.format(self.mode(), parameter, value)

        self.write(cmd)

    def _get_param(self, mode, parameter):
        return '{}:{}?'.format(mode, parameter)

    def _set_param(self, mode, parameter, value):
        return '{}:{} {}'.format(mode, parameter, value)