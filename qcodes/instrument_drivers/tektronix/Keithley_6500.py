from functools import partial

from qcodes import VisaInstrument
from qcodes.utils.validators import Bool, Enum, Ints, MultiType, Numbers


def parse_output_string(string_value):
    """ Parses and cleans string output of the multimeter. Removes the surrounding
        whitespace, newline characters and quotes from the parsed data. Some results
        are converted for readablitity (e.g. mov changes to moving).

    Args:
        string_value (str): The data returned from the mulitmeter reading commands.

    Returns:
        String: The cleanup output data.
    """
    s = string_value.strip().lower()
    if (s[0] == s[-1]) and s.startswith(("'", '"')):
        s = s[1:-1]

    conversions = {'mov': 'moving', 'rep': 'repeat'}
    if s in conversions.keys():
        s = conversions[s]
    return s


def parse_output_bool(numeric_value):
    """ Parses and converts the value to boolean type. True is 1.

    Args:
        numeric_value (int, float): The numberical value to convert.

    Returns:
        Bool: The cleanup output data.
    """
    return int(numeric_value) == 1


class Keithley_6500(VisaInstrument):

    def __init__(self, name, address, reset=False, **kwargs):
        """ Driver for the Keithley 6500 multimeter. Based on the Keithley 2000 driver,
            commands have been adapted for the Keithley 6500. This driver does not contain
            all commands available, but only the ones most commonly used.

            Status: beta-version.

        Args:
            name (str): The name used internally by QCoDeS in the DataSet.
            address (str): The VISA device address.
            reset (bool): Reset the device on startup if true.
        """
        super().__init__(name, address, terminator='\n', **kwargs)

        self._trigger_sent = False

        self._mode_map = {'ac current': 'CURR:AC', 'dc current': 'CURR:DC', 'ac voltage': 'VOLT:AC',
                          'dc voltage': 'VOLT:DC', '2w resistance': 'RES', '4w resistance': 'FRES',
                          'temperature': 'TEMP', 'frequency': 'FREQ'}

        self.add_parameter('mode',
                           get_cmd='SENS:FUNC?',
                           set_cmd="SENS:FUNC {}",
                           val_mapping=self._mode_map)

        self.add_parameter('nplc',
                           get_cmd=partial(self._get_mode_param, 'NPLC', float),
                           set_cmd=partial(self._set_mode_param, 'NPLC'),
                           vals=Numbers(min_value=0.01, max_value=10))

        #  TODO: validator, this one is more difficult since different modes
        #  require different validation ranges.
        self.add_parameter('range',
                           get_cmd=partial(self._get_mode_param, 'RANG', float),
                           set_cmd=partial(self._set_mode_param, 'RANG'),
                           vals=Numbers())

        self.add_parameter('auto_range_enabled',
                           get_cmd=partial(self._get_mode_param, 'RANG:AUTO', parse_output_bool),
                           set_cmd=partial(self._set_mode_param, 'RANG:AUTO'),
                           vals=Bool())

        self.add_parameter('digits',
                           get_cmd='DISP:VOLT:DC:DIG?', get_parser=int,
                           set_cmd='DISP:VOLT:DC:DIG? {}',
                           vals=Ints(min_value=4, max_value=7))

        self.add_parameter('averaging_type',
                           get_cmd=partial(self._get_mode_param, 'AVER:TCON', parse_output_string),
                           set_cmd=partial(self._set_mode_param, 'AVER:TCON'),
                           vals=Enum('moving', 'repeat'))

        self.add_parameter('averaging_count',
                           get_cmd=partial(self._get_mode_param, 'AVER:COUN', int),
                           set_cmd=partial(self._set_mode_param, 'AVER:COUN'),
                           vals=Ints(min_value=1, max_value=100))

        self.add_parameter('averaging_enabled',
                           get_cmd=partial(self._get_mode_param, 'AVER:STAT', parse_output_bool),
                           set_cmd=partial(self._set_mode_param, 'AVER:STAT'),
                           vals=Bool())

        # Global parameters
        self.add_parameter('display_enabled',
                           get_parser=parse_output_bool,
                           get_cmd='DISP:ENAB?',
                           set_cmd='DISP:ENAB {}', set_parser=int, vals=Bool())

        self.add_parameter('trigger_count',
                           get_parser=int,
                           get_cmd='ROUT:SCAN:COUN:SCAN?',
                           set_cmd='ROUT:SCAN:COUN:SCAN {}',
                           vals=MultiType(Ints(min_value=1, max_value=9999),
                                          Enum('inf', 'default', 'minimum', 'maximum')))

        self.add_parameter('trigger_delay',
                           get_parser=float,
                           get_cmd='TRIG:DEL?',
                           set_cmd='TRIG:DEL {}',
                           unit='s', vals=Numbers(min_value=0, max_value=999999.999))

        self.add_parameter('trigger_source',
                           get_cmd='TRIG:SOUR?',
                           set_cmd='TRIG:SOUR {}',
                           val_mapping={'immediate': 'NONE', 'timer': 'TIM', 'manual': 'MAN',
                                        'bus': 'BUS', 'external': 'EXT'})

        self.add_parameter('trigger_timer',
                           get_parser=float,
                           get_cmd='ROUT:SCAN:INT?',
                           set_cmd='ROUT:SCAN:INT {}',
                           unit='s', vals=Numbers(min_value=0.001, max_value=999999.999))

        self.add_parameter('amplitude',
                           get_cmd=self._read_next_value,
                           unit='arb.unit')

        self.add_function('reset', call_cmd='*RST')

        if reset:
            self.reset()
        self.write('FORM:DATA ASCII')
        self.connect_message()

    def _read_next_value(self):
          return float(self.ask('READ?'))

    def _get_mode_param(self, parameter, parser):
        """ Reads the current mode of the multimeter and ask for the given parameter.

        Args:
            parameter (str): The asked parameter after getting the current mode.
            parser (function): A function that parses the input buffer read.

        Returns:
            Any: the parsed ask command. The parser determines the return data-type.
        """
        mode = parse_output_string(self._mode_map[self.mode()])
        cmd = '{}:{}?'.format(mode, parameter)
        return parser(self.ask(cmd))

    def _set_mode_param(self, parameter, value):
        """ Reads the current mode of the multimeter and sets the given parameter.

        Args:
            parameter (str): The set parameter after getting the current mode.
            parser (function): A function that parses the input buffer read.
        """
        if isinstance(value, bool):
            value = int(value)

        mode = parse_output_string(self._mode_map[self.mode()])
        cmd = '{}:{} {}'.format(mode, parameter, value)
        self.write(cmd)
