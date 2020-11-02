from typing import Any, TypeVar, Callable
from functools import partial
from typing import Union

from qcodes import VisaInstrument
from qcodes.utils.validators import Bool, Enum, Ints, MultiType, Numbers

T = TypeVar("T")


def _parse_output_string(string_value: str) -> str:
    """ Parses and cleans string output of the multimeter. Removes the surrounding
        whitespace, newline characters and quotes from the parsed data. Some results
        are converted for readablitity (e.g. mov changes to moving).

    Args:
        string_value: The data returned from the multimeter reading commands.

    Returns:
        The cleaned-up output of the multimeter.
    """
    s = string_value.strip().lower()
    if (s[0] == s[-1]) and s.startswith(("'", '"')):
        s = s[1:-1]

    conversions = {'mov': 'moving', 'rep': 'repeat'}
    if s in conversions.keys():
        s = conversions[s]
    return s


def _parse_output_bool(numeric_value: float) -> bool:
    """ Parses and converts the value to boolean type. True is 1.

    Args:
        numeric_value: The numerical value to convert.

    Returns:
        The boolean representation of the numeric value.
    """
    return bool(numeric_value)


class CommandSetError(Exception):
    pass


class Keithley_6500(VisaInstrument):

    def __init__(
            self,
            name: str,
            address: str,
            reset_device: bool = False,
            **kwargs: Any):
        """ Driver for the Keithley 6500 multimeter. Based on the Keithley 2000 driver,
            commands have been adapted for the Keithley 6500. This driver does not contain
            all commands available, but only the ones most commonly used.

            Status: beta-version.

        Args:
            name (str): The name used internally by QCoDeS in the DataSet.
            address (str): The VISA device address.
            reset_device (bool): Reset the device on startup if true.
        """
        super().__init__(name, address, terminator='\n', **kwargs)

        command_set = self.ask('*LANG?')
        if command_set != 'SCPI':
            error_msg = "This driver only compatible with the 'SCPI' command " \
                        "set, not '{}' set".format(command_set)
            raise CommandSetError(error_msg)

        self._trigger_sent = False

        self._mode_map = {'ac current': 'CURR:AC', 'dc current': 'CURR:DC', 'ac voltage': 'VOLT:AC',
                          'dc voltage': 'VOLT:DC', '2w resistance': 'RES', '4w resistance': 'FRES',
                          'temperature': 'TEMP', 'frequency': 'FREQ'}

        self.add_parameter('mode',
                           get_cmd='SENS:FUNC?',
                           set_cmd="SENS:FUNC {}",
                           val_mapping=self._mode_map)

        self.add_parameter('nplc',
                           get_cmd=partial(
                               self._get_mode_param, 'NPLC', float),
                           set_cmd=partial(self._set_mode_param, 'NPLC'),
                           vals=Numbers(min_value=0.01, max_value=10))

        #  TODO: validator, this one is more difficult since different modes
        #  require different validation ranges.
        self.add_parameter('range',
                           get_cmd=partial(
                               self._get_mode_param, 'RANG', float),
                           set_cmd=partial(self._set_mode_param, 'RANG'),
                           vals=Numbers())

        self.add_parameter('auto_range_enabled',
                           get_cmd=partial(self._get_mode_param,
                                           'RANG:AUTO', _parse_output_bool),
                           set_cmd=partial(self._set_mode_param, 'RANG:AUTO'),
                           vals=Bool())

        self.add_parameter('digits',
                           get_cmd='DISP:VOLT:DC:DIG?', get_parser=int,
                           set_cmd='DISP:VOLT:DC:DIG? {}',
                           vals=Ints(min_value=4, max_value=7))

        self.add_parameter('averaging_type',
                           get_cmd=partial(self._get_mode_param,
                                           'AVER:TCON', _parse_output_string),
                           set_cmd=partial(self._set_mode_param, 'AVER:TCON'),
                           vals=Enum('moving', 'repeat'))

        self.add_parameter('averaging_count',
                           get_cmd=partial(self._get_mode_param,
                                           'AVER:COUN', int),
                           set_cmd=partial(self._set_mode_param, 'AVER:COUN'),
                           vals=Ints(min_value=1, max_value=100))

        self.add_parameter('averaging_enabled',
                           get_cmd=partial(self._get_mode_param,
                                           'AVER:STAT', _parse_output_bool),
                           set_cmd=partial(self._set_mode_param, 'AVER:STAT'),
                           vals=Bool())

        # Global parameters
        self.add_parameter('display_backlight',
                           docstring='Control the brightness of the display '
                                     'backligt. Off turns the display off and'
                                     'Blackout also turns off indicators and '
                                     'key lights on the device.',
                           get_cmd='DISP:LIGH:STAT?',
                           set_cmd='DISP:LIGH:STAT {}',
                           val_mapping={'On 100': 'ON100',
                                        'On 75': 'ON75',
                                        'On 50': 'ON50',
                                        'On 25': 'ON25',
                                        'Off': 'OFF',
                                        'Blackout': 'BLACkout'})

        self.add_parameter('trigger_count',
                           get_parser=int,
                           get_cmd='ROUT:SCAN:COUN:SCAN?',
                           set_cmd='ROUT:SCAN:COUN:SCAN {}',
                           vals=MultiType(Ints(min_value=1, max_value=9999),
                                          Enum('inf', 'default', 'minimum', 'maximum')))

        for trigger in range(1, 5):
            self.add_parameter('trigger%i_delay' % trigger,
                               docstring='Set and read trigger delay for '
                                         'timer %i.' % trigger,
                               get_parser=float,
                               get_cmd='TRIG:TIM%i:DEL?' % trigger,
                               set_cmd='TRIG:TIM%i:DEL {}' % trigger,
                               unit='s', vals=Numbers(min_value=0,
                                                      max_value=999999.999))

            self.add_parameter('trigger%i_source' % trigger,
                               docstring='Set the trigger source for '
                                         'timer %i.' % trigger,
                               get_cmd='TRIG:TIM%i:STAR:STIM?' % trigger,
                               set_cmd='TRIG:TIM%i:STAR:STIM {}' % trigger,
                               val_mapping={'immediate': 'NONE',
                                            'timer1': 'TIM1',
                                            'timer2': 'TIM2',
                                            'timer3': 'TIM3',
                                            'timer4': 'TIM4',
                                            'notify1': 'NOT1',
                                            'notify2': 'NOT2',
                                            'notify3': 'NOT3',
                                            'front-panel': 'DISP',
                                            'bus': 'COMM',
                                            'external': 'EXT'})

        # Control interval between scans; the default value from the instrument is 0,
        # hence 0 is included in the validator's range of this parameter.
        self.add_parameter('trigger_timer',
                           get_parser=float,
                           get_cmd='ROUT:SCAN:INT?',
                           set_cmd='ROUT:SCAN:INT {}',
                           unit='s', vals=Numbers(min_value=0, max_value=999999.999))

        self.add_parameter('amplitude',
                           get_cmd=self._read_next_value,
                           set_cmd=False,
                           unit='a.u.')

        if reset_device:
            self.reset()
        self.write('FORM:DATA ASCII')
        self.connect_message()

    def reset(self) -> None:
        """ Reset the device """
        self.write('*RST')

    def _read_next_value(self) -> float:
        return float(self.ask('READ?'))

    def _get_mode_param(self, parameter: str, parser: Callable[[str], T]) -> T:
        """ Reads the current mode of the multimeter and ask for the given parameter.

        Args:
            parameter: The asked parameter after getting the current mode.
            parser: A function that parses the input buffer read.

        Returns:
            Any: the parsed ask command. The parser determines the return data-type.
        """
        mode = _parse_output_string(self._mode_map[self.mode()])
        cmd = f'{mode}:{parameter}?'
        return parser(self.ask(cmd))

    def _set_mode_param(self, parameter: str, value: Union[str, float, bool]) -> None:
        """ Gets the current mode of the multimeter and sets the given parameter.

        Args:
            parameter: The set parameter after getting the current mode.
            value: Value to set
        """
        if isinstance(value, bool):
            value = int(value)

        mode = _parse_output_string(self._mode_map[self.mode()])
        cmd = f'{mode}:{parameter} {value}'
        self.write(cmd)
