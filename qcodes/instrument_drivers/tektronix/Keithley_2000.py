
import logging
from functools import partial

from qcodes.instrument.visa import VisaInstrument
from qcodes.utils.validators import Strings as StringValidator
from qcodes.utils.validators import Ints as IntsValidator
from qcodes.utils.validators import Numbers as NumbersValidator

from qcodes.utils.validators import Enum, MultiType

# %% Helper functions


def bool_to_str(val):
    '''
    Function to convert boolean to 'ON' or 'OFF'
    '''
    if val:
        return "ON"
    else:
        return "OFF"

# %% Driver for Keithley_2000


def parseint(v):
    logging.debug('parseint: %s -> %d' % (v, int(v)))
    return int(v)


def parsebool(v):
    r = bool(int(v))
    logging.debug('parsetobool: %s -> %d' % (v, r))
    return r


def parsestr(v):
    return v.strip().strip('"')


class Keithley_2000(VisaInstrument):
    '''
    This is the qcodes driver for the Keithley_2000 Multimeter

    Usage: Initialize with
    <name> =  = Keithley_2700(<name>, address='<GPIB address>', reset=<bool>,
            change_display=<bool>, change_autozero=<bool>)

    Status: beta-version.

    This driver will most likely work for multiple Keithley SourceMeters.

    This driver does not contain all commands available, but only the ones
    most commonly used.
    '''
    def __init__(self, name, address, reset=False, **kwargs):
        super().__init__(name, address, **kwargs)

        self._modes = ['VOLT:AC', 'VOLT:DC', 'CURR:AC', 'CURR:DC', 'RES',
                       'FRES', 'TEMP', 'FREQ']
        self._averaging_types = ['MOV', 'REP']
        self._trigger_sent = False

        self.add_parameter('mode',
                           get_cmd=':CONF?',
                           get_parser=parsestr,
                           set_cmd=':CONF:{}',
                           vals=StringValidator())

        # Mode specific parameters
        self.add_parameter('nplc',
                           get_cmd=partial(self._get_mode_param, 'NPLC',
                                           float),
                           set_cmd=partial(self._set_mode_param, 'NPLC'),
                           vals=NumbersValidator(min_value=0.01, max_value=10))

        # TODO: validator, this one is more difficult since different modes
        # require different validation ranges
        self.add_parameter('range',
                           get_cmd=partial(self._get_mode_param, 'RANG',
                                           float),
                           set_cmd=partial(self._set_mode_param, 'RANG'),
                           vals=NumbersValidator())

        self.add_parameter('auto_range',
                           get_cmd=partial(self._get_mode_param, 'RANG:AUTO',
                                           parser=parsebool),
                           set_cmd=partial(self._set_mode_param, 'RANG:AUTO'),
                           vals=Enum('on', 'off'))

        self.add_parameter('digits',
                           get_cmd=partial(self._get_mode_param, 'DIG', int),
                           set_cmd=partial(self._set_mode_param, 'DIG'),
                           vals=IntsValidator(min_value=4, max_value=7))

        self.add_parameter('averaging_type',
                           get_cmd=partial(self._get_mode_param, 'AVER:TCON',
                                           parsestr),
                           set_cmd=partial(self._set_mode_param, 'AVER:TCON'),
                           vals=Enum('moving', 'repeat'))

        self.add_parameter('averaging_count',
                           get_cmd=partial(self._get_mode_param, 'AVER:COUN',
                                           int),
                           set_cmd=partial(self._set_mode_param, 'AVER:COUN'),
                           vals=IntsValidator(min_value=1, max_value=100))

        self.add_parameter('averaging',
                           get_cmd=partial(self._get_mode_param, 'AVER:STAT',
                                           parser=parsebool),
                           set_cmd=partial(self._set_mode_param, 'AVER:STAT'),
                           vals=Enum('on', 'off'))

        # Global parameters
        self.add_parameter('display',
                           get_cmd=self._mode_par('DISP', 'ENAB'),
                           get_parser=parsebool,
                           set_cmd=self._mode_par_value('DISP', 'ENAB', '{}'),
                           vals=Enum('on', 'off'))

        self.add_parameter('trigger_continuous',
                           get_cmd='INIT:CONT?',
                           get_parser=parsebool,
                           set_cmd='INIT:CONT {}',
                           vals=Enum('on', 'off'))

        self.add_parameter('trigger_count',
                           get_cmd='TRIG:COUN?',
                           set_cmd='TRIG:COUN {}',
                           vals=MultiType(IntsValidator(min_value=1,
                                                        max_value=9999),
                                          Enum('inf')))

        self.add_parameter('trigger_delay',
                           get_cmd='TRIG:DEL?',
                           set_cmd='TRIG:DEL {}',
                           units='s',
                           vals=NumbersValidator(min_value=0,
                                                 max_value=999999.999))

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
                           vals=NumbersValidator(min_value=0.001,
                                                 max_value=999999.999))

        self.add_parameter('amplitude',
                           units='arb.unit',
                           get_cmd=self._read_next_value)

        self.add_function('reset', call_cmd='*RST')

        if reset:
            self.reset()

        self.connect_message()

    def _determine_mode(self, mode):
        '''
        Return the mode string to use.
        If mode is None it will return the currently selected mode.
        '''
        logging.debug('Determine mode with mode=%s' % mode)
        if mode is None:
            mode = self.mode.get_latest()  # _mode(query=False)
        if mode not in self._modes and mode not in ('INIT', 'TRIG', 'SYST',
                                                    'DISP'):
            logging.warning('Invalid mode %s, assuming current' % mode)
            mode = self.mode.get_latest()
        logging.debug('Determine mode: mode=%s' % mode)
        return mode

    def _mode_par_value(self, mode, par, val):
        '''
        For internal use only!!
        Create command string based on current mode

        Input:
            mode (string) : The mode to use
            par (string)  : Parameter
            val (depends) : Value

        Output:
            None
        '''
        mode = self._determine_mode(mode)
        string = ':%s:%s %s' % (mode, par, val)
        return string

    def _mode_par(self, mode, par):
        '''
        For internal use only!!
        Create command string based on current mode

        Input:
            mode (string) : The mode to use
            par (string)  : Parameter
            val (depends) : Value

        Output:
            None
        '''
        mode = self._determine_mode(mode)
        string = ':%s:%s?' % (mode, par, )
        return string

    def trigger(self):
        if self.trigger_continuous() == 'off':
            self.write('INIT')
            self._trigger_sent = True

    def _read_next_value(self):
        # Prevent a timeout when no trigger has been sent
        if self.trigger_continuous() == 'off' and not self._trigger_sent:
            return 0.0

        self._trigger_sent = False

        return float(self.ask('DATA:FRESH?'))

    def _get_mode_param(self, parameter, parser):
        cmd = '{}:{}?'.format(self.mode(), parameter)

        return parser(self.ask(cmd))

    def _set_mode_param(self, parameter, value):
        cmd = '{}:{} {}'.format(self.mode(), parameter, value)

        self.write(cmd)
