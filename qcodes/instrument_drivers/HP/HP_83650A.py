# Driver for microwave source HP_83650A
#
# Written by Bruno Buijtendorp (brunobuijtendorp@gmail.com)
import logging
from typing import Any, Optional

from qcodes import validators as vals
from qcodes.instrument import VisaInstrument

log = logging.getLogger(__name__)


def parsestr(v: str) -> str:
    return v.strip().strip('"')


class HP83650A(VisaInstrument):
    """
    QCoDeS driver for HP 83650A

    """

    def __init__(self,
                 name: str,
                 address: str,
                 verbose: int = 1,
                 reset: bool = False,
                 server_name: Optional[str] = None,
                 **kwargs: Any):

        self.verbose = verbose
        log.debug('Initializing instrument')
        super().__init__(name, address, **kwargs)

        self.add_parameter('frequency',
                           label='Frequency',
                           get_cmd='FREQ:CW?',
                           set_cmd='FREQ:CW {}',
                           vals=vals.Numbers(10e6, 40e9),
                           docstring='Microwave frequency, ....',
                           get_parser=float,
                           unit='Hz')

        self.add_parameter('freqmode',
                           label='Frequency mode',
                           get_cmd='FREQ:MODE?',
                           set_cmd='FREQ:MODE {}',
                           vals=vals.Strings(),
                           get_parser=parsestr,
                           docstring='Microwave frequency mode, ....')

        self.add_parameter('power',
                           label='Power',
                           get_cmd='SOUR:POW?',
                           set_cmd='SOUR:POW {}',
                           vals=vals.Numbers(-20, 20),
                           get_parser=float,
                           unit='dBm',
                           docstring='Microwave power, ....')

        self.add_parameter('rfstatus',
                           label='RF status',
                           get_cmd=':POW:STAT?',
                           set_cmd=':POW:STAT {}',
                           val_mapping={'on': '1', 'off': '0'},
                           vals=vals.Strings(),
                           get_parser=parsestr,
                           docstring='Status, ....')

        self.add_parameter('fmstatus',
                           label='FM status',
                           get_cmd=':FM:STAT?',
                           set_cmd=':FM:STAT {}',
                           val_mapping={'on': '1', 'off': '0'},
                           vals=vals.Strings(),
                           get_parser=parsestr,
                           docstring='FM status, ....')

        self.add_parameter('fmcoup',
                           label='FM coupling',
                           get_cmd=':FM:COUP?',
                           set_cmd=':FM:COUP {}',
                           vals=vals.Strings(),
                           get_parser=parsestr,
                           docstring='FM coupling, ....')

        self.add_parameter('amstatus',
                           label='AM status',
                           get_cmd=':AM:STAT?',
                           set_cmd=':AM:STAT {}',
                           val_mapping={'on': '1', 'off': '0'},
                           vals=vals.Strings(),
                           get_parser=parsestr,
                           docstring='AM status, ....')

        self.add_parameter('pulsestatus',
                           label='Pulse status',
                           get_cmd=':PULS:STAT?',
                           set_cmd=':PULS:STAT {}',
                           val_mapping={'on': '1', 'off': '0'},
                           vals=vals.Strings(),
                           get_parser=parsestr,
                           docstring='Pulse status, ....')

        self.add_parameter('pulsesource',
                           label='Pulse source',
                           get_cmd=':PULS:SOUR?',
                           set_cmd=':PULS:SOUR {}',
                           vals=vals.Strings(),
                           get_parser=parsestr,
                           docstring='Pulse source, ....')
        self.connect_message()

    def reset(self) -> None:
        log.debug('Resetting instrument')
        self.write('*RST')
        self.print_all()

    def print_all(self) -> None:
        log.debug('Reading all settings from instrument')
        print(self.rfstatus.label + ':', self.rfstatus.get())
        print(self.power.label + ':', self.power.get(), self.power.unit)
        print(self.frequency.label +
              ': %e' % self.frequency.get(), self.frequency.unit)
        print(self.freqmode.label + ':', self.freqmode.get())
        self.print_modstatus()

    def print_modstatus(self) -> None:
        print(self.fmstatus.label + ':', self.fmstatus.get())
        print(self.fmcoup.label + ':', self.fmcoup.get())
        print(self.amstatus.label + ':', self.amstatus.get())
        print(self.pulsestatus.label + ':', self.pulsestatus.get())
        print(self.pulsesource.label + ':', self.pulsesource.get())


class HP_83650A(HP83650A):
    """
    Alias of HP83650A for backwards compatibility.
    Will eventually be deprecated and removed.
    """
