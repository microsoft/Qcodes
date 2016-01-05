# load the qcodes path, until we have this installed as a package
import sys
qcpath = 'D:\GitHubRepos\Qcodes'
if qcpath not in sys.path:
    sys.path.append(qcpath)

from qcodes.instrument.visa import VisaInstrument
from qcodes.utils import validators as vals


class RS_SGS100A(VisaInstrument):
    '''
    This is the qcodes driver for the Rohde & Schwarz SGS100A signal generator

    This driver will most likely work for multiple Rohde & Schwarz sources.
    it would be a good
    This driver does not contain all commands available for the RS_SGS100A but
    only the ones most commonly used.
    '''
    def __init__(self, name, address):
        super().__init__(name, address)
        self.add_parameter('IDN', get_cmd='*IDN?')
        self.add_parameter(name='frequency',
                           label='Frequency (Hz)',
                           get_cmd='SOUR:FREQ' + '?',
                           set_cmd='SOUR:FREQ' + ' {:.2f}',
                           parse_function=float,
                           vals=vals.Numbers(1e9, 20e9))
        self.add_parameter(name='phase',
                           label='Phase (deg)',
                           get_cmd='SOUR:PHAS' + '?',
                           set_cmd='SOUR:PHAS' + ' {:.2f}',
                           parse_function=float,
                           vals=vals.Numbers(0, 360))
        self.add_parameter(name='power',
                           label='Power (dBm)',
                           get_cmd='SOUR:POW' + '?',
                           set_cmd='SOUR:POW' + ' {:.2f}',
                           parse_function=float,
                           vals=vals.Numbers(-120, 25))
        self.add_parameter('status',
                           get_cmd=':OUTP:STAT?',
                           set_cmd=self.set_status,
                           parse_function=self.parse_on_off,
                           vals=vals.Strings())
        self.add_parameter('pulsemod_state',
                           get_cmd=':SOUR:PULM:STAT?',
                           set_cmd=self.set_pulsemod_state,
                           parse_function=self.parse_on_off,
                           vals=vals.Strings())
        self.add_parameter('pulsemod_source',
                           get_cmd='SOUR:PULM:SOUR?',
                           set_cmd=self.set_pulsemod_source,
                           vals=vals.Strings())
        self.add_function('reset', call_cmd='*RST')
        self.add_function('run_self_tests', call_cmd='*TST?')

    def parse_on_off(self, stat):
        if stat == '0\n':
            stat = 'Off'
        elif stat == '1\n':
            stat = 'On'
        return stat

    def set_status(self, stat):
        if stat.upper() in ('ON', 'OFF'):
            self.visa_handle.write(':OUTP:STAT %s' % stat)
        else:
            raise ValueError('Unable to set status to %s,\
                             expected "ON" or "OFF"' % stat)

    def set_pulsemod_state(self, stat):
        if stat.upper() in ('ON', 'OFF'):
            self.visa_handle.write(':PULM:SOUR EXT')
            self.visa_handle.write(':SOUR:PULM:STAT %s' % stat)
        else:
            raise ValueError('Unable to set status to %s,\
                             expected "ON" or "OFF"' % stat)

    def set_pulsemod_source(self, source):
        if source.upper() in ('INT', 'EXT'):
            self.visa_handle.write(':SOUR:PULM:SOUR %s' % source)
        else:
            raise ValueError('Unable to set source to %s,\
                             expected "INT" or "EXT"' % source)
