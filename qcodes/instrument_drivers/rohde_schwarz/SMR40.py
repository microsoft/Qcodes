# Driver for microwave source RS_SMR40
#
# Written by Takafumi Fujita (t.fujita@tudelft.nl)
# This program is based on RS_SMR40.py in QTlab
#

import logging

from qcodes import VisaInstrument
from qcodes import validators as vals

log = logging.getLogger(__name__)

class RohdeSchwarz_SMR40(VisaInstrument):
    """This is the qcodes driver for the Rohde & Schwarz SMR40 signal generator
    Status: beta-version.

    .. todo::

        - Add all parameters that are in the manual
        - Add test suite
        - See if there can be a common driver for RS mw sources from which
          different models inherit

    This driver does not contain all commands available for the SMR40 but
    only the ones most commonly used.

    """

    def __init__(self, name, address, verbose=1, reset=False, **kwargs):
        self.verbose = verbose
        log.debug(__name__ + ' : Initializing instrument')
        super().__init__(name, address, **kwargs)

        # TODO(TF): check what parser parameters can do
        #           check what 'tags=['sweep']' and 'types' do in qtlab
        #           fix format types
        self.add_parameter('frequency',
                           label='Frequency',
                           get_cmd=self.do_get_frequency,
                           set_cmd=self.do_set_frequency,
                           vals=vals.Numbers(10e6, 40e9),
                           unit='Hz')
        self.add_parameter('power',
                           label='Power',
                           get_cmd=self.do_get_power,
                           set_cmd=self.do_set_power,
                           vals=vals.Numbers(-30, 25),
                           unit='dBm')
        self.add_parameter('status',
                           get_cmd=self.do_get_status,
                           set_cmd=self.do_set_status,
                           vals=vals.Strings())

        # TODO(TF): check how to fix the get functions
        self.add_parameter('status_of_modulation',
                           get_cmd=self.do_get_status_of_modulation,
                           set_cmd=self.do_set_status_of_modulation,
                           vals=vals.Strings())
        self.add_parameter('status_of_ALC',
                           get_cmd=self.do_get_status_of_ALC,
                           set_cmd=self.do_set_status_of_ALC,
                           vals=vals.Strings())
        self.add_parameter('pulse_delay',
                           get_cmd=self.do_get_pulse_delay,
                           set_cmd=self.do_set_pulse_delay)

        # TODO(TF): check the way of defining this type of functions, where logging is added
        # self.add_function('reset')
        # self.add_function('get_all')

        if reset:
            self.reset()
        else:
            self.get_all()

        self.connect_message()

    # Functions
    def reset(self):
        """Resets the instrument to default values.

        Args:
            None

        Output:
            None

        """
        log.info(__name__ + ' : Resetting instrument')
        self.write('*RST')
        # TODO: make it printable
        self.get_all()

    def get_all(self):
        """Reads all implemented parameters from the instrument, and updates
        the wrapper.

        Args:
            None

        Output:
            None

        """
        log.info(__name__ + ' : reading all settings from instrument')
        # TODO: make it printable
        self.frequency.get()
        self.power.get()
        self.status.get()

    # Communication functions
    def do_get_frequency(self):
        """Get frequency from device.

        Args:
            None

        Output:
            frequency (float) : frequency in Hz

        """
        log.debug(__name__ + ' : reading frequency from instrument')
        return float(self.ask('SOUR:FREQ?'))

    def do_set_frequency(self, frequency):
        """Set frequency of device.

        Args:
            frequency (float) : frequency in Hz

        Output:
            None

        """
        log.debug(__name__ + ' : setting frequency to %s GHz' % frequency)
        self.write('SOUR:FREQ %e' % frequency)

    def do_get_power(self):
        """Get output power from device.

        Args:
            None

        Output:
            power (float) : output power in dBm

        """
        log.debug(__name__ + ' : reading power from instrument')
        return float(self.ask('SOUR:POW?'))

    def do_set_power(self, power):
        """Set output power of device.

        Args:
            power (float) : output power in dBm

        Output:
            None

        """
        log.debug(__name__ + ' : setting power to %s dBm' % power)
        self.write('SOUR:POW %e' % power)

    def do_get_status(self):
        """Get status from instrument.

        Args:
            None

        Output:
            status (str) : 'on or 'off'

        """
        log.debug(__name__ + ' : reading status from instrument')
        stat = self.ask(':OUTP:STAT?')

        # TODO: fix
        if stat == '1\n':
            return 'ON'
        elif stat == '0\n':
            return 'OFF'
        else:
            raise ValueError('Output status not specified : %s' % stat)

    def do_set_status(self, status):
        """Set status of instrument.

        Args:
            status (str) : 'on or 'off'

        Output:
            None

        """
        log.debug(__name__ + ' : setting status to "%s"' % status)
        if status.upper() in ('ON', 'OFF'):
            status = status.upper()
        else:
            raise ValueError('set_status(): can only set on or off')
        self.write(':OUTP:STAT %s' % status)

    def do_get_status_of_modulation(self):
        """Get status from instrument.

        Args:
            None

        Output:
            status (str) : 'on' or 'off'

        """
        log.debug(__name__ + ' : reading status from instrument')
        stat = self.ask(':SOUR:PULM:STAT?')

        # TODO: fix
        # if stat == '1':
        # return 'ON'
        # elif stat == '0':
        # return 'OFF'
        if stat == '1\n':
            return 'ON'
        elif stat == '0\n':
            return 'OFF'
        else:
            raise ValueError('Output status not specified : %s' % stat)

    def do_set_status_of_modulation(self, status):
        """Set status of modulation.

        Args:
            status (str) : 'on' or 'off'

        Output:
            None

        """
        log.debug(__name__ + ' : setting status to "%s"' % status)
        if status.upper() in ('ON', 'OFF'):
            status = status.upper()
        else:
            raise ValueError('set_status(): can only set on or off')
        self.write(':SOUR:PULM:STAT %s' % status)

    def do_get_status_of_ALC(self):
        """Get status from instrument.

        Args:
            None

        Output:
            status (str) : 'on or 'off'

        """
        log.debug(__name__ + ' : reading ALC status from instrument')
        stat = self.ask(':SOUR:POW:ALC?')

        # TODO: fix
        # if stat == '1':
        # return 'ON'
        # elif stat == '0':
        # return 'OFF'
        if stat == '1\n':
            return 'ON'
        elif stat == '0\n':
            return 'OFF'
        else:
            raise ValueError('Output status not specified : %s' % stat)

    def do_set_status_of_ALC(self, status):
        """Set status of instrument.

        Args:
            status (str) : 'on or 'off'

        Output:
            None

        """
        log.debug(__name__ + ' : setting ALC status to "%s"' % status)
        if status.upper() in ('ON', 'OFF'):
            status = status.upper()
        else:
            raise ValueError('set_status(): can only set on or off')
        self.write(':SOUR:POW:ALC %s' % status)

    def do_get_pulse_delay(self):
        """Get output power from device.

        Args:
            None

        Output:
            power (float) : output power in dBm

        """
        log.debug(__name__ + ' : reading pulse delay from instrument')
        return float(self.ask('SOUR:PULS:DEL?'))

    def do_set_pulse_delay(self, delay):
        """Set output power of device.

        Args:
            power (float) : output power in dBm

        Output:
            None

        """
        log.debug(
            __name__ + ' : setting pulse delay to %s seconds' % str(delay))
        self.write('SOUR:PULS:DEL 1us')

    # Shortcuts
    def off(self):
        """Set status to 'off'.

        Args:
            None

        Output:
            None

        """
        self.status.set('off')

    def on(self):
        """Set status to 'on'.

        Args:
            None

        Output:
            None

        """
        self.status.set('on')

    def off_modulation(self):
        """Set status of modulation to 'off'.

        Args:
            None

        Output:
            None

        """
        self.set_status_of_modulation('off')

    def on_modulation(self):
        """Set status of modulation to 'on'.

        Args:
            None

        Output:
            None

        """
        self.set_status_of_modulation('on')

    def set_ext_trig(self):
        """Set to the external trigger mode.

        Args:
            None

        Output:
            None

        """
        log.debug(__name__ + ' : setting to the external trigger mode')
        self.write('TRIG:PULS:SOUR EXT_TRIG')
