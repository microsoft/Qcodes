"""

# Please refer to Cryogenic's Magnet Power Supply SMS120C manual for further details and functionality.
# This magnet PS model is not SCPI compliant.
# Note: Some commands return more than one line in the output,
        some are unidirectional, with no return (eg. 'write' rathern than 'ask').

This magnet PS driver has been tested with:
    FTDI chip drivers (USB to serial), D2XX version installed.

"""

import visa
import re
import logging
import time

from qcodes.utils.validators import Numbers
from qcodes import VisaInstrument
import pyvisa.constants as vi_const


log = logging.getLogger(__name__)


class CryogenicSMS120C(VisaInstrument):

    """
    The following hard-coded, default values for Cryogenic magnets are safety limits
    and should not be modified.
    - these values should be set using the corresponding arguments when the class is called.
    """
    default_current_ramp_limit = 0.0506  # [A/s]
    default_max_current_ramp_limit = 0.12  # [A/s]

    """
    Driver for the Cryogenic SMS120C magnet power supply.
    This class controls a single magnet PSU.
    Magnet and magnet PSU limits : max B=12T, I=105.84A, V=3.5V

    Args:
        name (string): a name for the instrument
        address (string): (serial to USB) COM number of the power supply
        coil_constant (float): coil constant in Tesla per ampere, fixed at 0.113375T/A
        current_rating (float): maximum current rating in ampere, fixed at 105.84A
        current_ramp_limit (float): current ramp limit in ampere per second,
            for 50mK operation 0.0506A/s (5.737E-3 T/s, 0.34422T/min) - usually used
            for 4K operation 0.12A/s (0.013605 T/s, 0.8163 T/min) - not recommended

    Note about timing : SMS120C needs a minimum of 200ms delay between commands being sent
    """

    # Reg. exp. to match a float or exponent in a string
    _re_float_exp = r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?'

    def __init__(self, name, address, coil_constant=0.113375, current_rating=105.84,
                 current_ramp_limit=0.0506, reset=False, timeout=5, terminator='\r\n', **kwargs):

        log.debug('Initializing instrument')
        super().__init__(name, address, terminator=terminator, **kwargs)

        self.visa_handle.baud_rate = 9600
        self.visa_handle.parity = visa.constants.Parity.none
        self.visa_handle.stop_bits = visa.constants.StopBits.one
        self.visa_handle.data_bits = 8
        self.visa_handle.flow_control = 0
        self.visa_handle.flush(vi_const.VI_READ_BUF_DISCARD |
                               vi_const.VI_WRITE_BUF_DISCARD)  # keep for debugging

        idn = self.IDN.get()
        print(idn)

        self._persistentField = 0  # temp code stub
        self._coil_constant = coil_constant
        self._current_rating = current_rating
        self._current_ramp_limit = current_ramp_limit
        self._field_rating = coil_constant * \
            current_rating  # corresponding max field based
        self._field_ramp_limit = coil_constant * current_ramp_limit

        self.add_parameter(name='unit',
                           get_cmd=self._get_unit,
                           set_cmd=self._set_unit,
                           val_mapping={'AMPS': 0, 'TESLA': 1})

        self.add_parameter('rampStatus',
                           get_cmd=self._get_rampStatus,
                           val_mapping={'HOLDING': 0,
                                        'RAMPING': 1,
                                        'QUENCH DETECTED': 2,
                                        'EXTERNAL TRIP': 3,
                                        'FAULT': 4,
                                        })

        self.add_parameter('polarity',
                           get_cmd=self._get_polarity,
                           set_cmd=self._set_polarity,
                           val_mapping={'POSITIVE': '+', 'NEGATIVE': '-'})

        self.add_parameter(name='switchHeater',
                           get_cmd=self._get_switchHeater,
                           set_cmd=self._set_switchHeater,
                           val_mapping={False: 0, True: 1})

        self.add_parameter('persistentMode',
                           get_cmd=self._get_persistentMode,
                           set_cmd=self._set_persistentMode,
                           val_mapping={False: 0, True: 1})

        self.add_parameter(name='persistentField',
                           get_cmd=self._get_persistentField,
                           vals=Numbers(self._persistentField))

        self.add_parameter(name='field',
                           get_cmd=self._get_field,
                           set_cmd=self._set_field,
                           vals=Numbers(-self._field_rating,  # i.e. ~12T, calculated
                                        self._field_rating))

        self.add_parameter(name='maxField',
                           get_cmd=self._get_maxField,
                           set_cmd=self._set_maxField,
                           vals=Numbers(0,  # i.e. ~12T, calculated
                                        self._field_rating))

        self.add_parameter(name='rampRate',
                           get_cmd=self._get_rampRate,
                           set_cmd=self._set_rampRate,
                           vals=Numbers(0,
                                        self._current_ramp_limit))

        self.add_parameter('pauseRamp',
                           set_cmd=self._set_pauseRamp,
                           get_cmd=self._get_pauseRamp,
                           val_mapping={False: 0, True: 1})


    def get_idn(self):
        r"""
        Overwrites the get_idn function using constants as the hardware
        does not have a proper \*IDN function.
        """
        idparts = ['Cryogenic', 'Magnet PS SMS120C', 'None', '1.0']

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))

    def query(self, msg):
        """
        Message outputs do not follow the standard SCPI format,
        separate regexp to parse unique/variable instrument message structures.

        Returns:
            key : unused
            value : parsed value extracted from output message
        """
        value = self.ask(msg)
        m = re.match(r'((\S{8})\s)+(([^:]+)(:([^:]+))?)', value)
        if m:
            if m[2] == '------->':
                log.error(
                    'Command information or unrecognizable qualifier: "%s"' % m[3])
                return None, None
            else:
                return m[4].strip(), m[6].strip()
        else:
            log.error(
                'Malformed message received from the magnet PS: "%s"' % value)
            return None, None

    def _get_limit(self):  # Get voltage limits, returns a float
        _, value = self.query('GET VL')
        # extract number from string
        m = re.match(r'({}) VOLTS'.format(
            CryogenicSMS120C._re_float_exp), value)
        limit = float(m[1])
        return limit

    # get heater status, returns a boolean ON (1) or OFF (0)
    def _get_switchHeater(self):
        _, value = self.query('HEATER')
        if 'OFF' in value:
            switchHeater = 0
        elif 'ON' in value:
            switchHeater = 1
        return switchHeater

    # check if magnet is in persistent mode, and if so return current in the
    # magnet
    def _get_persistentMode(self):
        _, value = self.query('HEATER')
        field = self._get_field()
        # check for switch heater OFF, and non-zero current
        if 'OFF' in value and abs(field <= 0.007):
            persistentField = self._get_persistentField()
            units = self._get_unit()
            if units == 1:
                log.info("Magnet in persistent mode, at a field of %f T" %
                         persistentField)
            elif units == 0:
                log.info("Magnet in persistent mode, at a field of %f A" %
                         persistentField)
            persistentMode = True
        else:
            log.info("Magnet not persistent.")
            persistentMode = False
        return persistentMode

    # get units, returns a boolean integer - Tesla (1) or Amps(0)
    def _get_persistentField(self):
        # read persistent field from controller
        BLeads = self._get_field()
        if (self._get_switchHeater() == 1):
            log.info("Switch heater ON, magnet not in persistent mode.")
            persistentField = 0
        elif (self._get_switchHeater() == 0) and (abs(BLeads) > 0.007):
            log.info(
                "Switch heater OFF, but current is still in present in leads - not in persistent mode. Leads at: %f" % BLeads)
            perString = self.ask('GET PER')
            m = re.match(r'((\S{8})\s)+([^:]+)', perString)
            persistentField = float(m[2])
        else:
            perString = self.ask('GET PER')
            # handles a different string return format
            m = re.match(r'((\S{8})\s)+([^:]+)', perString)
            persistentField = float(m[2])
        return persistentField

    def _get_unit(self):  # get units, returns a boolean integer - Tesla (1) or Amps(0)
        _, value = self.query('TESLA')
        if value == 'TESLA':
            unit = 1
        else:  # assume in Amps
            unit = 0
        return unit

    # get direction of current, returns a string - Positive (1) or Negative(0)
    def _get_polarity(self):
        _, value = self.query('GET SIGN')
        if value == 'POSITIVE':
            polarity = '+'
        elif value == 'NEGATIVE':  # assume Negative
            polarity = '-'
        return polarity

    def _get_maxField(self):  # Get the maximum B field, returns a float (in Amps or Tesla)
        _, value = self.query('GET MAX')
        units = self._get_unit()
        if units == 1:
            m = re.match(r'({}) TESLA'.format(
                CryogenicSMS120C._re_float_exp), value)
        elif units == 0:
            m = re.match(r'({}) AMPS'.format(
                CryogenicSMS120C._re_float_exp), value)
        maxField = float(m[1])
        return maxField

    # Get current magnetic field, returns a float (assume in Tesla)
    def _get_field(self):
        _, value = self.query('GET OUTPUT')
        m = re.match(r'({}) TESLA AT ({}) VOLTS'.format(CryogenicSMS120C._re_float_exp,CryogenicSMS120C._re_float_exp), value)
        field = float(m[1])
        return field

    def _get_rampStatus(self):  # get current magnet status, returns an integer
        _, value = self.query('RAMP STATUS')
        if 'HOLDING' in value:  # holding on
            rampStatus = 0
        elif 'RAMPING' in value:  # magnet ramping
            rampStatus = 1
        elif 'QUENCH' in value:  # detect magnet quench
            rampStatus = 2
        elif 'EXTERNAL' in value:  # detect external trip
            rampStatus = 3
        elif 'FAULT' in value:  # detect either controller or power fault
            rampStatus = 4
        return rampStatus

    # checks if controller is paused (1) or active (0), returns a boolean
    # integer
    def _get_pauseRamp(self):
        _, value = self.query('PAUSE')
        if value == 'ON':
            pauseRamp = 1
        else:  # assume pause OFF
            pauseRamp = 0
        return pauseRamp

    # Get current magnet ramping rate, returns a float (in units of Amps/sec
    # only)
    def _get_rampRate(self):
        _, value = self.query('GET RATE')
        m = re.match(
            r'({}) A/SEC'.format(CryogenicSMS120C._re_float_exp), value)
        rampRate = float(m[1])
        return rampRate

    # Set magnet sweep direction : "+" for positive B, "-" for negative B
    def _set_polarity(self, val):
        # using standard write as read returns an error/is non-existent.
        if self._get_persistentMode() == False and abs(self._get_field()) <= 0.007:
            self.write('DIRECTION %s' % val)
            return True
        elif self._get_persistentMode() == True:
            log.error(
                'Cannot switch polarity, magnet in persistent mode - please engage switch heater and go to zero field before changing sign.')
            return False
        elif abs(self._get_field()) > 0.007:
            log.error('Recommended to switch sign only when field is at zero.')
            return False
        else:
            log.error('Cannot switch polarity, check magnet.')
            return False

    def _set_unit(self, val):        # Set unit to Tesla(1) or Amps(0),
        # Enables us to set units of Tesla
        self.ask('SET TPA %f' % self._coil_constant)
        self.ask('TESLA %d' % val)

    def _set_maxField(self, val):  # Set the maximum field (in Amps or Tesla)
        self.ask('SET MAX %0.2f' % val)

    def _set_switchHeater(self, val):  # Turn heater ON(1) or OFF(0)
        if self._get_rampStatus() == 1:
            log.error(
                'Cannot switch heater during a ramp, first pause the controller.')
        else:
            # Switch ON, if currently OFF
            if val == 1 and (self._get_switchHeater() == False):
                strHeaterStatus = self.ask('HEATER %d' % val)
                switchHeater = 1
            # Switch OFF, if currently ON
            elif val == 0 and (self._get_switchHeater() == True):
                strHeaterStatus = self.ask('HEATER %d' % val)
                switchHeater = 0
            else:  # assume no change to current switch heater state
                strHeaterStatus = self.ask('HEATER %d' % val)
                switchHeater = self._get_switchHeater()
            log.info(strHeaterStatus)
            return switchHeater

    # Move into persistent mode (1) or out of persistent mode(0)
    def _set_persistentMode(self, val):
        if self._get_rampStatus() == 0:     # Check magnet on HOLD
            currField = self._get_field()
            log.info("Leads now at %f ." % currField)
            # Enter persistent mode from non-persistent
            if val == 1:
                if self._get_persistentMode() == False and (self._get_switchHeater() == True):
                    log.info('Moving into persistent mode:')
                    switchHeater = 0
                    strHeaterStatus = self.ask('HEATER %d' % switchHeater)
                    log.info(strHeaterStatus)
                    log.info('Waiting 60s for switch heater to cool down.')
                    time.sleep(60)
                    log.info('Ramping down magnet leads...')
                    self._set_field(0)
                    while True:
                        if self._get_rampStatus() == 0:
                            log.info('Leads at zero.')
                            persistentMode = 1
                            persistentField = self._get_persistentField()
                            log.info(
                                'Magnet is in persistent mode at Field = %f.' % persistentField)
                            break
                        time.sleep(5)  # check every 5 seconds
                elif self._get_persistentMode() == True:
                    persistentMode = 1
                    persistentField = self._get_persistentField()
                    log.info('Already in persistent mode.')
            # Exit persistent mode
            elif val == 0:
                persistentField = self._get_persistentField()
                if self._get_persistentMode() == True and (self._get_switchHeater() == False):
                    log.info('Exiting persistent mode from a field of %f' %
                             persistentField)
                    switchHeater = 1
                    strHeaterStatus = self.ask('HEATER %d' % switchHeater)
                    log.info(strHeaterStatus)
                    log.info('Waiting 30s for switch heater to warm up.')
                    time.sleep(30)
                    log.info(
                        'Matching magnet lead current to persistent field of %f...' % persistentField)
                    self._set_field(persistentField)
                    while True:
                        if self._get_rampStatus() == 0:
                            persistentMode = 0
                            persistentField = 0
                            log.info('Magnet is non-persistent.')
                            break
                        time.sleep(5)  # check every 5 seconds
                elif self._get_persistentMode() == False:
                    persistentMode = 0
                    persistentField = 0
                    log.info('Magnet already non-persistent.')
        else:
            log.warning(
                'Cannot change (non-)persistent mode state, check magnet status.')
        return persistentMode, persistentField

    def _set_pauseRamp(self, val):  # Pause magnet controller Pause=1, Unpause=0
        self.ask('PAUSE %d' % val)

    # Set ramp speed Amps/sec , check it is reasonable if it is being manually
    # modified
    def _set_rampRate(self, val):
        if self._current_ramp_limit is None:
            self._current_ramp_limit = CryogenicSMS120C.default_current_ramp_limit

        if val <= self._current_ramp_limit:
            self.ask('SET RAMP %0.2f' % val)
            return True
        elif val > CryogenicSMS120C.default_max_current_ramp_limit:
            self.ask('SET RAMP %0.2f' %
                     CryogenicSMS120C.default_current_ramp_limit)
            msg = 'Requested rate of {} is unsafe and over the maximum limit of {} A/s. Coerced to default ramp rate.'
            log.error(msg.format(
                val, CryogenicSMS120C.default_max_current_ramp_limit))
            return False
        else:
            msg = 'Requested ramp speed is over the limit of {} A/s. Change limit, at your own risk after consulting the SMS120C manual.'
            log.error(msg.format(self._current_ramp_limit))
            return False

    # Check magnet state and ramp speed to see if it is safe to ramp, returns
    # boolean
    def _can_startRamping(self):
        state = self._get_rampStatus()

        if self._get_rampRate() <= self._current_ramp_limit:
            if state == 2:         # Quench
                log.error(
                    'Magnet quench detected - please check magnet status before ramping.')
                return False
            elif state == 1:       # Ramping
                log.info('Magnet currently ramping.')
                return True
            elif state == 0:       # Holding
                log.info('Magnet currently holding.')
                return True
            log.error(
                'Could not ramp, magnet in state: {}'.format(state))
            return False
        else:
            log.warning(
                'Could not ramp, ramp rate is over the set limit, please lower.')
            return False

    # Between any two commands, there are must be around 200ms waiting time.
    def _set_field(self, val):
        if not self.switchHeater(): # If switch heater is OFF
            log.error('Unable to set field, switch heater is off, persistent mode may be active')
            return
        # check ramp status is OK
        if self._can_startRamping():
            # Check that field is not outside max.field limit
            if (self._get_unit() == 1 and (val <= self._get_maxField())) or (
                    self._get_unit() == 0 and (val <= self._current_rating)):
                # pause the controller if it is currently ramping
                self._set_pauseRamp(1)
                self.ask('SET MID %0.2f' % val)       # Set target field
                self._set_pauseRamp(0)               # Unpause the controller
                # Ramp magnet/field to MID or ZERO (Note: Using standard write
                # as read returns an error/is non-existent).
                if val == 0:
                    self.write('RAMP ZERO')
                    log.info('Ramping magnetic field to zero...')
                else:
                    self.write('RAMP MID')
                    log.info('Ramping magnetic field...')
            else:
                log.error(
                    'Target field is outside max. limits, please lower the target value.')
        else:
            log.error('Cannot set field - check magnet status.')

    def _set_field_bidirectional(self, val):
        polarity = self._get_polarity()
        desired_polarity = '-' if val < 0 else '+'

        if ((polarity == '+' and desired_polarity == '-') or
                (polarity == '-' and desired_polarity == '+')):
            self._set_field(0)
            # This is, sadly, blocking
            self._wait_for_field_zero(0)
            self._set_polarity(desired_polarity)

        self._set_field(abs(val))

    def _wait_for_field_zero(self, field_threshold=0.003, refresh_time=0.1):
        """Waits for the field to be within a certain threshold"""
        while abs(self.field()) > field_threshold:
            time.sleep(refresh_time)
