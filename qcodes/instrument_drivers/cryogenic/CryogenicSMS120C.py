# Cryomagnetics_SMS120C, Cryomagnetics_SMS120C magnet power supply driver

"""
# Created on Fri 29 Nov 2017
# @author: lyeoh

# Last modified by lyeoh 27/10/2017 
# Special thanks to cjvandiepen, acorna, pteendebak.

---
# Please refer to Cryogenic's magnet PS manual for further details and more functionality
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

from qcodes.utils.validators import Numbers, Anything
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
        persistent_mode (bool): check if magnet is in persistent mode (True/False)
        timing : SMS120C needs a minimum of 200ms delay between commands being sent
    """

    # Reg. exp. to match a float or exponent in a string
    _re_float_exp = '[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?'

    def __init__(self, name, address, coil_constant=0.113375, current_rating=105.84,
                 current_ramp_limit=0.0506,
                 reset=False, timeout=5, terminator='\r\n', **kwargs):

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
                           val_mapping={False: 0, True: 1})

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
        """
        Overwrites the get_idn function using constants as the hardware
        does not have a proper \*IDN function.
        """
        idparts = ['Cryogenic', 'Magnet PS SMS120C', 'None', '1.0']

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))

    def query(self, msg):
        """
        Message outputs do not follow the standard SCPI format, 
        separate regexp to parse unique/variable instrument message structures.
        """
        value = self.ask(msg)
        m = re.match(r'((\S{8})\s)+(([^:]+)(:([^:]+))?)', value)
        if m:
            if m[2] == '------->':
                log.error(
                    'Command information or unrecognizable qualifier: "%s"' % m[3])
            else:
                return m[4].strip(), m[6].strip()
        else:
            log.error(
                'Malformed message received from the magnet PS: "%s"' % value)

    def _get_limit(self):  # Get voltage limits, returns a float
        key, value = self.query('GET VL')
        # extract number from string
        m = re.match(r'({}) VOLTS'.format(
            CryogenicSMS120C._re_float_exp), value)
        limit = float(m[1])
        return limit

    # get heater status, returns a boolean ON (1) or OFF (0)
    def _get_switchHeater(self):
        key, value = self.query('HEATER')
        if 'OFF' in value:
            switchHeater = 0
        elif 'ON' in value:
            switchHeater = 1
        return switchHeater

    # check if magnet is in persistent mode, and if so return current in the
    # magnet
    def _get_persistentMode(self):
        key, value = self.query('HEATER')
        field = self._get_field()
        # check for switch heater OFF, and non-zero current
        if 'OFF' in value and (field != 0):
            units = self._get_unit()
            if units == 1:
                print("Magnet in persistent mode, at a field of %f T" % field)
            elif units == 0:
                print("Magnet in persistent mode, at a field of %f A" % field)
            persistent_Mode = True
        else:
            print("Magnet not persistent.")
            persistent_Mode = False
        return persistent_Mode

    def _get_unit(self):  # get units, returns a boolean integer - Tesla (1) or Amps(0)
        key, value = self.query('TESLA')
        if value == 'TESLA':
            unit = 1
        else:  # assume in Amps
            unit = 0
        return unit

    # get direction of current, returns a string - Positive (1) or Negative(0)
    def _get_polarity(self):
        key, value = self.query('GET SIGN')
        if value == 'POSITIVE':
            polarity = '+'
        elif value == 'NEGATIVE':  # assume Negative
            polarity = '-'
        return polarity

    def _get_maxField(self):  # Get the maximum B field, returns a float (in Amps or Tesla)
        key, value = self.query('GET MAX')
        units = self._get_unit()
        if units == 1:
            m = re.match(r'({}) TESLA'.format(
                CryogenicSMS120C._re_float_exp), value)
        elif units == 0:
            m = re.match(r'({}) AMPS'.format(
                CryogenicSMS120C._re_float_exp), value)
        maxField = float(m[1])
        return maxField

    # Get current magnetic field, returns a float (in Amps or Tesla)
    def _get_field(self):
        key, value = self.query('GET OUTPUT')
        units = self._get_unit()
        if units == 1:
            m = re.match(r'({}) TESLA AT ({}) VOLTS'.format(CryogenicSMS120C._re_float_exp,
                                                            CryogenicSMS120C._re_float_exp), value)
        elif units == 0:
            m = re.match(r'({}) AMPS AT ({}) VOLTS'.format(CryogenicSMS120C._re_float_exp,
                                                           CryogenicSMS120C._re_float_exp), value)
        field = float(m[1])
        return field

    def _get_rampStatus(self):  # get current magnet status, returns an integer
        key, value = self.query('RAMP STATUS')
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
        key, value = self.query('PAUSE')
        if value == 'ON':
            pauseRamp = 1
        else:  # assume pause OFF
            pauseRamp = 0
        return pauseRamp

    # Get current magnet ramping rate, returns a float (in units of Amps/sec
    # only)
    def _get_rampRate(self):
        key, value = self.query('GET RATE')
        m = re.match(
            r'({}) A/SEC'.format(CryogenicSMS120C._re_float_exp), value)
        rampRate = float(m[1])
        return rampRate

    # Set magnet sweep direction : "+" for positive B, "-" for negative B
    def _set_polarity(self, val):
        # using standard write as read returns an error/is non-existent.
        self.write('DIRECTION %s' % val)

    def _set_unit(self, val):        # Set unit to Tesla(1) or Amps(0),
        # Enables us to set units of Tesla
        self.ask('SET TPA %f' % self._coil_constant)
        self.ask('TESLA %d' % val)

    def _set_maxField(self, val):  # Set the maximum field (in Amps or Tesla)
        self.ask('SET MAX %0.2f' % val)

    def _set_switchHeater(self, val):  # Turn heater ON(1) or OFF(0)
        if self._get_rampStatus() == 1:
            log.error('Cannot switch heater during a ramp, first pause the ramp.')
        else:
            # Switch ON, if currently OFF
            if val == 1 and (self._get_switchHeater() == False):
                strHeaterStatus = self.ask('HEATER %d' % val)
                switchHeater = 1
                print('Waiting 30s for switch heater to warm up.')
                time.sleep(30)  # wait for magnet to settle after switch
            # Switch OFF, if currently ON
            elif val == 0 and (self._get_switchHeater() == True):
                strHeaterStatus = self.ask('HEATER %d' % val)
                switchHeater = 0
                if self._get_field != 0:  # condition for persistent mode
                    self.persistent_mode = 1
                    print('Waiting 60s for switch heater to cool.')
                    # wait for magnet to settle into persistent mode after
                    # switch
                    time.sleep(60)
                else:
                    print('Waiting 30s for switch heater to cool.')
                    time.sleep(30)  # wait for magnet to settle after switch
            else:  # assume no change to current switch heater state
                strHeaterStatus = self.ask('HEATER %d' % val)
                log.info(strHeaterStatus)
            return switchHeater

    def _set_pauseRamp(self, val):  # Pause magnet controller Pause=1, Unpause=0
        self.ask('PAUSE %d' % val)

    # Set ramp speed Amps/sec , check it is reasonable if it is being manually
    # modified
    def _set_rampRate(self, val):
        if self._current_ramp_limit == None:
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
                    __name__ + ': Magnet quench detected - please check magnet status before ramping.')
                return False
            elif state == 1:       # Ramping
                if self._get_switchHeater() == 1:
                    print('Magnet currently ramping.')
                    return True
                else:
                    log.error(
                        __name__ + ': Magnet in unusual state - ramping with switch heater off, please check magnet status before ramping.')
                    return False
            elif state == 0:       # Holding
                if (not self._get_persistentMode()):
                    print('Magnet currently holding.')
                    return True
                else:
                    print('Magnet in persistent mode.')
                    return True
            log.error(
                __name__ + ': Could not ramp, magnet in state: {}'.format(state))
            return False
        else:
            log.warning(
                __name__ + ': Could not ramp, ramp rate is over the set limit, please lower.')
            return False

    # Between any two commands, there are must be around 200ms waiting time.
    def _set_field(self, val):
        # check ramp status is OK
        if self._can_startRamping():

            # Check that field is not outside max.field limit
            if (self._get_unit() == 1 and (val <= self._get_maxField())) or (self._get_unit() == 0 and (val <= self._current_rating)):
                # pause the controller if it is currently ramping
                self._set_pauseRamp(1)
                if self._get_switchHeater() == 0:   # set switch heater if not already ON
                    self._set_switchHeater(1)
                self.ask('SET MID %0.2f' % val)       # Set target field
                self._set_pauseRamp(0)               # Unpause the controller
                # Ramp magnet/field to MID (Note: Using standard write as read
                # returns an error/is non-existent).
                self.write('RAMP MID')
                print('Ramping magnetic field...')
            else:
                log.error(
                    'Target field is outside max. limits, please lower the target value.')
        else:
            log.error('Cannot set field - check magnet status.')
