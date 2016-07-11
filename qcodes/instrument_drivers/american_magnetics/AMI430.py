import time

import numpy as np

from qcodes import Instrument, VisaInstrument
from qcodes.utils.validators import Numbers, Ints, Enum, MultiType

class AMI430(VisaInstrument):
    """
    Driver for the American Magnetics Model 430 magnet power supply programmer
    """
    def __init__(self, name, address,
                 coil_constant, current_rating, current_ramp_limit, persistent_switch=True,
                 terminator='\n', reset=False, **kwargs):
        super().__init__(name, address, **kwargs)

        self._coil_constant = coil_constant
        self._current_rating = current_rating
        self._current_ramp_limit = current_ramp_limit
        self._persistent_switch = persistent_switch

        self._field_rating = coil_constant * current_rating
        self._field_ramp_limit = coil_constant * current_ramp_limit

        # Make sure the ramp rate time unit is seconds
        if self.ask('RAMP:RATE:UNITS') == '1':
            self.write('CONF:RAMP:RATE:UNITS 0')

        # Make sure the field unit is Tesla
        if self.ask('FIELD:UNITS?') == '0':
            self.write('CONF:FIELD:UNITS 1')

        self.add_parameter('field',
                           get_cmd='FIELD:MAG?',
                           get_parser=float,
                           set_cmd=self._set_field,
                           units='T',
                           vals=Numbers(-self._field_rating, self._field_rating))

        self.add_function('ramp_to',
                          call_cmd=self._ramp_to,
                          args=[Numbers(-self._field_rating, self._field_rating)])

        self.add_parameter('ramp_rate',
                           get_cmd=self._get_ramp_rate,
                           set_cmd=self._set_ramp_rate,
                           units='T/s',
                           vals=Numbers(0, self._field_ramp_limit))

        self.add_parameter('setpoint',
                           get_cmd='FIELD:TARG?',
                           get_parser=float,
                           units='T')

        if persistent_switch:
            self.add_parameter('switch_heater_enabled',
                               get_cmd='PS?',
                               set_cmd=self._set_switch_heater,
                               val_mapping={False: '0', True: '1'})

            self.add_parameter('in_persistent_mode',
                               get_cmd='PERS?',
                               val_mapping={False: '0', True: '1'})

        self.add_parameter('is_quenched',
                           get_cmd='QU?',
                           val_mapping={False: '0', True: '1'})

        self.add_function('reset_quench', call_cmd='QU 0')
        self.add_function('set_quenched', call_cmd='QU 1')

        self.add_parameter('ramping_state',
                           get_cmd='STATE?',
                           val_mapping={
                               'ramping': 1,
                               'holding': 2,
                               'paused': 3,
                               'manual up': 4,
                               'manual down': 5,
                               'zeroing current': 6,
                               'quench detected': 7,
                               'at zero current': 8,
                               'heating switch': 9,
                               'cooling switch': 10,
                           })

        self.add_function('get_error', get_cmd='SYST:ERR?')

        self.add_function('ramp', call_cmd='RAMP')
        self.add_function('pause', call_cmd='PAUSE')
        self.add_function('zero', call_cmd='ZERO')

        self.add_function('reset', call_cmd='*RST')

        if reset:
            self.reset()

        self.connect_message()

    def _can_start_ramping(self):
        """
        Check the current state of the magnet to see if we can start ramping
        """
        if self.is_quenched():
            return False

        if self._persistent_switch and self.in_persistent_mode():
            return False

        state = self.ramping_state()

        if state == 'ramping':
            # If we don't have a persistent switch, or it's heating: OK to ramp
            if not self._persistent_switch or self.switch_heater_enabled():
                return True
        elif state in ['holding', 'paused', 'at zero current']:
            return True

        return False

    def _set_field(self, value):
        """ BLocking method to ramp to a certain field """
        if self._can_start_ramping():
            self.pause()

            # Set the ramp target
            self.write('CONF:FIELD:TARG {}'.format(value))

            # If we have a persistent switch, make sure it is resistive
            if self._persistent_switch:
                if not self.switch_heater_enabled():
                    self.switch_heater_enabled(True)

            self.ramp()

            time.sleep(0.5)

            # Wait until no longer ramping
            while self.ramping_state() == 'ramping':
                time.sleep(0.3)

            time.sleep(2.0)

            # If we are now holding, it was succesful
            if self.ramping_state() == 'holding':
                self.pause()
            else:
                pass # ramp ended

    def _ramp_to(self, value):
        """ Non-blocking method to ramp to a certain field """
        if self._can_start_ramping():
            self.pause()

            # Set the ramp target
            self.write('CONF:FIELD:TARG {}'.format(value))

            # If we have a persistent switch, make sure it is resistive
            if self._persistent_switch:
                if not self.switch_heater_enabled():
                    self.switch_heater_enabled(True)

            self.ramp()

    def _get_ramp_rate(self):
        results = self.ask('RAMP:RATE:FIELD:1?').split(',')

        return float(results[0])

    def _set_ramp_rate(self, rate):
        cmd = 'CONF:RAMP:RATE:FIELD 1,{},{}'.format(rate, self._field_rating)

        self.write(cmd)

    def _set_persistent_switch_heater(self, on):
        """
        Blocking function that sets the persistent switch heater state and
        waits until it has finished either heating or cooling

        on: False/True
        """
        if on:
            self.write('PS 1')

            time.sleep(0.5)

            # Wait until heating is finished
            while self.ramping_state() == 'heating switch':
                time.sleep(0.3)
        else:
            self.write('PS 0')

            time.sleep(0.5)

            # Wait until cooling is finished
            while self.ramping_state() == 'cooling switch':
                time.sleep(0.3)

class AMI430_2D(Instrument):
    """
    Virtual driver for a system of two AMI430 magnet power supplies.

    This driver provides methods that simplify setting fields as vectors.
    """
    def __init__(self, name, magnet_x, magnet_y, **kwargs):
        super().__init__(name, **kwargs)

        self.magnet_x, self.magnet_y = magnet_x, magnet_y

        self._alpha = 0.0
        self._field = 0.0

        self.add_parameter('alpha',
                           get_cmd=self._get_alpha,
                           set_cmd=self._set_alpha,
                           units='deg',
                           vals=Numbers(0, 360))

        self.add_parameter('field',
                           get_cmd=self._get_field,
                           set_cmd=self._set_field,
                           units='T',
                           vals=Numbers())

    def _get_alpha(self):
        return np.arctan2(self.magnet_y.field(), self.magnet_x.field())

    def _set_alpha(self, alpha):
        self._alpha = alpha

        self._set_field(self._field)

    def _get_field(self):
        return np.hypot(self.magnet_x.field(), self.magnet_y.field())

    def _set_field(self, field):
        self._field = field

        B_x = field * np.cos(self._alpha)
        B_y = field * np.sin(self._alpha)

        # First ramp the magnet that is decreasing in field strength
        if np.abs(self.magnet_x.field()) < np.abs(B_x):
            self.magnet_x.field(B_x)
            self.magnet_y.field(B_y)
        else:
            self.magnet_y.field(B_y)
            self.magnet_x.field(B_x)

