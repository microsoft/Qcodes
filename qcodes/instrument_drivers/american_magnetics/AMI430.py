import logging
import numpy as np
import time

from qcodes import Instrument, VisaInstrument
from qcodes.utils.validators import Numbers


class AMI430(VisaInstrument):
    """
    Driver for the American Magnetics Model 430 magnet power supply programmer
    """
    def __init__(self, name, address, coil_constant, current_rating,
                 current_ramp_limit, persistent_switch=True, terminator='\n',
                 reset=False, **kwargs):
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
            logging.error(__name__ + ': Could not ramp because of quench')
            return False

        if self._persistent_switch and self.in_persistent_mode():
            logging.error(__name__ + ': Could not ramp because persistent')
            return False

        state = self.ramping_state()

        if state == 'ramping':
            # If we don't have a persistent switch, or it's heating: OK to ramp
            if not self._persistent_switch or self.switch_heater_enabled():
                return True
        elif state in ['holding', 'paused', 'at zero current']:
            return True

        logging.error(__name__ + ': Could not ramp, state: {}'.format(state))

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

            state = self.ramping_state()

            # If we are now holding, it was succesful
            if state == 'holding':
                self.pause()
            else:
                msg = ': _set_field({}) failed with state: {}'
                logging.error(__name__ + msg.format(value, state))

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

        self._angle_offset = 0.0
        self._angle = 0.0
        self._field = 0.0

        self.add_parameter('angle_offset',
                           get_cmd=self._get_angle_offset,
                           set_cmd=self._set_angle_offset,
                           units='deg',
                           vals=Numbers(0, 360))

        self.add_parameter('angle',
                           get_cmd=self._get_angle,
                           set_cmd=self._set_angle,
                           units='deg',
                           vals=Numbers(0, 360))

        self.add_parameter('field',
                           get_cmd=self._get_field,
                           set_cmd=self._set_field,
                           units='T',
                           vals=Numbers())

    def _get_angle_offset(self):
        return np.degrees(self._angle_offset)

    def _set_angle_offset(self, angle):
        # Adjust the field if the offset angle is changed
        if self._angle_offset != np.radians(angle):
            self._angle_offset = np.radians(angle)
            self._set_field(self._field)

    def _get_angle(self):
        angle = np.arctan2(self.magnet_y.field(), self.magnet_x.field())

        return np.degrees(angle - self._angle_offset)

    def _set_angle(self, angle):
        self._angle = np.radians(angle)

        self._set_field(self._field)

    def _get_field(self):
        return np.hypot(self.magnet_x.field(), self.magnet_y.field())

    def _set_field(self, field):
        self._field = field

        B_x = field * np.cos(self._angle + self._angle_offset)
        B_y = field * np.sin(self._angle + self._angle_offset)

        # First ramp the magnet that is decreasing in field strength
        if np.abs(self.magnet_x.field()) < np.abs(B_x):
            self.magnet_x.field(B_x)
            self.magnet_y.field(B_y)
        else:
            self.magnet_y.field(B_y)
            self.magnet_x.field(B_x)


class AMI430_3D(Instrument):
    """
    Virtual driver for a system of three AMI430 magnet power supplies.

    This driver provides methods that simplify setting fields as vectors.
    """
    def __init__(self, name, magnet_x, magnet_y, magnet_z, **kwargs):
        super().__init__(name, **kwargs)

        self.magnet_x, self.magnet_y, self.magnet_z = magnet_x, magnet_y, magnet_z

        self._phi,   self._phi_offset = 0.0, 0.0
        self._theta, self._theta_offset = 0.0, 0.0
        self._field = 0.0

        self.add_parameter('phi',
                           get_cmd=self._get_phi,
                           set_cmd=self._set_phi,
                           units='deg',
                           vals=Numbers(0, 360))

        self.add_parameter('phi_offset',
                           get_cmd=self._get_phi_offset,
                           set_cmd=self._set_phi_offset,
                           units='deg',
                           vals=Numbers(0, 360))

        self.add_parameter('theta',
                           get_cmd=self._get_theta,
                           set_cmd=self._set_theta,
                           units='deg',
                           vals=Numbers(0, 360))

        self.add_parameter('theta_offset',
                           get_cmd=self._get_theta_offset,
                           set_cmd=self._set_theta_offset,
                           units='deg',
                           vals=Numbers(0, 360))

        self.add_parameter('field',
                           get_cmd=self._get_field,
                           set_cmd=self._set_field,
                           units='T',
                           vals=Numbers())

    def _get_phi(self):
        angle = np.arctan2(self.magnet_y.field(), self.magnet_x.field())

        return np.degrees(angle - self._phi_offset)

    def _set_phi(self, phi):
        self._phi = np.radians(phi)

        self._set_field(self._field)

    def _get_phi_offset(self):
        return np.degrees(self._phi_offset)

    def _set_phi_offset(self, phi):
        # Adjust the field if the offset phi is changed
        if self._phi_offset != np.radians(phi):
            self._angle_offset = np.radians(phi)

            self._set_field(self._field)

    def _get_theta(self):
        return np.arccos(self.magnet_z.field() / self._get_field())

    def _set_theta(self, theta):
        self._theta = np.radians(theta)

        self._set_field(self._field)

    def _get_theta_offset(self):
        return np.degrees(self._theta_offset)

    def _set_theta_offset(self, theta):
        # Adjust the field if the offset theta is changed
        if self._theta_offset != np.radians(theta):
            self._angle_offset = np.radians(theta)

            self._set_field(self._field)

    def _get_field(self):
        x, y, z = self.magnet_x.field(), self.magnet_y.field(), self.magnet_z.field()

        return np.sqrt(x**2 + y**2 + z**2)

    def _set_field(self, field):
        self._field = field

        phi = self._phi + self._phi_offset
        theta = self._theta + self._theta_offset

        B_x = field * np.sin(theta) * np.cos(phi)
        B_y = field * np.sin(theta) * np.sin(phi)
        B_z = field * np.cos(theta)

        swept_x, swept_y, swept_z = False, False, False

        # First ramp the coils that are decreasing in field strength
        if np.abs(self.magnet_x.field()) < np.abs(B_x):
            self.magnet_x.field(B_x)
            swept_x = True

        if np.abs(self.magnet_y.field()) < np.abs(B_y):
            self.magnet_y.field(B_y)
            swept_y = True

        if np.abs(self.magnet_z.field()) < np.abs(B_z):
            self.magnet_z.field(B_z)
            swept_z = True

        # Finally, ramp up the coils that are increasing
        if not swept_x:
            self.magnet_x.field(B_x)

        if not swept_y:
            self.magnet_y.field(B_y)

        if not swept_z:
            self.magnet_z.field(B_z)
