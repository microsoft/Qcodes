import logging
import numpy as np
import time

from qcodes import Instrument, VisaInstrument
from qcodes.utils.validators import Numbers


def R_y(theta):
    """ Construct rotation matrix around y-axis. """
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])


def R_z(theta):
    """ Construct rotation matrix around z-axis. """
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])


class AMI430(VisaInstrument):
    """
    Driver for the American Magnetics Model 430 magnet power supply programmer.

    This class controls a single magnet power supply. In order to use two or
    three magnets simultaniously to set field vectors, first instantiate the
    individual magnets using this class and then pass them as arguments to
    either the AMI430_2D or AMI430_3D virtual instrument classes.

    Args:
        name (string): a name for the instrument
        address (string): IP address of the power supply programmer
        coil_constant (float): coil constant in Tesla per ampere
        current_rating (float): maximum current rating in ampere
        current_ramp_limit (float): current ramp limit in ampere per second
        persistent_switch (bool): whether this magnet has a persistent switch
    """
    def __init__(self, name, address, coil_constant, current_rating,
                 current_ramp_limit, persistent_switch=True,
                 reset=False, **kwargs):
        super().__init__(name, address, terminator='\n', **kwargs)

        self._parent_instrument = None

        self._coil_constant = coil_constant
        self._current_rating = current_rating
        self._current_ramp_limit = current_ramp_limit
        self._persistent_switch = persistent_switch

        self._field_rating = coil_constant * current_rating
        self._field_ramp_limit = coil_constant * current_ramp_limit

        # Make sure the ramp rate time unit is seconds
        if self.ask('RAMP:RATE:UNITS?') == '1':
            self.write('CONF:RAMP:RATE:UNITS 0')

        # Make sure the field unit is Tesla
        if self.ask('FIELD:UNITS?') == '0':
            self.write('CONF:FIELD:UNITS 1')

        self.add_parameter('field',
                           get_cmd='FIELD:MAG?',
                           get_parser=float,
                           set_cmd=self._set_field,
                           units='T',
                           vals=Numbers(-self._field_rating,
                                        self._field_rating))

        self.add_function('ramp_to',
                          call_cmd=self._ramp_to,
                          args=[Numbers(-self._field_rating,
                                        self._field_rating)])

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
                               set_cmd=self._set_persistent_switch,
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

        self.add_function('get_error', call_cmd='SYST:ERR?')

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

    def _set_field(self, value, *, perform_safety_check=True):
        """
        Blocking method to ramp to a certain field

        Args:
            perform_safety_check (bool): Whether to set the field via a parent
                driver (if present), which might perform additional safety
                checks.
        """
        # If part of a parent driver, set the value using that driver
        if np.abs(value) > self._field_rating:
            msg = ': Aborted _set_field; {} is higher than limit of {}'
            logging.error(__name__ + msg.format(value, self._field_rating))

            return

        if self._parent_instrument is not None and perform_safety_check:
            self._parent_instrument._request_field_change(self, value)

            return

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
        if np.abs(value) > self._field_rating:
            msg = ': Aborted _ramp_to; {} is higher than limit of {}'
            logging.error(__name__ + msg.format(value, self._field_rating))

            return

        if self._parent_instrument is not None:
            msg = (": Initiating a blocking instead of non-blocking "
                   " function because this magnet belongs to a parent driver")
            logging.warning(__name__ + msg)

            self._parent_instrument._request_field_change(self, value)

            return

        if np.abs(value) > self._ramp_rating:
            msg = ': Aborted _ramp_to because setpoint higher than maximum'
            logging.error(__name__ + msg)

            return

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
        """ Return the ramp rate of the first segment in Tesla per second """
        results = self.ask('RAMP:RATE:FIELD:1?').split(',')

        return float(results[0])

    def _set_ramp_rate(self, rate):
        """ Set the ramp rate of the first segment in Tesla per second """
        cmd = 'CONF:RAMP:RATE:FIELD 1,{},{}'.format(rate, self._field_rating)

        self.write(cmd)

    def _set_persistent_switch(self, on):
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

    Args:
        name (string): a name for the instrument
        magnet_x (AMI430): magnet for the x component
        magnet_y (AMI430): magnet for the y component
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
                           vals=Numbers())

        self.add_parameter('angle',
                           get_cmd=self._get_angle,
                           set_cmd=self._set_angle,
                           units='deg',
                           vals=Numbers())

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
        x = self.magnet_x.field()
        y = self.magnet_y.field()

        return np.sqrt(x**2 + y**2)

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
    This driver provides methods that simplify setting fields in
    different coordinate systems.

    Cartesian, spherical and cylindrical coordinates are supported, with
    the following parameters:

    Carthesian:     x,      y,      z
    Spherical:      phi,    theta,  field
    Cylindrical:    phi,    rho,    z

    For the spherical system theta is the polar angle from the positive
    z-axis to the negative z-axis (0 to pi). Phi is the azimuthal angle
    starting at the positive x-axis in the direction of the positive
    y-axis (0 to 2*pi).

    In the cylindrical system phi is identical to that in the spherical
    system, and z is identical to that in the cartesian system.

    All angles are set and returned in units of degrees and are automatically
    phase-wrapped.

    If you want to control the magnets in this virtual driver individually,
    one can set the _parent_instrument parameter of the magnet to None.
    This is done at your own risk, as it skips field strength checks and
    might result in a magnet quench.

    Example of instantiation:

    magnet = AMI430_3D('AMI430_3D',
        AMI430('AMI430_X', '192.168.2.3', 0.0146, 68.53, 0.2),
        AMI430('AMI430_Y', '192.168.2.2', 0.0426, 70.45, 0.05),
        AMI430('AMI430_Z', '192.168.2.1', 0.1107, 81.33, 0.08),
        field_limit=1.0)

    Args:
        name (string): a name for the instrument
        magnet_x (AMI430): magnet driver for the x component
        magnet_y (AMI430): magnet driver for the y component
        magnet_z (AMI430): magnet driver for the z component
    """
    def __init__(self, name, magnet_x, magnet_y, magnet_z, field_limit,
                 **kwargs):
        super().__init__(name, **kwargs)

        # Register this instrument as the parent of the individual magnets
        for m in [magnet_x, magnet_y, magnet_z]:
            m._parent_instrument = self

        self._magnet_x = magnet_x
        self._magnet_y = magnet_y
        self._magnet_z = magnet_z
        # Make this into a parameter?
        self._field_limit = field_limit

        # Internal coordinates
        self._x, self._y, self._z = 0.0, 0.0, 0.0
        self._phi = 0.0
        self._theta = 0.0
        self._field = 0.0
        self._rho = 0.0

        self.add_parameter('x',
                           get_cmd=self._get_x,
                           set_cmd=self._set_x,
                           units='T',
                           vals=Numbers())

        self.add_parameter('y',
                           get_cmd=self._get_y,
                           set_cmd=self._set_y,
                           units='T',
                           vals=Numbers())

        self.add_parameter('z',
                           get_cmd=self._get_z,
                           set_cmd=self._set_z,
                           units='T',
                           vals=Numbers())

        self.add_parameter('phi',
                           get_cmd=self._get_phi,
                           set_cmd=self._set_phi,
                           units='deg',
                           vals=Numbers())

        self.add_parameter('theta',
                           get_cmd=self._get_theta,
                           set_cmd=self._set_theta,
                           units='deg',
                           vals=Numbers())

        self.add_parameter('field',
                           get_cmd=self._get_field,
                           set_cmd=self._set_field,
                           units='T',
                           vals=Numbers())

        self.add_parameter('rho',
                           get_cmd=self._get_rho,
                           set_cmd=self._set_rho,
                           units='T',
                           vals=Numbers())

    def _spherical_to_cartesian(self, phi, theta, r):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        return x, y, z

    def _cylindrical_to_cartesian(self, phi, rho, z):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)

        return x, y, z

    def _cartesian_to_other(self, x, y, z):
        field = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arctan2(y, x)
        # TODO: handle divide by zero?
        theta = np.arccos(z / field)
        rho = np.sqrt(x**2 + y**2)

        return field, phi, theta, rho

    def _update_internal_coords(self):
        field, phi, theta, rho = self._cartesian_to_other(self._x,
                                                          self._y,
                                                          self._z)

        self._field = field
        self._phi = phi
        self._theta = theta
        self._rho = rho

    def _request_field_change(self, magnet, value):
        if magnet is self._magnet_x:
            self.x(value)
        elif magnet is self._magnet_y:
            self.y(value)
        elif magnet is self._magnet_z:
            self.z(value)
        else:
            msg = ': This magnet doesnt belong to its specified parent!'
            logging.error(__name__ + msg)

    def _measure(self):
        """
        Measure the actual (not setpoint) field strengths of all 3 individual
        magnets and calculate the parameters for all 3 coordinate systems.
        """
        x = self._magnet_x.field()
        y = self._magnet_y.field()
        z = self._magnet_z.field()

        field, phi, theta, rho = self._cartesian_to_other(x, y, z)

        coords = {
            'x': x,
            'y': y,
            'z': z,
            'field': field,
            'phi': np.degrees(phi),
            'theta': np.degrees(theta),
            'rho': rho
        }

        return coords

    def _get_x(self):
        return self._measure()['x']

    def _set_x(self, value):
        self._x = value
        self._set_fields(self._x, self._y, self._z)

    def _get_y(self):
        return self._measure()['y']

    def _set_y(self, value):
        self._y = value
        self._set_fields(self._x, self._y, self._z)

    def _get_z(self):
        return self._measure()['z']

    def _set_z(self, value):
        self._z = value
        self._set_fields(self._x, self._y, self._z)

    def _get_phi(self):
        return self._measure()['phi']

    def _set_phi(self, value):
        self._update_internal_coords()
        self._phi = np.radians(value)

        self._x, self._y, self._z = self._spherical_to_cartesian(self._phi,
                                                                 self._theta,
                                                                 self._field)

        self._set_fields(self._x, self._y, self._z)

    def _get_theta(self):
        return self._measure()['theta']

    def _set_theta(self, value):
        self._update_internal_coords()
        self._theta = np.radians(value)

        self._x, self._y, self._z = self._spherical_to_cartesian(self._phi,
                                                                 self._theta,
                                                                 self._field)

        self._set_fields(self._x, self._y, self._z)

    def _get_field(self):
        return self._measure()['field']

    def _set_field(self, field):
        self._update_internal_coords()
        self._field = field

        self._x, self._y, self._z = self._spherical_to_cartesian(self._phi,
                                                                 self._theta,
                                                                 self._field)

        self._set_fields(self._x, self._y, self._z)

    def _get_rho(self):
        return self._measure()['rho']

    def _set_rho(self, value):
        self._update_internal_coords()
        self._rho = value

        self._x, self._y, self._z = self._cylindrical_to_cartesian(self._phi,
                                                                   self._rho,
                                                                   self._z)

        self._set_fields(self._x, self._y, self._z)

    def _set_fields(self, x, y, z):
        # Check if exceeding the field limit
        if np.sqrt(x**2 + y**2 + z**2) > self._field_limit:
            msg = ' _set_fields refused; field would exceed limit of {}'
            logging.error(__name__ + msg.format(self._field_limit))

            return

        swept_x, swept_y, swept_z = False, False, False

        m = self._measure()

        # First ramp the coils that are decreasing in field strength
        if np.abs(m['x']) < np.abs(x):
            self._magnet_x._set_field(x, perform_safety_check=False)
            swept_x = True

        if np.abs(m['y']) < np.abs(y):
            self._magnet_y._set_field(y, perform_safety_check=False)
            swept_y = True

        if np.abs(m['z']) < np.abs(z):
            self._magnet_z._set_field(z, perform_safety_check=False)
            swept_z = True

        # Finally, ramp up the coils that are increasing
        if not swept_x:
            self._magnet_x._set_field(x, perform_safety_check=False)

        if not swept_y:
            self._magnet_y._set_field(y, perform_safety_check=False)

        if not swept_z:
            self._magnet_z._set_field(z, perform_safety_check=False)
