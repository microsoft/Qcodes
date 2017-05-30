import logging
import numpy as np
import time

from qcodes import Instrument, VisaInstrument, IPInstrument
from qcodes.utils.validators import Numbers, Anything

from functools import partial


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


class AMI430(IPInstrument):
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
    def __init__(self, name, address, port, coil_constant, current_rating,
                 current_ramp_limit, persistent_switch=True,
                 reset=False, terminator='\r\n', **kwargs):
        super().__init__(name, address, port, terminator=terminator,
                         write_confirmation=False, **kwargs)

        self._parent_instrument = None

        self._coil_constant = coil_constant
        self._current_rating = current_rating
        self._current_ramp_limit = current_ramp_limit
        self._persistent_switch = persistent_switch

        self._field_rating = coil_constant * current_rating
        self._field_ramp_limit = coil_constant * current_ramp_limit

        # Make sure the ramp rate time unit is seconds
        if int(self.ask('RAMP:RATE:UNITS?')) == 1:
            self.write('CONF:RAMP:RATE:UNITS 0')

        # Make sure the field unit is Tesla
        if int(self.ask('FIELD:UNITS?')) == 0:
            self.write('CONF:FIELD:UNITS 1')

        self.add_parameter('field',
                           get_cmd='FIELD:MAG?',
                           get_parser=float,
                           set_cmd=self._set_field,
                           unit='T',
                           vals=Numbers(-self._field_rating,
                                        self._field_rating))

        self.add_function('ramp_to',
                          call_cmd=self._ramp_to,
                          args=[Numbers(-self._field_rating,
                                        self._field_rating)])

        self.add_parameter('ramp_rate',
                           get_cmd=self._get_ramp_rate,
                           set_cmd=self._set_ramp_rate,
                           unit='T/s',
                           vals=Numbers(0, self._field_ramp_limit))

        self.add_parameter('setpoint',
                           get_cmd='FIELD:TARG?',
                           get_parser=float,
                           unit='T')

        if persistent_switch:
            self.add_parameter('switch_heater_enabled',
                               get_cmd='PS?',
                               set_cmd=self._set_persistent_switch,
                               val_mapping={False: 0, True: 1})

            self.add_parameter('in_persistent_mode',
                               get_cmd='PERS?',
                               val_mapping={False: 0, True: 1})

        self.add_parameter('is_quenched',
                           get_cmd='QU?',
                           val_mapping={False: 0, True: 1})

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
            msg = 'Aborted _set_field; {} is higher than limit of {}'

            raise ValueError(msg.format(value, self._field_rating))

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
                msg = '_set_field({}) failed with state: {}'

                raise Exception(msg.format(value, state))

    def _ramp_to(self, value):
        """ Non-blocking method to ramp to a certain field """
        if np.abs(value) > self._field_rating:
            msg = 'Aborted _ramp_to; {} is higher than limit of {}'

            raise ValueError(msg.format(value, self._field_rating))

        if self._parent_instrument is not None:
            msg = (": Initiating a blocking instead of non-blocking "
                   " function because this magnet belongs to a parent driver")
            logging.warning(__name__ + msg)

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

    def _connect(self):
        """
        Append the IPInstrument connect to flush the welcome message of the AMI
        430 programmer
        :return: None
        """
        super()._connect()
        print(self._recv())

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
                           unit='deg',
                           vals=Numbers())

        self.add_parameter('angle',
                           get_cmd=self._get_angle,
                           set_cmd=self._set_angle,
                           unit='deg',
                           vals=Numbers())

        self.add_parameter('field',
                           get_cmd=self._get_field,
                           set_cmd=self._set_field,
                           unit='T',
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

        # Initialize the internal magnet field setpoints
        self.update_internal_setpoints()

        # Get-only parameters that return a measured value
        self.add_parameter('cartesian_measured',
                           get_cmd=partial(self._get_measured, 'x', 'y', 'z'),
                           unit='T')

        self.add_parameter('x_measured',
                           get_cmd=partial(self._get_measured, 'x'),
                           unit='T')

        self.add_parameter('y_measured',
                           get_cmd=partial(self._get_measured, 'y'),
                           unit='T')

        self.add_parameter('z_measured',
                           get_cmd=partial(self._get_measured, 'z'),
                           unit='T')

        self.add_parameter('spherical_measured',
                           get_cmd=partial(self._get_measured, 'field',
                                                               'theta',
                                                               'phi'),
                           unit='T')

        self.add_parameter('phi_measured',
                           get_cmd=partial(self._get_measured, 'phi'),
                           unit='deg')

        self.add_parameter('theta_measured',
                           get_cmd=partial(self._get_measured, 'theta'),
                           unit='deg')

        self.add_parameter('field_measured',
                           get_cmd=partial(self._get_measured, 'field'),
                           unit='T')

        self.add_parameter('cylindrical_measured',
                           get_cmd=partial(self._get_measured, 'rho',
                                                               'phi',
                                                               'z'),
                           unit='T')

        self.add_parameter('rho_measured',
                           get_cmd=partial(self._get_measured, 'rho'),
                           unit='T')

        # Get and set parameters for the setpoints of the coordinates
        self.add_parameter('cartesian',
                           get_cmd=partial(self._get_setpoints, 'x', 'y', 'z'),
                           set_cmd=self._set_fields,
                           unit='T',
                           vals=Anything())

        self.add_parameter('x',
                           get_cmd=partial(self._get_setpoints, 'x'),
                           set_cmd=self._set_x,
                           unit='T',
                           vals=Numbers())

        self.add_parameter('y',
                           get_cmd=partial(self._get_setpoints, 'y'),
                           set_cmd=self._set_y,
                           unit='T',
                           vals=Numbers())

        self.add_parameter('z',
                           get_cmd=partial(self._get_setpoints, 'z'),
                           set_cmd=self._set_z,
                           unit='T',
                           vals=Numbers())

        self.add_parameter('spherical',
                           get_cmd=partial(self._get_setpoints, 'field',
                                                                'theta',
                                                                'phi'),
                           set_cmd=self._set_spherical,
                           unit='tuple?',
                           vals=Anything())

        self.add_parameter('phi',
                           get_cmd=partial(self._get_setpoints, 'phi'),
                           set_cmd=self._set_phi,
                           unit='deg',
                           vals=Numbers())

        self.add_parameter('theta',
                           get_cmd=partial(self._get_setpoints, 'theta'),
                           set_cmd=self._set_theta,
                           unit='deg',
                           vals=Numbers())

        self.add_parameter('field',
                           get_cmd=partial(self._get_setpoints, 'field'),
                           set_cmd=self._set_field,
                           unit='T',
                           vals=Numbers())

        self.add_parameter('cylindrical',
                           get_cmd=partial(self._get_setpoints, 'rho',
                                                                'phi',
                                                                'z'),
                           set_cmd=self._set_cylindrical,
                           unit='tuple?',
                           vals=Anything())

        self.add_parameter('rho',
                           get_cmd=partial(self._get_setpoints, 'rho'),
                           set_cmd=self._set_rho,
                           unit='T',
                           vals=Numbers())

    def update_internal_setpoints(self):
        """
        Set the internal setpoints to the measured field values.
        This can be done in case the magnets have been adjusted manually.
        """
        self.__x = self._magnet_x.field()
        self.__y = self._magnet_y.field()
        self.__z = self._magnet_z.field()

    def _request_field_change(self, magnet, value):
        """
        This method is called by the child x/y/z magnets if they are set
        individually. It results in additional safety checks being
        performed by this 3D driver.
        """
        if magnet is self._magnet_x:
            self.x(value)
        elif magnet is self._magnet_y:
            self.y(value)
        elif magnet is self._magnet_z:
            self.z(value)
        else:
            msg = 'This magnet doesnt belong to its specified parent {}'

            raise NameError(msg.format(self))

    def _cartesian_to_other(self, x, y, z):
        """ Convert a cartesian set of coordinates to values of interest. """
        field = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arctan2(y, x)
        rho = np.sqrt(x**2 + y**2)

        # Define theta to be 0 for zero field
        theta = 0.0
        if field > 0.0:
            theta = np.arccos(z / field)

        return phi, theta, field, rho

    def _from_xyz(self, x, y, z, *names):
        """
        Convert x/y/z values into the other coordinates and return a
        tuple of the requested values.

        Args:
            *names: a series of coordinate names as specified in the function.
        """
        phi, theta, field, rho = self._cartesian_to_other(x, y, z)

        coords = {
            'x': x,
            'y': y,
            'z': z,
            'field': field,
            'phi': np.degrees(phi),
            'theta': np.degrees(theta),
            'rho': rho
        }

        returned = tuple(coords[name] for name in names)

        if len(returned) == 1:
            return returned[0]
        else:
            return returned

    def _get_measured(self, *names):
        """ Return the measured coordinates specified in names. """
        x = self._magnet_x.field()
        y = self._magnet_y.field()
        z = self._magnet_z.field()

        return self._from_xyz(x, y, z, *names)

    def _get_setpoints(self, *names):
        """ Return the setpoints specified in names. """
        return self._from_xyz(self.__x, self.__y, self.__z, *names)

    def _set_x(self, value):
        self._set_fields((value, self.__y, self.__z))

    def _set_y(self, value):
        self._set_fields((self.__x, value, self.__z))

    def _set_z(self, value):
        self._set_fields((self.__x, self.__y, value))

    def _set_spherical(self, values):
        field, theta, phi = values

        phi, theta = np.radians(phi), np.radians(theta)

        x = field * np.sin(theta) * np.cos(phi)
        y = field * np.sin(theta) * np.sin(phi)
        z = field * np.cos(theta)

        self._set_fields((x, y, z))

    def _set_phi(self, value):
        field, theta, phi = self._get_setpoints('field', 'theta', 'phi')

        phi = np.radians(value)

        self._set_spherical((field, theta, phi))

    def _set_theta(self, value):
        field, theta, phi = self._get_setpoints('field', 'theta', 'phi')

        theta = np.radians(value)

        self._set_spherical((field, theta, phi))

    def _set_field(self, value):
        field, theta, phi = self._get_setpoints('field', 'theta', 'phi')

        field = value

        self._set_spherical((field, theta, phi))

    def _set_cylindrical(self, values):
        phi, rho, z = values

        phi = np.radians(phi)

        x = rho * np.cos(phi)
        y = rho * np.sin(phi)

        self._set_fields((x, y, z))

    def _set_rho(self, value):
        phi, rho = self._get_setpoints('phi', 'rho')

        rho = value

        self._set_cylindrical((phi, rho, self.__z))

    def _set_fields(self, values):
        """
        Set the fields of the x/y/z magnets. This function is called
        whenever the field is changed and performs several safety checks
        to make sure no limits are exceeded.

        Args:
            values (tuple): a tuple of cartesian coordinates (x, y, z).
        """
        x, y, z = values

        # Check if exceeding an individual magnet field limit
        # These will throw a ValueError on an invalid value
        self._magnet_x.field.validate(x)
        self._magnet_y.field.validate(y)
        self._magnet_z.field.validate(z)

        # Check if exceeding the global field limit
        if np.sqrt(x**2 + y**2 + z**2) > self._field_limit:
            msg = '_set_fields aborted; field would exceed limit of {} T'

            raise ValueError(msg.format(self._field_limit))

        # Check if the individual magnet are not already ramping
        for m in [self._magnet_x, self._magnet_y, self._magnet_z]:
            if m.ramping_state() == 'ramping':
                msg = '_set_fields aborted; magnet {} is already ramping'

                raise ValueError(msg.format(m))

        swept_x, swept_y, swept_z = False, False, False

        # First ramp the coils that are decreasing in field strength
        # If the new setpoint is practically equal to the current one
        # then leave it be
        if np.isclose(self.__x, x, rtol=0, atol=1e-8):
            swept_x = True
        elif np.abs(self._magnet_x.field()) > np.abs(x):
            self._magnet_x._set_field(x, perform_safety_check=False)
            swept_x = True

        if np.isclose(self.__y, y, rtol=0, atol=1e-8):
            swept_y = True
        elif np.abs(self._magnet_y.field()) > np.abs(y):
            self._magnet_y._set_field(y, perform_safety_check=False)
            swept_y = True

        if np.isclose(self.__z, z, rtol=0, atol=1e-8):
            swept_z = True
        elif np.abs(self._magnet_z.field()) > np.abs(z):
            self._magnet_z._set_field(z, perform_safety_check=False)
            swept_z = True

        # Finally, ramp up the coils that are increasing
        if not swept_x:
            self._magnet_x._set_field(x, perform_safety_check=False)

        if not swept_y:
            self._magnet_y._set_field(y, perform_safety_check=False)

        if not swept_z:
            self._magnet_z._set_field(z, perform_safety_check=False)

        # Set the new actual setpoints
        self.__x = x
        self.__y = y
        self.__z = z
