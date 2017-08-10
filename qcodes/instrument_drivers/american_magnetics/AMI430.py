import logging
import numpy as np
import time

from functools import partial

from qcodes import Instrument, VisaInstrument, IPInstrument
from qcodes.utils.validators import Numbers, Anything
from qcodes.math.field_vector import FieldVector


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

    def set_field(self, value, *, perform_safety_check=True):
        self._set_field(value, perform_safety_check=perform_safety_check)

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


class AMI430_3D(Instrument):
    def __init__(self, name, instrument_x, instrument_y, instrument_z, field_limit, **kwargs):
        super().__init__(name, **kwargs)

        self._instrument_x = instrument_x
        self._instrument_y = instrument_y
        self._instrument_z = instrument_z

        self._field_limit = field_limit
        self._set_point = FieldVector(
            x=self._instrument_x.field(),
            y=self._instrument_y.field(),
            z=self._instrument_z.field()
        )

        # Get-only parameters that return a measured value
        self.add_parameter(
            'cartesian_measured',
            get_cmd=partial(self._get_measured, 'x', 'y', 'z'),
            unit='T'
        )

        self.add_parameter(
            'x_measured',
            get_cmd=partial(self._get_measured, 'x'),
            unit='T'
        )

        self.add_parameter(
            'y_measured',
            get_cmd=partial(self._get_measured, 'y'),
            unit='T'
        )

        self.add_parameter(
            'z_measured',
            get_cmd=partial(self._get_measured, 'z'),
            unit='T'
        )

        self.add_parameter(
            'spherical_measured',
            get_cmd=partial(
                self._get_measured,
                'field',
                'theta',
                'phi'
            ),
            unit='T'
        )

        self.add_parameter(
            'phi_measured',
            get_cmd=partial(self._get_measured, 'phi'),
            unit='deg'
        )

        self.add_parameter(
            'theta_measured',
            get_cmd=partial(self._get_measured, 'theta'),
            unit='deg'
        )

        self.add_parameter(
            'field_measured',
            get_cmd=partial(self._get_measured, 'r'),
            unit='T')

        self.add_parameter(
            'cylindrical_measured',
            get_cmd=partial(self._get_measured,
                            'rho',
                            'phi',
                            'z'),
            unit='T')

        self.add_parameter(
            'rho_measured',
            get_cmd=partial(self._get_measured, 'rho'),
            unit='T'
        )

        # Get and set parameters for the set points of the coordinates
        self.add_parameter(
            'cartesian',
            get_cmd=partial(self._get_setpoints, 'x', 'y', 'z'),
            set_cmd=self._set_fields,
            unit='T',
            vals=Anything()
        )

        self.add_parameter(
            'x',
            get_cmd=partial(self._get_setpoints, 'x'),
            set_cmd=self.set_x,
            unit='T',
            vals=Numbers()
        )

        self.add_parameter(
            'y',
            get_cmd=partial(self._get_setpoints, 'y'),
            set_cmd=self.set_y,
            unit='T',
            vals=Numbers()
        )

        self.add_parameter(
            'z',
            get_cmd=partial(self._get_setpoints, 'z'),
            set_cmd=self.set_z,
            unit='T',
            vals=Numbers()
        )

        self.add_parameter(
            'spherical',
            get_cmd=partial(
                self._get_setpoints,
                'r',
                'theta',
                'phi'
            ),
            set_cmd=self.set_spherical,
            unit='tuple?',
            vals=Anything()
        )

        self.add_parameter(
            'phi',
            get_cmd=partial(self.get_setpoints, 'phi'),
            set_cmd=self.set_phi,
            unit='deg',
            vals=Numbers()
        )

        self.add_parameter(
            'theta',
            get_cmd=partial(self.get_setpoints, 'theta'),
            set_cmd=self.set_theta,
            unit='deg',
            vals=Numbers()
        )

        self.add_parameter(
            'field',
            get_cmd=partial(self.get_setpoints, 'field'),
            set_cmd=self.set_r,
            unit='T',
            vals=Numbers()
        )

        self.add_parameter(
            'cylindrical',
            get_cmd=partial(
                self.get_setpoints,
                'rho',
                'phi',
                'z'
            ),
            set_cmd=self.set_cylindrical,
            unit='tuple?',
            vals=Anything()
        )

        self.add_parameter(
            'rho',
            get_cmd=partial(self.get_setpoints, 'rho'),
            set_cmd=self.set_rho,
            unit='T',
            vals=Numbers()
        )

        # TODO: Add the rest of the stuff

    def _get_measured(self, names):
        """ Return the measured coordinates specified in names. """
        x = self._instrument_x.field()
        y = self._instrument_y.field()
        z = self._instrument_z.field()
        return FieldVector(x=x, y=y, z=z).get_components(*names)

    def _get_setpoints(self, names):
        """Get the set point coordinates of the specified names. """
        return self._set_point.get_components(names)

    def _set_fields(self, values):
        """
        Set the fields of the x/y/z magnets. This function is called
        whenever the field is changed and performs several safety checks
        to make sure no limits are exceeded.

        Args:
            values (tuple): a tuple of cartesian coordinates (x, y, z).
        """

        # Check if exceeding the global field limit
        if np.linalg.norm(values) > self._field_limit:
            msg = '_set_fields aborted; field would exceed limit of {} T'
            raise ValueError(msg.format(self._field_limit))

        # Check if the individual instruments are ready
        for name, value in zip(["x", "y", "z"], values):

            instrument = getattr(self, "_instrument_{}".format(name))
            instrument.field.validate(value)
            if instrument.ramping_state() == "ramping":
                msg = '_set_fields aborted; magnet {} is already ramping'
                raise ValueError(msg.format(instrument))

        # Now that we know we can proceed, call the individual instruments

        for operator in [np.less, np.greater]:
            # First ramp the coils that are decreasing in field strength.
            # TODO: Add comments explaining why we do this
            for name, value in zip(["x", "y", "z"], values):

                instrument = getattr(self, "_instrument_{}".format(name))
                current_actual = instrument.field()
                # If the new set point is practically equal to the current one then do nothing
                if np.isclose(value, current_actual, rtol=0, atol=1e-8):
                    continue
                # evaluate if the new set point is lesser or greater then the current value
                if not operator(abs(value), abs(current_actual)):
                    continue

                instrument.set_field(value, perform_safety_check=False)

    # ###########   Public Interface Functions ###########

    def request_field_change(self, instrument, value):
        """
        This method is called by the child x/y/z magnets if they are set
        individually. It results in additional safety checks being
        performed by this 3D driver.
        """
        if instrument is self._instrument_x:
            self.set_x(value)
        elif instrument is self._instrument_y:
            self.set_y(value)
        elif instrument is self._instrument_z:
            self.set_z(value)
        else:
            msg = 'This magnet doesnt belong to its specified parent {}'
            raise NameError(msg.format(self))

    def get_measured(self, names):
        measured_values = self._get_measured(names)

        # Convert angles from radians to degrees
        d = dict(zip(names, measured_values))
        for angle_name in ["phi", "theta"]:
            if angle_name in names:
                d[angle_name] = np.degrees(d[angle_name])

        return [d[name] for name in names]  # Do not do "return list(d.values())", because then there is no
        # guaranty that the order in which the values are returned is the same as the original intention

    def get_setpoints(self, names):

        measured_values = self._get_setpoints(names)

        # Convert angles from radians to degrees
        d = dict(zip(names, measured_values))
        for angle_name in ["phi", "theta"]:
            if angle_name in names:
                d[angle_name] = np.degrees(d[angle_name])

        return [d[name] for name in names]  # Do not do "return list(d.values())", because then there is no
        # guaranty that the order in which the values are returned is the same as the original intention

    def set_cartesian(self, values):
        x, y, z = values
        self._set_point.set_vector(x=x, y=y, z=z)
        self._set_fields(self._set_point.get_components("x", "y", "z"))

    def set_x(self, x):
        self._set_point.set_component(x=x)
        self._set_fields(self._set_point.get_components("x", "y", "z"))

    def set_y(self, y):
        self._set_point.set_component(y=y)
        self._set_fields(self._set_point.get_components("x", "y", "z"))

    def set_z(self, z):
        self._set_point.set_component(z=z)
        self._set_fields(self._set_point.get_components("x", "y", "z"))

    def set_spherical(self, values):
        r, theta, phi = values
        self._set_point.set_vector(r=r, theta=np.radians(theta), phi=np.radians(phi))
        self._set_fields(self._set_point.get_components("x", "y", "z"))

    def set_r(self, r):
        self._set_point.set_component(r=r)
        self._set_fields(self._set_point.get_components("x", "y", "z"))

    def set_theta(self, theta):
        self._set_point.set_component(theta=np.radians(theta))
        self._set_fields(self._set_point.get_components("x", "y", "z"))

    def set_phi(self, phi):
        self._set_point.set_component(phi=np.radians(phi))
        self._set_fields(self._set_point.get_components("x", "y", "z"))

    def set_cylindrical(self, values):
        phi, rho, z = values
        self._set_point.set_vector(phi=np.radians(phi), rho=rho, z=z)
        self._set_fields(self._set_point.get_components("x", "y", "z"))

    def set_rho(self, rho):
        self._set_point.set_component(rho=rho)
        self._set_fields(self._set_point.get_components("x", "y", "z"))