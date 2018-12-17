import collections
import logging
import time
from functools import partial
from warnings import warn

import numpy as np

from qcodes import Instrument, IPInstrument, InstrumentChannel
from qcodes.math.field_vector import FieldVector
from qcodes.utils.validators import Bool, Numbers, Ints, Anything

log = logging.getLogger(__name__)


class AMI430Exception(Exception):
    pass


class AMI430Warning(UserWarning):
    pass


class AMI430SwitchHeater(InstrumentChannel):
    class _Decorators:
        @classmethod
        def check_enabled(cls, f):
            def check_enabled_decorator(self, *args, **kwargs):
                if not self.check_enabled():
                    raise AMI430Exception("Switch not enabled")
                return f(self, *args, **kwargs)
            return check_enabled_decorator

    def __init__(self, parent: 'AMI430') -> None:
        super().__init__(parent, "SwitchHeater")

        # Add state parameters
        self.add_parameter('enabled',
                           label='Switch Heater Enabled',
                           get_cmd=self.check_enabled,
                           set_cmd=lambda x: (self.enable() if x
                                              else self.disable()),
                           vals=Bool())
        self.add_parameter('state',
                           label='Switch Heater On',
                           get_cmd=self.check_state,
                           set_cmd=lambda x: (self.on() if x
                                              else self.off()),
                           vals=Bool())
        self.add_parameter('in_persistent_mode',
                           label='Persistent Mode',
                           get_cmd="PERS?",
                           val_mapping={True: 1, False: 0})

        # Configuration Parameters
        self.add_parameter('current',
                           label='Switch Heater Current',
                           unit='mA',
                           get_cmd='PS:CURR?',
                           get_parser=float,
                           set_cmd='CONF:PS:CURR {}',
                           vals=Numbers(0, 125))
        self.add_parameter('heat_time',
                           label='Heating Time',
                           unit='s',
                           get_cmd='PS:HTIME?',
                           get_parser=int,
                           set_cmd='CONF:PS:HTIME {}',
                           vals=Ints(5, 120))
        self.add_parameter('cool_time',
                           label='Cooling Time',
                           unit='s',
                           get_cmd='PS:CTIME?',
                           get_parser=int,
                           set_cmd='CONF:PS:CTIME {}',
                           vals=Ints(5, 3600))

    def disable(self):
        """Turn measurement off"""
        self.write('CONF:PS 0')
        self._enabled = False

    def enable(self):
        """Turn measurement on"""
        self.write('CONF:PS 1')
        self._enabled = True

    def check_enabled(self):
        return bool(self.ask('PS:INST?').strip())

    @_Decorators.check_enabled
    def on(self):
        self.write("PS 1")
        while self._parent.ramping_state() == "heating switch":
            self._parent._sleep(0.5)

    @_Decorators.check_enabled
    def off(self):
        self.write("PS 0")
        while self._parent.ramping_state() == "cooling switch":
            self._parent._sleep(0.5)

    @_Decorators.check_enabled
    def check_state(self):
        return bool(self.ask("PS?").strip())


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
        current_ramp_limit: A current ramp limit, in units of A/s
    """
    _SHORT_UNITS = {'seconds': 's', 'minutes': 'min',
                    'tesla': 'T', 'kilogauss': 'kG'}
    _DEFAULT_CURRENT_RAMP_LIMIT = 0.06  # [A/s]

    def __init__(self, name, address=None, port=None,
                 reset=False, terminator='\r\n',
                 current_ramp_limit=None, has_current_rating=False,
                 **kwargs):

        super().__init__(name, address, port, terminator=terminator,
                         write_confirmation=False, **kwargs)
        self._parent_instrument = None
        self.has_current_rating = has_current_rating

        # Add reset function
        self.add_function('reset', call_cmd='*RST')
        if reset:
            self.reset()

        # Add parameters setting instrument units
        self.add_parameter("ramp_rate_units",
                           get_cmd='RAMP:RATE:UNITS?',
                           set_cmd=(lambda units:
                                    self._update_units(ramp_rate_units=units)),
                           val_mapping={'seconds': 0,
                                        'minutes': 1})
        self.add_parameter('field_units',
                           get_cmd='FIELD:UNITS?',
                           set_cmd=(lambda units:
                                    self._update_units(field_units=units)),
                           val_mapping={'kilogauss': 0,
                                        'tesla': 1})

        # Set programatic safety limits
        self.add_parameter('current_ramp_limit',
                           get_cmd=lambda: self._current_ramp_limit,
                           set_cmd=self._update_ramp_rate_limit,
                           unit="A/s")
        self.add_parameter('field_ramp_limit',
                           get_cmd=lambda: self.current_ramp_limit(),
                           set_cmd=lambda x: self.current_ramp_limit(x),
                           scale=1/float(self.ask("COIL?")),
                           unit="T/s")
        if current_ramp_limit is None:
            self._update_ramp_rate_limit(AMI430._DEFAULT_CURRENT_RAMP_LIMIT,
                                         update=False)
        else:
            self._update_ramp_rate_limit(current_ramp_limit, update=False)

        # Add solenoid parameters
        self.add_parameter('coil_constant',
                           get_cmd=self._update_coil_constant,
                           set_cmd=self._update_coil_constant,
                           vals=Numbers(0.001, 999.99999))

        # TODO: Not all AMI430s expose this setting. Currently, we
        # don't know why, but this most likely a firmware version issue,
        # so eventually the following condition will be smth like
        # if firmware_version > XX
        if has_current_rating:
            self.add_parameter('current_rating',
                               get_cmd="CURR:RATING?",
                               get_parser=float,
                               set_cmd="CONF:CURR:RATING {}",
                               unit="A",
                               vals=Numbers(0.001, 9999.9999))

            self.add_parameter('field_rating',
                               get_cmd=lambda: self.current_rating(),
                               set_cmd=lambda x: self.current_rating(x),
                               scale=1/float(self.ask("COIL?")))

        self.add_parameter('current_limit',
                           unit="A",
                           set_cmd="CONF:CURR:LIMIT {}",
                           get_cmd='CURR:LIMIT?',
                           get_parser=float,
                           vals=Numbers(0, 80))  # what are good numbers here?

        self.add_parameter('field_limit',
                           set_cmd=self.current_limit.set,
                           get_cmd=self.current_limit.get,
                           scale=1/float(self.ask("COIL?")))

        # Add current solenoid parameters
        # Note that field is validated in set_field
        self.add_parameter('field',
                           get_cmd='FIELD:MAG?',
                           get_parser=float,
                           set_cmd=self.set_field)
        self.add_parameter('ramp_rate',
                           get_cmd=self._get_ramp_rate,
                           set_cmd=self._set_ramp_rate)
        self.add_parameter('setpoint',
                           get_cmd='FIELD:TARG?',
                           get_parser=float)
        self.add_parameter('is_quenched',
                           get_cmd='QU?',
                           val_mapping={True: 1, False: 0})
        self.add_function('reset_quench', call_cmd='QU 0')
        self.add_function('set_quenched', call_cmd='QU 1')
        self.add_parameter('ramping_state',
                           get_cmd='STATE?',
                           get_parser=int,
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

        # Add persistent switch
        switch_heater = AMI430SwitchHeater(self)
        self.add_submodule("switch_heater", switch_heater)

        # Add interaction functions
        self.add_function('get_error', call_cmd='SYST:ERR?')
        self.add_function('ramp', call_cmd='RAMP')
        self.add_function('pause', call_cmd='PAUSE')
        self.add_function('zero', call_cmd='ZERO')

        # Correctly assign all units
        self._update_units()

        self.connect_message()

    def _sleep(self, t):
        """
        Sleep for a number of seconds t. If we are or using
        the PyVISA 'sim' backend, omit this
        """

        simmode = getattr(self, 'visabackend', False) == 'sim'

        if simmode:
            return
        else:
            time.sleep(t)

    def _can_start_ramping(self):
        """
        Check the current state of the magnet to see if we can start ramping
        """
        if self.is_quenched():
            logging.error(__name__ + ': Could not ramp because of quench')
            return False

        if self.switch_heater.in_persistent_mode():
            logging.error(__name__ + ': Could not ramp because persistent')
            return False

        state = self.ramping_state()
        if state == 'ramping':
            # If we don't have a persistent switch, or it's warm
            if not self.switch_heater.enabled():
                return True
            elif self.switch_heater.state():
                return True
        elif state in ['holding', 'paused', 'at zero current']:
            return True

        logging.error(__name__ + ': Could not ramp, state: {}'.format(state))
        return False

    def set_field(self, value, *, block=True, perform_safety_check=True):
        """
        Ramp to a certain field

        Args:
            block (bool): Whether to wait unit the field has finished setting
            perform_safety_check (bool): Whether to set the field via a parent
                driver (if present), which might perform additional safety
                checks.
        """
        # Check we aren't violating field limits
        field_lim = float(self.ask("COIL?"))*self.current_limit()
        if np.abs(value) > field_lim:
            msg = 'Aborted _set_field; {} is higher than limit of {}'
            raise ValueError(msg.format(value, field_lim))

        # If part of a parent driver, set the value using that driver
        if self._parent_instrument is not None and perform_safety_check:
            self._parent_instrument._request_field_change(self, value)
            return

        # Check we can ramp
        if not self._can_start_ramping():
            raise AMI430Exception("Cannot ramp in current state")

        # Then, do the actual ramp
        self.pause()
        # Set the ramp target
        self.write('CONF:FIELD:TARG {}'.format(value))

        # If we have a persistent switch, make sure it is resistive
        if self.switch_heater.enabled():
            if not self.switch_heater.state():
                raise AMI430Exception("Switch heater is not on")
        self.ramp()

        # Check if we want to block
        if not block:
            return

        # Otherwise, wait until no longer ramping
        self.log.debug(f'Starting blocking ramp of {self.name} to {value}')
        while self.ramping_state() == 'ramping':
            self._sleep(0.3)
        self._sleep(2.0)
        state = self.ramping_state()
        self.log.debug(f'Finished blocking ramp')
        # If we are now holding, it was successful
        if state != 'holding':
            msg = '_set_field({}) failed with state: {}'
            raise AMI430Exception(msg.format(value, state))

    def ramp_to(self, value, block=False):
        """ User accessible method to ramp to field """
        # This function duplicates set_field, let's deprecate it...
        warn("This method is deprecated."
             " Use set_field with named parameter block=False instead.",
             DeprecationWarning)
        if self._parent_instrument is not None:
            if not block:
                msg = (": Initiating a blocking instead of non-blocking "
                       " function because this magnet belongs to a parent "
                       "driver")
                logging.warning(__name__ + msg)

            self._parent_instrument._request_field_change(self, value)
        else:
            self.set_field(value, block=False)

    def _get_ramp_rate(self):
        """ Return the ramp rate of the first segment in Tesla per second """
        results = self.ask('RAMP:RATE:FIELD:1?').split(',')
        return float(results[0])

    def _set_ramp_rate(self, rate):
        """ Set the ramp rate of the first segment in Tesla per second """
        if rate > self.field_ramp_limit():
            raise ValueError(f"{rate} {self.ramp_rate.unit} "
                             f"is above the ramp rate limit of "
                             f"{self.field_ramp_limit()} "
                             f"{self.field_ramp_limit()}")
        self.write('CONF:RAMP:RATE:SEG 1')
        self.write('CONF:RAMP:RATE:FIELD 1,{},0'.format(rate))

    def _connect(self):
        """
        Append the IPInstrument connect to flush the welcome message of the AMI
        430 programmer
        :return: None
        """
        super()._connect()
        self.flush_connection()

    def _update_ramp_rate_limit(self, new_current_rate_limit, update=True):
        """
        Update the maximum current ramp rate
        The value passed here is scaled by the units set in
        self.ramp_rate_units
        """
        # Warn if we are going above the default
        warn_level = AMI430._DEFAULT_CURRENT_RAMP_LIMIT
        if new_current_rate_limit > AMI430._DEFAULT_CURRENT_RAMP_LIMIT:
            warning_message = ("Increasing maximum ramp rate: we have a "
                               "default current ramp rate limit of "
                               "{} {}".format(warn_level,
                                              self.current_ramp_limit.unit) +
                               ". We do not want to ramp faster than a set "
                               "maximum so as to avoid quenching "
                               "the magnet. A value of "
                               "{} {}".format(warn_level,
                                              self.current_ramp_limit.unit) +
                               " seems like a safe, conservative value for"
                               " any magnet. Change this value at your own "
                               "responsibility after consulting the specs of "
                               "your particular magnet")
            warn(warning_message, category=AMI430Warning)

        # Update ramp limit
        self._current_ramp_limit = new_current_rate_limit
        # And update instrument limits
        if update:
            field_ramp_limit = self.field_ramp_limit()
            if self.ramp_rate() > field_ramp_limit:
                self.ramp_rate(field_ramp_limit)

    def _update_coil_constant(self, new_coil_constant=None):
        """
        Update the coil constant and relevant scaling factors.
        If new_coil_constant is none, query the coil constant from the
        instrument
        """
        # Query coil constant from instrument
        if new_coil_constant is None:
            new_coil_constant = float(self.ask("COIL?"))
        else:
            self.write("CONF:COIL {}".format(new_coil_constant))

        # Update scaling factors
        if self.has_current_rating:
            self.field_ramp_limit.scale = 1/new_coil_constant
            self.field_rating.scale = 1/new_coil_constant

        # Return new coil constant
        return new_coil_constant

    def _update_units(self, ramp_rate_units=None, field_units=None):
        # Get or set units on device
        if ramp_rate_units is None:
            ramp_rate_units = self.ramp_rate_units()
        else:
            self.write("CONF:RAMP:RATE:UNITS {}".format(ramp_rate_units))
            ramp_rate_units = self.ramp_rate_units.val_mapping[ramp_rate_units]
        if field_units is None:
            field_units = self.field_units()
        else:
            self.write("CONF:FIELD:UNITS {}".format(field_units))
            field_units = self.field_units.val_mapping[field_units]

        # Map to shortened unit names
        ramp_rate_units = AMI430._SHORT_UNITS[ramp_rate_units]
        field_units = AMI430._SHORT_UNITS[field_units]

        # And update all units
        self.coil_constant.unit = "{}/A".format(field_units)
        self.field_limit.unit = f"{field_units}"
        self.field.unit = "{}".format(field_units)
        self.setpoint.unit = "{}".format(field_units)
        self.ramp_rate.unit = "{}/{}".format(field_units, ramp_rate_units)
        self.current_ramp_limit.unit = "A/{}".format(ramp_rate_units)
        self.field_ramp_limit.unit = f"{field_units}/{ramp_rate_units}"

        # And update scaling factors
        # Note: we don't update field_ramp_limit scale as it redirects
        #       to ramp_rate_limit we don't update ramp_rate units as
        #       the instrument stores changed units
        if ramp_rate_units == "min":
            self.current_ramp_limit.scale = 1/60
        else:
            self.current_ramp_limit.scale = 1
        self._update_coil_constant()


class AMI430_3D(Instrument):
    def __init__(self, name, instrument_x, instrument_y,
                 instrument_z, field_limit, **kwargs):
        super().__init__(name, **kwargs)

        if not isinstance(name, str):
            raise ValueError("Name should be a string")

        instruments = [instrument_x, instrument_y, instrument_z]

        if not all([isinstance(instrument, AMI430)
                    for instrument in instruments]):
            raise ValueError("Instruments need to be instances "
                             "of the class AMI430")

        self._instrument_x = instrument_x
        self._instrument_y = instrument_y
        self._instrument_z = instrument_z

        if repr(field_limit).isnumeric() or isinstance(field_limit, collections.abc.Iterable):
            self._field_limit = field_limit
        else:
            raise ValueError("field limit should either be"
                             " a number or an iterable")

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
                'r',
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
            get_cmd=partial(self._get_setpoints, ('x', 'y', 'z')),
            set_cmd=partial(self._set_setpoints, ('x', 'y', 'z')),
            unit='T',
            vals=Anything()
        )

        self.add_parameter(
            'x',
            get_cmd=partial(self._get_setpoints, ('x',)),
            set_cmd=partial(self._set_setpoints, ('x',)),
            unit='T',
            vals=Numbers()
        )

        self.add_parameter(
            'y',
            get_cmd=partial(self._get_setpoints, ('y',)),
            set_cmd=partial(self._set_setpoints, ('y',)),
            unit='T',
            vals=Numbers()
        )

        self.add_parameter(
            'z',
            get_cmd=partial(self._get_setpoints, ('z',)),
            set_cmd=partial(self._set_setpoints, ('z',)),
            unit='T',
            vals=Numbers()
        )

        self.add_parameter(
            'spherical',
            get_cmd=partial(
                self._get_setpoints, ('r', 'theta', 'phi')
            ),
            set_cmd=partial(
                self._set_setpoints, ('r', 'theta', 'phi')
            ),
            unit='tuple?',
            vals=Anything()
        )

        self.add_parameter(
            'phi',
            get_cmd=partial(self._get_setpoints, ('phi',)),
            set_cmd=partial(self._set_setpoints, ('phi',)),
            unit='deg',
            vals=Numbers()
        )

        self.add_parameter(
            'theta',
            get_cmd=partial(self._get_setpoints, ('theta',)),
            set_cmd=partial(self._set_setpoints, ('theta',)),
            unit='deg',
            vals=Numbers()
        )

        self.add_parameter(
            'field',
            get_cmd=partial(self._get_setpoints, ('r',)),
            set_cmd=partial(self._set_setpoints, ('r',)),
            unit='T',
            vals=Numbers()
        )

        self.add_parameter(
            'cylindrical',
            get_cmd=partial(
                self._get_setpoints, ('rho', 'phi', 'z')
            ),
            set_cmd=partial(
                self._set_setpoints, ('rho', 'phi', 'z')
            ),
            unit='tuple?',
            vals=Anything()
        )

        self.add_parameter(
            'rho',
            get_cmd=partial(self._get_setpoints, ('rho',)),
            set_cmd=partial(self._set_setpoints, ('rho',)),
            unit='T',
            vals=Numbers()
        )

        self.add_parameter(
            'block_during_ramp',
            set_cmd=None,
            initial_value=True,
            unit='',
            vals=Bool()
        )

    def _verify_safe_setpoint(self, setpoint_values):

        if repr(self._field_limit).isnumeric():
            return np.linalg.norm(setpoint_values) < self._field_limit

        answer = any([limit_function(*setpoint_values) for
                      limit_function in self._field_limit])

        return answer

    def _adjust_child_instruments(self, values):
        """
        Set the fields of the x/y/z magnets. This function is called
        whenever the field is changed and performs several safety checks
        to make sure no limits are exceeded.

        Args:
            values (tuple): a tuple of cartesian coordinates (x, y, z).
        """
        self.log.debug("Checking whether fields can be set")

        # Check if exceeding the global field limit
        if not self._verify_safe_setpoint(values):
            raise ValueError("_set_fields aborted; field would exceed limit")

        # Check if the individual instruments are ready
        for name, value in zip(["x", "y", "z"], values):

            instrument = getattr(self, "_instrument_{}".format(name))
            if instrument.ramping_state() == "ramping":
                msg = '_set_fields aborted; magnet {} is already ramping'
                raise AMI430Exception(msg.format(instrument))

        # Now that we know we can proceed, call the individual instruments

        self.log.debug("Field values OK, proceeding")
        for operator in [np.less, np.greater]:
            # First ramp the coils that are decreasing in field strength.
            # This will ensure that we are always in a safe region as
            # far as the quenching of the magnets is concerned
            for name, value in zip(["x", "y", "z"], values):

                instrument = getattr(self, "_instrument_{}".format(name))
                current_actual = instrument.field()

                # If the new set point is practically equal to the
                # current one then do nothing
                if np.isclose(value, current_actual, rtol=0, atol=1e-8):
                    continue
                # evaluate if the new set point is smaller or larger
                # than the current value
                if not operator(abs(value), abs(current_actual)):
                    continue

                instrument.set_field(value, perform_safety_check=False,
                                     block=self.block_during_ramp.get())

    def _request_field_change(self, instrument, value):
        """
        This method is called by the child x/y/z magnets if they are set
        individually. It results in additional safety checks being
        performed by this 3D driver.
        """
        if instrument is self._instrument_x:
            self._set_x(value)
        elif instrument is self._instrument_y:
            self._set_y(value)
        elif instrument is self._instrument_z:
            self._set_z(value)
        else:
            msg = 'This magnet doesnt belong to its specified parent {}'
            raise NameError(msg.format(self))

    def _get_measured(self, *names):

        x = self._instrument_x.field()
        y = self._instrument_y.field()
        z = self._instrument_z.field()
        measured_values = FieldVector(x=x, y=y, z=z).get_components(*names)

        # Convert angles from radians to degrees
        d = dict(zip(names, measured_values))

        # Do not do "return list(d.values())", because then there is
        # no guaranty that the order in which the values are returned
        # is the same as the original intention
        return_value = [d[name] for name in names]

        if len(names) == 1:
            return_value = return_value[0]

        return return_value

    def _get_setpoints(self, names):

        measured_values = self._set_point.get_components(*names)

        # Convert angles from radians to degrees
        d = dict(zip(names, measured_values))
        return_value = [d[name] for name in names]
        # Do not do "return list(d.values())", because then there is
        # no guarantee that the order in which the values are returned
        # is the same as the original intention

        if len(names) == 1:
            return_value = return_value[0]

        return return_value

    def _set_setpoints(self, names, values):

        kwargs = dict(zip(names, np.atleast_1d(values)))

        set_point = FieldVector()
        set_point.copy(self._set_point)
        if len(kwargs) == 3:
            set_point.set_vector(**kwargs)
        else:
            set_point.set_component(**kwargs)

        self._adjust_child_instruments(
            set_point.get_components("x", "y", "z")
        )

        self._set_point = set_point

