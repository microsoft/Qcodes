import collections
import logging
import numbers
import time
import warnings
from collections import defaultdict
from functools import partial
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np

from qcodes import Instrument, InstrumentChannel, IPInstrument, Parameter
from qcodes.math_utils.field_vector import FieldVector
from qcodes.utils.deprecate import QCoDeSDeprecationWarning
from qcodes.utils.validators import Anything, Bool, Enum, Ints, Numbers

log = logging.getLogger(__name__)

CartesianFieldLimitFunction = \
    Callable[[float, float, float], bool]

T = TypeVar('T')


class AMI430Exception(Exception):
    pass


class AMI430Warning(UserWarning):
    pass


class AMI430SwitchHeater(InstrumentChannel):
    class _Decorators:
        @classmethod
        def check_enabled(cls, f: Callable[..., T]) -> Callable[..., T]:
            def check_enabled_decorator(self: "AMI430SwitchHeater",
                                        *args: Any, **kwargs: Any) -> T:
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

    def disable(self) -> None:
        """Turn measurement off"""
        self.write('CONF:PS 0')
        self._enabled = False

    def enable(self) -> None:
        """Turn measurement on"""
        self.write('CONF:PS 1')
        self._enabled = True

    def check_enabled(self) -> bool:
        return bool(int(self.ask('PS:INST?').strip()))

    @_Decorators.check_enabled
    def on(self) -> None:
        self.write("PS 1")
        while self._parent.ramping_state() == "heating switch":
            self._parent._sleep(0.5)

    @_Decorators.check_enabled
    def off(self) -> None:
        self.write("PS 0")
        while self._parent.ramping_state() == "cooling switch":
            self._parent._sleep(0.5)

    @_Decorators.check_enabled
    def check_state(self) -> bool:
        return bool(int(self.ask("PS?").strip()))


class AMI430(IPInstrument):
    """
    Driver for the American Magnetics Model 430 magnet power supply programmer.

    This class controls a single magnet power supply. In order to use two or
    three magnets simultaneously to set field vectors, first instantiate the
    individual magnets using this class and then pass them as arguments to
    either the AMI430_2D or AMI430_3D virtual instrument classes.

    Args:
        name: a name for the instrument
        address: IP address of the power supply programmer
        current_ramp_limit: A current ramp limit, in units of A/s
    """
    _SHORT_UNITS = {'seconds': 's', 'minutes': 'min',
                    'tesla': 'T', 'kilogauss': 'kG'}
    _DEFAULT_CURRENT_RAMP_LIMIT = 0.06  # [A/s]

    def __init__(
            self,
            name: str,
            address: str,
            port: Optional[int] = None,
            reset: bool = False,
            terminator: str = '\r\n',
            current_ramp_limit: Optional[float] = None,
            **kwargs: Any):
        if "has_current_rating" in kwargs.keys():
            warnings.warn(
                "'has_current_rating' kwarg to AMI430 "
                "is deprecated and has no effect",
                category=QCoDeSDeprecationWarning)
            kwargs.pop("has_current_rating")

        super().__init__(name, address, port, terminator=terminator,
                         write_confirmation=False, **kwargs)
        self._parent_instrument = None

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

        # Set programmatic safety limits
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
        self.add_parameter('ramping_state_check_interval',
                           initial_value=0.05,
                           unit="s",
                           vals=Numbers(0, 10),
                           set_cmd=None)

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

    def _sleep(self, t: float) -> None:
        """
        Sleep for a number of seconds t. If we are or using
        the PyVISA 'sim' backend, omit this
        """

        simmode = getattr(self, 'visabackend', False) == 'sim'

        if simmode:
            return
        else:
            time.sleep(t)

    def _can_start_ramping(self) -> bool:
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

        logging.error(__name__ + f': Could not ramp, state: {state}')
        return False

    def set_field(self,
                  value: float,
                  *,
                  block: bool = True,
                  perform_safety_check: bool = True) -> None:
        """
        Ramp to a certain field

        Args:
            value: Value to ramp to.
            block: Whether to wait unit the field has finished setting
            perform_safety_check: Whether to set the field via a parent
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
            raise AMI430Exception(f"Cannot ramp in current state: "
                                  f"state is {self.ramping_state()}")

        # Then, do the actual ramp
        self.pause()
        # Set the ramp target
        self.write(f'CONF:FIELD:TARG {value}')

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
        exit_state = self.wait_while_ramping()
        self.log.debug(f'Finished blocking ramp')
        # If we are now holding, it was successful
        if exit_state != 'holding':
            msg = '_set_field({}) failed with state: {}'
            raise AMI430Exception(msg.format(value, exit_state))

    def wait_while_ramping(self) -> str:

        while self.ramping_state() == 'ramping':
            self._sleep(self.ramping_state_check_interval())

        return self.ramping_state()

    def _get_ramp_rate(self) -> float:
        """ Return the ramp rate of the first segment in Tesla per second """
        results = self.ask('RAMP:RATE:FIELD:1?').split(',')
        return float(results[0])

    def _set_ramp_rate(self, rate: float) -> None:
        """ Set the ramp rate of the first segment in Tesla per second """
        if rate > self.field_ramp_limit():
            raise ValueError(f"{rate} {self.ramp_rate.unit} "
                             f"is above the ramp rate limit of "
                             f"{self.field_ramp_limit()} "
                             f"{self.field_ramp_limit()}")
        self.write('CONF:RAMP:RATE:SEG 1')
        self.write(f'CONF:RAMP:RATE:FIELD 1,{rate},0')

    def _connect(self) -> None:
        """
        Append the IPInstrument connect to flush the welcome message of the AMI
        430 programmer
        :return: None
        """
        super()._connect()
        self.flush_connection()

    def _update_ramp_rate_limit(self,
                                new_current_rate_limit: float,
                                update: bool = True) -> None:
        """
        Update the maximum current ramp rate
        The value passed here is scaled by the units set in
        self.ramp_rate_units
        """
        # Update ramp limit
        self._current_ramp_limit = new_current_rate_limit
        # And update instrument limits
        if update:
            field_ramp_limit = self.field_ramp_limit()
            if self.ramp_rate() > field_ramp_limit:
                self.ramp_rate(field_ramp_limit)

    def _update_coil_constant(
            self,
            new_coil_constant: Optional[float] = None
    ) -> float:
        """
        Update the coil constant and relevant scaling factors.
        If new_coil_constant is none, query the coil constant from the
        instrument
        """
        # Query coil constant from instrument
        if new_coil_constant is None:
            new_coil_constant = float(self.ask("COIL?"))
        else:
            self.write(f"CONF:COIL {new_coil_constant}")

        # Update scaling factors
        self.field_ramp_limit.scale = 1/new_coil_constant
        self.field_limit.scale = 1/new_coil_constant

        # Return new coil constant
        return new_coil_constant

    def _update_units(
        self, ramp_rate_units: Optional[int] = None, field_units: Optional[int] = None
    ) -> None:
        # Get or set units on device
        if ramp_rate_units is None:
            ramp_rate_units_int: str = self.ramp_rate_units()
        else:
            self.write(f"CONF:RAMP:RATE:UNITS {ramp_rate_units}")
            ramp_rate_units_int = self.ramp_rate_units.\
                inverse_val_mapping[ramp_rate_units]
        if field_units is None:
            field_units_int: str = self.field_units()
        else:
            self.write(f"CONF:FIELD:UNITS {field_units}")
            field_units_int = self.field_units.inverse_val_mapping[field_units]

        # Map to shortened unit names
        ramp_rate_units_short = AMI430._SHORT_UNITS[ramp_rate_units_int]
        field_units_short = AMI430._SHORT_UNITS[field_units_int]

        # And update all units
        self.coil_constant.unit = f"{field_units_short}/A"
        self.field_limit.unit = f"{field_units_short}"
        self.field.unit = f"{field_units_short}"
        self.setpoint.unit = f"{field_units_short}"
        self.ramp_rate.unit = f"{field_units_short}/{ramp_rate_units_short}"
        self.current_ramp_limit.unit = f"A/{ramp_rate_units_short}"
        self.field_ramp_limit.unit = f"{field_units_short}/{ramp_rate_units_short}"

        # And update scaling factors
        # Note: we don't update field_ramp_limit scale as it redirects
        #       to ramp_rate_limit; we don't update ramp_rate units as
        #       the instrument stores changed units
        if ramp_rate_units_short == "min":
            self.current_ramp_limit.scale = 1/60
        else:
            self.current_ramp_limit.scale = 1

        # If the field units change, the value of the coil constant also
        # changes, hence we read the new value of the coil constant from the
        # instrument via the `coil_constant` parameter (which in turn also
        # updates settings of some parameters due to the fact that the coil
        # constant changes)
        self.coil_constant()


class AMI430_3D(Instrument):
    def __init__(self,
                 name: str,
                 instrument_x: Union[AMI430, str],
                 instrument_y: Union[AMI430, str],
                 instrument_z: Union[AMI430, str],
                 field_limit: Union[numbers.Real,
                                    Iterable[CartesianFieldLimitFunction]],
                 **kwargs: Any):
        """
        Driver for controlling three American Magnetics Model 430 magnet power
        supplies simultaneously for setting magnetic field vectors.

        The individual magnet power supplies can be passed in as either
        instances of AMI430 driver or as names of existing AMI430 instances.
        In the latter case, the instances will be found via the passed names.

        Args:
            name: a name for the instrument
            instrument_x: AMI430 instance or a names of existing AMI430
                instance for controlling the X axis of magnetic field
            instrument_y: AMI430 instance or a names of existing AMI430
                instance for controlling the Y axis of magnetic field
            instrument_z: AMI430 instance or a names of existing AMI430
                instance for controlling the Z axis of magnetic field
            field_limit: a number for maximum allows magnetic field or an
                iterable of callable field limit functions that define
                region(s) of allowed values in 3D magnetic field space
        """
        super().__init__(name, **kwargs)

        if not isinstance(name, str):
            raise ValueError("Name should be a string")

        for instrument, arg_name in zip(
                (instrument_x, instrument_y, instrument_z),
                ("instrument_x", "instrument_y", "instrument_z"),
        ):
            if not isinstance(instrument, (AMI430, str)):
                raise ValueError(
                    f"Instruments need to be instances of the class AMI430 "
                    f"or be valid names of already instantiated instances "
                    f"of AMI430 class; {arg_name} argument is "
                    f"neither of those"
                )

        def find_ami430_with_name(ami430_name: str) -> AMI430:
            found_ami430 = AMI430.find_instrument(
                name=ami430_name, instrument_class=AMI430
            )
            return found_ami430

        self._instrument_x = (
            instrument_x if isinstance(instrument_x, AMI430)
            else find_ami430_with_name(instrument_x)
        )
        self._instrument_y = (
            instrument_y if isinstance(instrument_y, AMI430)
            else find_ami430_with_name(instrument_y)
        )
        self._instrument_z = (
            instrument_z if isinstance(instrument_z, AMI430)
            else find_ami430_with_name(instrument_z)
        )

        self._field_limit: Union[float, Iterable[CartesianFieldLimitFunction]]
        if isinstance(field_limit, collections.abc.Iterable):
            self._field_limit = field_limit
        elif isinstance(field_limit, numbers.Real):
            # Conversion to float makes related driver logic simpler
            self._field_limit = float(field_limit)
        else:
            raise ValueError("field limit should either be a number or "
                             "an iterable of callable field limit functions.")

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

        self.ramp_mode = Parameter(
            name="ramp_mode",
            instrument=self,
            get_cmd=None,
            set_cmd=None,
            vals=Enum("default", "simultaneous"),
            initial_value="default",
        )

        self.ramping_state_check_interval = Parameter(
            name="ramping_state_check_interval",
            instrument=self,
            initial_value=0.05,
            unit="s",
            vals=Numbers(0, 10),
            set_cmd=None,
            get_cmd=None,
        )

        self.vector_ramp_rate = Parameter(
            name="vector_ramp_rate",
            instrument=self,
            unit="T/s",
            vals=Numbers(min_value=0.0),
            set_cmd=None,
            get_cmd=None,
            set_parser=self._set_vector_ramp_rate_units,
            docstring="Ramp rate along a line (vector) in 3D space. Only active"
                      " if `ramp_mode='simultaneous'`."
        )
        """Ramp rate along a line (vector) in 3D field space"""

    def _set_vector_ramp_rate_units(self, val: float) -> float:
        _, common_ramp_rate_units = self._raise_if_not_same_field_and_ramp_rate_units()
        self.vector_ramp_rate.unit = common_ramp_rate_units
        return val

    def ramp_simultaneously(self, setpoint: FieldVector, duration: float) -> None:
        """
        Ramp all axes simultaneously to the given setpoint and in the given time

        The method calculates and sets the required ramp rates per magnet
        axis, and then initiates a ramp simultaneously on all the axes. The
        trajectory of the tip of the magnetic field vector is thus linear in
        3D space, from the current field value to the setpoint.

        If ``block_during_ramp`` parameter is ``True``, the method will block
        until all axes finished ramping.

        It is required for all axis instruments to have the same units for
        ramp rate and field, otherwise an exception is raised. The given
        setpoint and time are assumed to be in those common units.

        Args:
            setpoint: ``FieldVector`` setpoint
            duration: time in which the setpoint field has to be reached on all axes

        """
        (
            common_field_units,
            common_ramp_rate_units,
        ) = self._raise_if_not_same_field_and_ramp_rate_units()

        self.log.debug(
            f"Simultaneous ramp: setpoint {setpoint.repr_cartesian()} "
            f"{common_field_units} in {duration} {common_ramp_rate_units}"
        )

        # Get starting field value

        start_field = self._get_measured_field_vector()
        self.log.debug(
            f"Simultaneous ramp: start {start_field.repr_cartesian()} "
            f"{common_field_units}"
        )
        self.log.debug(
            f"Simultaneous ramp: delta {(setpoint - start_field).repr_cartesian()} "
            f"{common_field_units}"
        )

        # Calculate new vector ramp rate based on time and setpoint

        vector_ramp_rate = self.calculate_vector_ramp_rate_from_duration(
            start=start_field, setpoint=setpoint, duration=duration
        )
        self.vector_ramp_rate(vector_ramp_rate)
        self.log.debug(
            f"Simultaneous ramp: new vector ramp rate for {self.full_name} "
            f"is {vector_ramp_rate} {common_ramp_rate_units}"
        )

        # Launch the simultaneous ramp

        self.ramp_mode("simultaneous")
        self.cartesian(setpoint.get_components("x", "y", "z"))

    @staticmethod
    def calculate_axes_ramp_rates_for(
        start: FieldVector, setpoint: FieldVector, duration: float
    ) -> Tuple[float, float, float]:
        """
        Given starting and setpoint fields and expected ramp time calculates
        required ramp rates for x, y, z axes (in this order) where axes are
        ramped simultaneously.
        """
        vector_ramp_rate = AMI430_3D.calculate_vector_ramp_rate_from_duration(
            start, setpoint, duration
        )
        return AMI430_3D.calculate_axes_ramp_rates_from_vector_ramp_rate(
            start, setpoint, vector_ramp_rate
        )

    @staticmethod
    def calculate_vector_ramp_rate_from_duration(
        start: FieldVector, setpoint: FieldVector, duration: float
    ) -> float:
        return setpoint.distance(start) / duration

    @staticmethod
    def calculate_axes_ramp_rates_from_vector_ramp_rate(
        start: FieldVector, setpoint: FieldVector, vector_ramp_rate: float
    ) -> Tuple[float, float, float]:
        delta_field = setpoint - start
        ramp_rate_3d = delta_field / delta_field.norm() * vector_ramp_rate
        return abs(ramp_rate_3d["x"]), abs(ramp_rate_3d["y"]), abs(ramp_rate_3d["z"])

    def _raise_if_not_same_field_and_ramp_rate_units(self) -> Tuple[str, str]:
        instruments = (self._instrument_x, self._instrument_y, self._instrument_z)

        field_units_of_instruments = defaultdict(set)
        ramp_rate_units_of_instruments = defaultdict(set)

        for instrument in instruments:
            ramp_rate_units_of_instruments[instrument.ramp_rate_units.cache.get()].add(
                instrument.full_name
            )
            field_units_of_instruments[instrument.field_units.cache.get()].add(
                instrument.full_name
            )

        if len(field_units_of_instruments) != 1:
            raise ValueError(
                f"Magnet axes instruments should have the same "
                f"`field_units`, instead they have: "
                f"{field_units_of_instruments}"
            )

        if len(ramp_rate_units_of_instruments) != 1:
            raise ValueError(
                f"Magnet axes instruments should have the same "
                f"`ramp_rate_units`, instead they have: "
                f"{ramp_rate_units_of_instruments}"
            )

        common_field_units = tuple(field_units_of_instruments.keys())[0]
        common_ramp_rate_units = tuple(ramp_rate_units_of_instruments.keys())[0]

        return common_field_units, common_ramp_rate_units

    def _verify_safe_setpoint(
            self,
            setpoint_values: Tuple[float, float, float]
    ) -> bool:
        if isinstance(self._field_limit, (int, float)):
            return bool(np.linalg.norm(setpoint_values) < self._field_limit)

        answer = any([limit_function(*setpoint_values) for
                      limit_function in self._field_limit])

        return answer

    def _adjust_child_instruments(
            self,
            values: Tuple[float, float, float]
    ) -> None:
        """
        Set the fields of the x/y/z magnets. This function is called
        whenever the field is changed and performs several safety checks
        to make sure no limits are exceeded.

        Args:
            values: a tuple of cartesian coordinates (x, y, z).
        """
        self.log.debug("Checking whether fields can be set")

        # Check if exceeding the global field limit
        if not self._verify_safe_setpoint(values):
            raise ValueError("_set_fields aborted; field would exceed limit")

        # Check if the individual instruments are ready
        for name, value in zip(["x", "y", "z"], values):

            instrument = getattr(self, f"_instrument_{name}")
            if instrument.ramping_state() == "ramping":
                msg = '_set_fields aborted; magnet {} is already ramping'
                raise AMI430Exception(msg.format(instrument))

        # Now that we know we can proceed, call the individual instruments

        self.log.debug("Field values OK, proceeding")

        if self.ramp_mode() == "simultaneous":
            self._perform_simultaneous_ramp(values)
        else:
            self._perform_default_ramp(values)

    def _update_individual_axes_ramp_rates(
        self, values: Tuple[float, float, float]
    ) -> None:
        if self.vector_ramp_rate() is None or self.vector_ramp_rate() == 0:
            raise ValueError('The value of the `vector_ramp_rate` Parameter is '
                             'currently None or 0. Set it to an appropriate '
                             'value to use the simultaneous ramping feature.')

        new_axes_ramp_rates = self.calculate_axes_ramp_rates_from_vector_ramp_rate(
            start=self._get_measured_field_vector(),
            setpoint=FieldVector(x=values[0], y=values[1], z=values[2]),
            vector_ramp_rate=self.vector_ramp_rate.get(),
        )
        instruments = (self._instrument_x, self._instrument_y, self._instrument_z)
        for instrument, new_axis_ramp_rate in zip(instruments, new_axes_ramp_rates):
            instrument.ramp_rate.set(new_axis_ramp_rate)
            self.log.debug(
                f"Simultaneous ramp: new rate for {instrument.full_name} "
                f"is {new_axis_ramp_rate} {instrument.ramp_rate.unit}"
            )

    def _perform_simultaneous_ramp(self, values: Tuple[float, float, float]) -> None:
        self._update_individual_axes_ramp_rates(values)

        axes = (self._instrument_x, self._instrument_y, self._instrument_z)

        for axis_instrument, value in zip(axes, values):
            current_actual = axis_instrument.field()

            # If the new set point is practically equal to the
            # current one then do nothing
            if np.isclose(value, current_actual, rtol=0, atol=1e-8):
                self.log.debug(
                    f"Simultaneous ramp: {axis_instrument.short_name} is "
                    f"already at target field {value} "
                    f"{axis_instrument.field.unit} "
                    f"({current_actual} exactly)"
                )
                continue

            self.log.debug(
                f"Simultaneous ramp: setting {axis_instrument.short_name} "
                f"target field to {value} {axis_instrument.field.unit}"
            )
            axis_instrument.set_field(value, perform_safety_check=False, block=False)

        if self.block_during_ramp() is True:
            self.log.debug(f"Simultaneous ramp: blocking until ramp is finished")
            self.wait_while_all_axes_ramping()

        self.log.debug(f"Simultaneous ramp: returning from the ramp call")

    def _perform_default_ramp(self, values: Tuple[float, float, float]) -> None:
        operators: Tuple[Callable[[Any, Any], bool], ...] = (np.less, np.greater)
        for operator in operators:
            # First ramp the coils that are decreasing in field strength.
            # This will ensure that we are always in a safe region as
            # far as the quenching of the magnets is concerned
            for name, value in zip(["x", "y", "z"], values):

                instrument = getattr(self, f"_instrument_{name}")
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

    def wait_while_all_axes_ramping(self) -> None:
        """ Wait and blocks as long as any magnet axis is ramping. """
        while self.any_axis_is_ramping():
            self._instrument_x._sleep(self.ramping_state_check_interval.get())

    def any_axis_is_ramping(self) -> bool:
        """
        Returns True if any of the magnet axes are currently ramping, or False
        if none of the axes are ramping.
        """
        return any(
            axis_instrument.ramping_state() == "ramping"
            for axis_instrument in (
                self._instrument_x,
                self._instrument_y,
                self._instrument_z,
            )
        )

    def pause(self) -> None:
        """ Pause all magnet axes. """
        for axis_instrument in (self._instrument_x, self._instrument_y, self._instrument_z):
            axis_instrument.pause()

    def _request_field_change(self, instrument: AMI430,
                              value: numbers.Real) -> None:
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

    def _get_measured_field_vector(self) -> FieldVector:
        return FieldVector(
            x=self._instrument_x.field(),
            y=self._instrument_y.field(),
            z=self._instrument_z.field(),
        )

    def _get_measured(
            self,
            *names: str
    ) -> Union[numbers.Real, List[numbers.Real]]:
        measured_field_vector = self._get_measured_field_vector()

        measured_values = measured_field_vector.get_components(*names)

        # Convert angles from radians to degrees
        d = dict(zip(names, measured_values))

        # Do not do "return list(d.values())", because then there is
        # no guaranty that the order in which the values are returned
        # is the same as the original intention
        return_value = [d[name] for name in names]

        if len(names) == 1:
            return_value = return_value[0]

        return return_value

    def _get_setpoints(
            self,
            names: Sequence[str]
    ) -> Union[numbers.Real, List[numbers.Real]]:

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

    def _set_setpoints(
            self,
            names: Sequence[str],
            values: Sequence[float]
    ) -> None:

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
