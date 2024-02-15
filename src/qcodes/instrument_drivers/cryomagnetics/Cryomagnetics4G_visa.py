from __future__ import annotations

import logging
import numbers
import time
import warnings
from enum import Enum
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from functools import partial
from typing import Any, Callable, ClassVar, TypeVar, cast
#
import numpy as np
from pyvisa import VisaIOError
#
from qcodes.instrument import Instrument, InstrumentChannel, VisaInstrument
#from qcodes.math_utils import FieldVector
from qcodes.parameters import Parameter
from qcodes.utils import QCoDeSDeprecationWarning
from qcodes.validators import Anything, Bool, Enum, Ints, Numbers
#
log = logging.getLogger(__name__)

T = TypeVar("T")
#

class FieldUnits(Enum):
    GAUSS = "G"
    AMPERE = "A"


class Cryo4GException(Exception):
    pass


class Cryo4GWarning(UserWarning):
    pass


class Cryo4GSwitchHeater(InstrumentChannel):
    class _Decorators:
        @classmethod
        def check_enabled(cls, f: Callable[..., T]) -> Callable[..., T]:
            def check_enabled_decorator(
                self: Cryo4GSwitchHeater, *args: Any, **kwargs: Any
            ) -> T:
                if not self.check_enabled():
                    raise Cryo4GException("Switch not enabled")
                return f(self, *args, **kwargs)

            return check_enabled_decorator

    def __init__(self, parent: CryomagneticsModel4G) -> None:
        super().__init__(parent, "SwitchHeater")

        # Add state parameters
        self.add_parameter(
            "enabled",
            label="Switch Heater Enabled",
            get_cmd=self.check_enabled,
            set_cmd=lambda x: (self.enable() if x else self.disable()),
            vals=Bool(),
        )
        self.add_parameter(
            "state",
            label="Switch Heater On",
            get_cmd=self.check_state,
            set_cmd=lambda x: (self.on() if x else self.off()),
            vals=Bool(),
        )
        
        #Not sure about this cmd being correct
        self.add_parameter(
            "in_persistent_mode",
            label="Persistent Mode",
            get_cmd="MODE?",
            val_mapping={True: "Shim", False: "Manual"},
        )

        # Configuration Parameters
        #self.add_parameter(
        #    "current",
        #    label="Switch Heater Current",
        #    unit="mA",
        #    get_cmd="IMAG?",
        #    get_parser=float,
        #    set_cmd="IMAG {}",
        #    vals=Numbers(0, 125),
        #)
        #self.add_parameter(
        #    "heat_time",
        #    label="Heating Time",
        #    unit="s",
        #    get_cmd="PS:HTIME?",
        #    get_parser=int,
        #    set_cmd="CONF:PS:HTIME {}",
        #    vals=Ints(5, 120),
        #)
        #self.add_parameter(
        #    "cool_time",
        #    label="Cooling Time",
        #    unit="s",
        #    get_cmd="PS:CTIME?",
        #    get_parser=int,
        #    set_cmd="CONF:PS:CTIME {}",
        #    vals=Ints(5, 3600),
        #)

    def disable(self) -> None:
        """Turn measurement off"""
        self.write("PSHTR 0")
        self._enabled = False

    def enable(self) -> None:
        """Turn measurement on"""
        self.write("PSHTR 1")
        self._enabled = True

    def check_enabled(self) -> bool:
        return bool(int(self.ask("PSHTR?").strip()))

    @_Decorators.check_enabled
    def on(self) -> None:
        self.write("PSHTR 1")
        while self._parent.ramping_state() == "heating switch":
            self._parent._sleep(0.5)

    @_Decorators.check_enabled
    def off(self) -> None:
        self.write("PSHTR 0")
        while self._parent.ramping_state() == "cooling switch":
            self._parent._sleep(0.5)

    @_Decorators.check_enabled
    def check_state(self) -> bool:
        return bool(int(self.ask("PSHTR?").strip()))


class CryomagneticsModel4G(VisaInstrument):
    """
    Driver for the American Magnetics Model 430 magnet power supply programmer.

    This class controls a single magnet power supply. In order to use two or
    three magnets simultaneously to set field vectors, first instantiate the
    individual magnets using this class and then pass them as arguments to
    the CryomagneticsModel4G3D virtual instrument classes.

    Args:
        name: a name for the instrument
        address: VISA formatted address of the power supply programmer.
            Of the form ``TCPIP[board]::host address::port::SOCKET``
            e.g. ``TCPIP0::192.168.0.1::7800::SOCKET``
        current_ramp_limit: A current ramp limit, in units of A/s
    """

    _SHORT_UNITS: ClassVar[dict[str, str]] = {
        "seconds": "s",
        "minutes": "min",
        "tesla": "T",
        "kilogauss": "kG",
        "guass": "G",
        "amp": "A", 
    }
    _DEFAULT_CURRENT_RAMP_LIMIT = 0.06  # [A/s]
    _RETRY_WRITE_ASK = True
    _RETRY_TIME = 5

    def __init__(
        self,
        name: str,
        address: str,
        reset: bool = False,
        terminator: str = "\r\n",
        current_ramp_limit: float | None = None,
    ):

        super().__init__(
            name,
            address,
            terminator=terminator,
        )

        simmode = getattr(self, "visabackend", False) == "sim"
        # pyvisa-sim does not support connect messages
        if not simmode:
            # the AMI 430 sends a welcome message of
            # 'American Magnetics Model 430 IP Interface'
            # 'Hello'
            # here we read that out before communicating with the instrument
            # if that is not the first reply likely there is left over messages
            # in the buffer so read until empty
            message1 = self.visa_handle.read()
            if "Cryomagnetics Model 4G Power Supply Interface" not in message1:
                try:
                    while True:
                        self.visa_handle.read()
                except VisaIOError:
                    pass
            else:
                # read the hello part of the welcome message
                self.visa_handle.read()

        self._parent_instrument = None

        # Add reset function
        self.add_function("reset", call_cmd="*RST")
        if reset:
            self.reset()

        # Add parameters setting instrument units
        #need to create this
        #self.add_parameter(
        #    "ramp_rate_units",
        #    get_cmd="RAMP:RATE:UNITS?",
        #    set_cmd=(lambda units: self._update_units(ramp_rate_units=units)),
        #    val_mapping={"seconds": 0, "minutes": 1},
        #)
        self.add_parameter(
            "field_units",
            get_cmd="UNITS?",
            set_cmd=lambda units: self._update_units(units.value),  # Assuming _update_units takes the string representation.
            val_mapping={FieldUnits.GAUSS: "G", FieldUnits.AMPERE: "A"},
        )

        # Set programmatic safety limits
        self.add_parameter(
            "current_ramp_limit",
            get_cmd=lambda: self._current_ramp_limit,
            set_cmd=self._update_ramp_rate_limit,
            unit="A/s",
        )
        self.add_parameter(
            "field_ramp_limit",
            get_cmd=lambda: self.current_ramp_limit(),
            set_cmd=lambda x: self.current_ramp_limit(x),
            scale=1 / float(self.ask("COIL?")),
            unit="T/s",
        )
        if current_ramp_limit is None:
            self._update_ramp_rate_limit(
                CryomagneticsModel4G._DEFAULT_CURRENT_RAMP_LIMIT, update=False
            )
        else:
            self._update_ramp_rate_limit(current_ramp_limit, update=False)

        # Add solenoid parameters
        self.add_parameter(
            "coil_constant",
            get_cmd=self._update_coil_constant,
            set_cmd=self._update_coil_constant,
            vals=Numbers(0.001, 999.99999),
        )

        self.add_parameter(
            "current_limit",
            unit="A",
            set_cmd="CONF:CURR:LIMIT {}",
            get_cmd="CURR:LIMIT?",
            get_parser=float,
            vals=Numbers(0, 80),
        )  # what are good numbers here?

        self.add_parameter(
            "field_limit",
            set_cmd=self.current_limit.set,
            get_cmd=self.current_limit.get,
            scale=1 / float(self.ask("COIL?")),
        )

        # Add current solenoid parameters
        # Note that field is validated in set_field
        self.add_parameter(
            "field", get_cmd="IMAG?", get_parser=float, set_cmd=self.set_field
        )
        self.add_parameter(
            "ramp_rate", get_cmd=self._get_ramp_rate, set_cmd=self._set_ramp_rate
        )
        self.add_parameter("setpoint", get_cmd="FIELD:TARG?", get_parser=float)
        self.add_parameter(
            "is_quenched", get_cmd="QU?", val_mapping={True: 1, False: 0}
        )
        self.add_function("reset_quench", call_cmd="QRESET")
        self.add_function("set_quenched", call_cmd="QU 1")
        self.add_parameter(
            "ramping_state",
            get_cmd="STATE?",
            get_parser=int,
            val_mapping={
                "ramping": 1,
                "holding": 2,
                "paused": 3,
                "manual up": 4,
                "manual down": 5,
                "zeroing current": 6,
                "quench detected": 7,
                "at zero current": 8,
                "heating switch": 9,
                "cooling switch": 10,
            },
        )
        self.add_parameter(
            "ramping_state_check_interval",
            initial_value=0.05,
            unit="s",
            vals=Numbers(0, 10),
            set_cmd=None,
        )

        # Add persistent switch
        switch_heater = Cryo4GSwitchHeater(self)
        self.add_submodule("switch_heater", switch_heater)

        # Add interaction functions
        self.add_function("get_error", call_cmd="SYST:ERR?")
        self.add_function("ramp", call_cmd="RAMP")
        self.add_function("pause", call_cmd="PAUSE")
        self.add_function("zero", call_cmd="ZERO")

        # Correctly assign all units
        self._update_units()

        self.connect_message()

    def _update_units(self, field_units: FieldUnits) -> None:
        # Convert the enum to its string value before sending it to the instrument.
        unit_str: str = field_units.value
        self.write(f"UNITS {unit_str}")

    def _sleep(self, t: float) -> None:
        """
        Sleep for a number of seconds t. If we are or using
        the PyVISA 'sim' backend, omit this
        """

        simmode = getattr(self, "visabackend", False) == "sim"

        if simmode:
            return
        else:
            time.sleep(t)

    def _can_start_ramping(self) -> bool:
        """
        Check the current state of the magnet to see if we can start ramping
        """
        if self.is_quenched():
            logging.error(f"{__name__}: Could not ramp because of quench")
            return False

        if self.switch_heater.in_persistent_mode():
            logging.error(f"{__name__}: Could not ramp because persistent")
            return False

        state = self.ramping_state()
        if state == "ramping":
            # If we don't have a persistent switch, or it's warm
            if not self.switch_heater.enabled():
                return True
            elif self.switch_heater.state():
                return True
        elif state in ["holding", "paused", "at zero current"]:
            return True

        logging.error(f"{__name__}: Could not ramp, state: {state}")
        return False

    def set_field(
        self, value: float, *, block: bool = True, perform_safety_check: bool = True
    ) -> None:
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
        field_lim = self.coil_constant * self.current_limit()
        if np.abs(value) > field_lim:
            msg = "Aborted _set_field; {} is higher than limit of {}"
            raise ValueError(msg.format(value, field_lim))

        # If part of a parent driver, set the value using that driver
        #if self._parent_instrument is not None and perform_safety_check:
        #    self._parent_instrument._request_field_change(self, value)
        #    return

        # Check we can ramp
        if not self._can_start_ramping():
            raise Cryo4GException(
                f"Cannot ramp in current state: state is {self.ramping_state()}"
            )

        # Then, do the actual ramp
        self.pause()
        # Set the ramp target
        self.write(f"CONF:FIELD:TARG {value}")

        # If we have a persistent switch, make sure it is resistive
        if self.switch_heater.enabled():
            if not self.switch_heater.state():
                raise Cryo4GException("Switch heater is not on")
        self.ramp()

        # Check if we want to block
        if not block:
            return

        # Otherwise, wait until no longer ramping
        self.log.debug(f"Starting blocking ramp of {self.name} to {value}")
        exit_state = self.wait_while_ramping()
        self.log.debug("Finished blocking ramp")
        # If we are now holding, it was successful
        if exit_state != "holding":
            msg = "_set_field({}) failed with state: {}"
            raise Cryo4GException(msg.format(value, exit_state))

    def wait_while_ramping(self) -> str:

        while self.ramping_state() == "ramping":
            self._sleep(self.ramping_state_check_interval())

        return self.ramping_state()

    def _get_ramp_rate(self) -> float:
        """Return the ramp rate of the first segment in Tesla per second"""
        results = self.ask("RAMP:RATE:FIELD:1?").split(",")
        return float(results[0])

    def _set_ramp_rate(self, rate: float) -> None:
        """Set the ramp rate of the first segment in Tesla per second"""
        if rate > self.field_ramp_limit():
            raise ValueError(
                f"{rate} {self.ramp_rate.unit} "
                f"is above the ramp rate limit of "
                f"{self.field_ramp_limit()} "
                f"{self.field_ramp_limit()}"
            )
        self.write("CONF:RAMP:RATE:SEG 1")
        self.write(f"CONF:RAMP:RATE:FIELD 1,{rate},0")

    def _update_ramp_rate_limit(
        self, new_current_rate_limit: float, update: bool = True
    ) -> None:
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

    def _update_coil_constant(self, new_coil_constant: float | None = None) -> float:
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
        self.field_ramp_limit.scale = 1 / new_coil_constant
        self.field_limit.scale = 1 / new_coil_constant

        # Return new coil constant
        return new_coil_constant

    def _update_units(
        self, ramp_rate_units: int | None = None, field_units: int | None = None
    ) -> None:
        # Get or set units on device
        if ramp_rate_units is None:
            ramp_rate_units_int: str = self.ramp_rate_units()
        else:
            self.write(f"UNITS {ramp_rate_units}")
            ramp_rate_units_int = self.ramp_rate_units.inverse_val_mapping[
                ramp_rate_units
            ]
        if field_units is None:
            field_units_int: str = self.field_units()
        else:
            self.write(f"UNITS {field_units}")
            field_units_int = self.field_units.inverse_val_mapping[field_units]

        # Map to shortened unit names
        ramp_rate_units_short = CryomagneticsModel4G._SHORT_UNITS[ramp_rate_units_int]
        field_units_short = CryomagneticsModel4G._SHORT_UNITS[field_units_int]

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
            self.current_ramp_limit.scale = 1 / 60
        else:
            self.current_ramp_limit.scale = 1

        # If the field units change, the value of the coil constant also
        # changes, hence we read the new value of the coil constant from the
        # instrument via the `coil_constant` parameter (which in turn also
        # updates settings of some parameters due to the fact that the coil
        # constant changes)
        self.coil_constant()

    def write_raw(self, cmd: str) -> None:

        try:
            super().write_raw(cmd)
        except VisaIOError as err:
            # The ami communication has found to be unstable
            # so we retry the communication here
            msg = f"Got VisaIOError while writing {cmd} to instrument."
            if self._RETRY_WRITE_ASK:
                msg += f" Will retry in {self._RETRY_TIME} sec."
            self.log.exception(msg)
            if self._RETRY_WRITE_ASK:
                time.sleep(self._RETRY_TIME)
                self.device_clear()
                super().write_raw(cmd)
            else:
                raise err

    def ask_raw(self, cmd: str) -> str:

        try:
            result = super().ask_raw(cmd)
        except VisaIOError as err:
            # The ami communication has found to be unstable
            # so we retry the communication here
            msg = f"Got VisaIOError while asking the instrument: {cmd}"
            if self._RETRY_WRITE_ASK:
                msg += f" Will retry in {self._RETRY_TIME} sec."
            self.log.exception(msg)
            if self._RETRY_WRITE_ASK:
                time.sleep(self._RETRY_TIME)
                self.device_clear()
                result = super().ask_raw(cmd)
            else:
                raise err
        return result


class Cryo4G(CryomagneticsModel4G):
    pass


