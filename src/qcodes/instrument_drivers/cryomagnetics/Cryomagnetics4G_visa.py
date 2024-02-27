from __future__ import annotations

import logging
import numbers
import time
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from enum import Enum
from functools import partial
from typing import Any, Callable, ClassVar, TypeVar, cast

#
import numpy as np
from pyvisa import VisaIOError

#
from qcodes import validators
from qcodes.instrument import Instrument, InstrumentChannel, VisaInstrument

#from qcodes.math_utils import FieldVector
from qcodes.parameters import Parameter
from qcodes.utils import QCoDeSDeprecationWarning
from qcodes.validators import Anything, Bool, Enum, Ints, Numbers

#
log = logging.getLogger(__name__)

T = TypeVar("T")
#




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

        # Not sure about this cmd being correct
        self.add_parameter(
            "mode",
            label="Persistent Mode",
            get_cmd=self.in_persistent_mode,
        )

        self.add_parameter(
            "heat_time",
            label="Heating Time",
            unit="s",
            get_cmd=lambda: self._heat_time,
            set_cmd=lambda x: setattr(self, "_heat_time", x),
            vals=Ints(0, 120),
            initial_value=0
        )

        self.add_parameter(
            "cool_time",
            label="Cooling Time",
            unit="s",
            get_cmd=lambda: self._cool_time,
            set_cmd=lambda x: setattr(self, "_cool_time", x),
            vals=Ints(0, 3600),
            initial_value=0
        )

    def in_persistent_mode(self) -> bool:
        """
        Checks if the switch heater is in persistent mode.
        Persistent mode is when the switch heater is enabled and on
        This allows the switch heater to remain on even after
        a persistent mode timeout occurs.

        Returns:
            bool: True if in persistent mode, False otherwise.
        """

        return (
            self.get_parameter("mode").get_latest() == "Manual" 
            and self.get_parameter("state").get_latest() == "On"
        )

    
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
        "gauss": "G",
        "amps": "A", 
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

        self._parent_instrument = None

        # Add reset function
        self.add_function("reset", call_cmd="*RST")
        if reset:
            self.reset()

        # Add parameters setting instrument units
        self.add_parameter(
            "ramp_rate_units",
            get_cmd=False,
            set_cmd=(lambda units: self._update_units(ramp_rate_units=units)),
            vals=validators.Enum("seconds", "minutes"),
        )

        self.add_parameter(
            "units",
            get_cmd="UNITS?",
            set_cmd=self._set_field_units,  # New method to handle unit conversion and checks
            vals=validators.Enum("Gauss", "Amps"),
        )

        # need a method for field units 
        self.add_parameter(
            "field_units",
        )

        # Set programmatic safety limits
        self.add_parameter(
            "current_ramp_limit",
        #    get_cmd=lambda: self._current_ramp_limit,
        #    set_cmd=self._update_ramp_rate_limit,
        #    unit="A/s",
        )
        self.add_parameter(
            "field_ramp_limit",
        #    get_cmd=lambda: self.current_ramp_limit(),
        #    set_cmd=lambda x: self.current_ramp_limit(x),
        #    scale=1 / float(self.ask("COIL?")),
        #    unit="T/s",
        )

        self.add_parameter(
            "field_ramp_limit",
        #    get_cmd=lambda: self.current_ramp_limit(),
        #    set_cmd=lambda x: self.current_ramp_limit(x),
        #    scale=1 / float(self.ask("COIL?")),
            unit="T/s",
        )
        #if current_ramp_limit is None:
        #    self._update_ramp_rate_limit(
        #        AMIModel430._DEFAULT_CURRENT_RAMP_LIMIT, update=False
        #    )
        #else:
        #    self._update_ramp_rate_limit(current_ramp_limit, update=False)

        self.add_parameter(
            "coil_constant",
            get_cmd=self._update_coil_constant,
            set_cmd=self._update_coil_constant,
            vals=Numbers(0.001, 999.99999),
        )

        #Need private method for getting the current limit at the range it's in 
        self.add_parameter(
            "current_limit",
        ) 

        #Need private method for getting the field limit at the range it's in
        self.add_parameter(
            "field_limit",
        )

        # Note that field is validated in set_field
        self.add_parameter(
            "field",

        )
        self.add_parameter(
            "ramp_rate",
        )
         
        self.add_parameter(
            "is_quenched",
            #TODO add in correct CMD
            #get_cmd="QU?", val_mapping={True: 1, False: 0}
        )


        #TODO DO we need these?
        #self.add_function("reset_quench", call_cmd="QU 0")
        #self.add_function("set_quenched", call_cmd="QU 1")
        
        
        self.add_parameter(
            "ramping_state",
            #TODO get correct state mapping
            #get_cmd="STATE?",
            #get_parser=int,
            #val_mapping={
            #    "ramping": 1,
            #    "holding": 2,
            #    "paused": 3,
            #    "manual up": 4,
            #    "manual down": 5,
            #    "zeroing current": 6,
            #    "quench detected": 7,
            #    "at zero current": 8,
            #    "heating switch": 9,
            #    "cooling switch": 10,
            #},
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

    def set_field(
        self, value: float, *, block: bool = True, perform_safety_check: bool = True
    ) -> None:
        pass


    # No current reason to have method for converting current units (I think)
    def _update_field_units(
        self, ramp_rate_units: int | None = None, field_units: int | None = None
    ) -> None:
        pass

    def _update_ramp_rate_limit(
        self, new_current_rate_limit: float, update: bool = True
    ) -> None:
        pass
        
    def _update_coil_constant(self, new_coil_constant: float | None = None) -> float:
        pass 

    def wait_while_ramping(self) -> str:
        pass
    
    def _get_ramp_rate(self) -> float:
        pass

    def _set_ramp_rate(self, rate: float) -> None:
        pass

    def _update_coil_constant(self, new_coil_constant: float | None = None) -> float:
        """
        Update the coil constant and relevant scaling factors.
        If new_coil_constant is none, query the coil constant from the
        instrument
        """
        # cryo 4g doesn't have query for coil constant
        #if new_coil_constant is None:
        #    new_coil_constant = float(self.ask("COIL?"))
        #else:
        #    self.write(f"CONF:COIL {new_coil_constant}")

        # Update scaling factors
        self.field_ramp_limit.scale = 1 / new_coil_constant
        self.field_limit.scale = 1 / new_coil_constant

        # Return new coil constant
        return new_coil_constant



    

    _RETRY_COUNT = 3  # Configurable number of retries
    _RETRY_TIME = 1   # Configurable retry delay in seconds

    def _retry_communication(self, communication_method: Callable, cmd: str) ->int | str | None:
        """
        Retries a communication method (write or ask) if a VisaIOError occurs.

        Args:
            communication_method (callable): The super() method to call (write_raw or ask_raw).
            cmd (str): The command to send to the instrument.
        
        Returns:
            The result from the communication method, if any.
        """
        for attempt in range(self._RETRY_COUNT):
            try:
                return communication_method(cmd)
            except VisaIOError as err:
                if attempt < self._RETRY_COUNT - 1:
                    # Log as warning if we're going to retry
                    self.log.warning(f"Attempt {attempt + 1} failed for command {cmd}. "
                                     f"Retrying in {self._RETRY_TIME} seconds...")
                    time.sleep(self._RETRY_TIME)
                    self.device_clear()
                else:
                    # Log as exception if we've exhausted retries
                    self.log.exception(f"All {self._RETRY_COUNT} attempts failed for command {cmd}.")
                    raise err

    def write_raw(self, cmd: str) -> None:
        """
        Write a command to the instrument, with retries for VisaIOError.

        Args:
            cmd (str): The command to send to the instrument.
        """
        self._retry_communication(super().write_raw, cmd)

    def ask_raw(self, cmd: str) -> str:
        """
        Ask the instrument and return the response, with retries for VisaIOError.

        Args:
            cmd (str): The command to send to the instrument.
        
        Returns:
            str: The response from the instrument.
        """
        return self._retry_communication(super().ask_raw, cmd)

class Cryo4G(CryomagneticsModel4G):
    pass


