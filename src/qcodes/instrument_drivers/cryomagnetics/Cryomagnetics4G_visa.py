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
from typing import Any, Callable, ClassVar, TypeVar, cast, List

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
from typing import List
from pydantic import BaseModel, conlist
#
log = logging.getLogger(__name__)

T = TypeVar("T")``
#




class Cryo4GException(Exception):
    pass


class Cryo4GWarning(UserWarning):
    pass


class RangeRatePair(BaseModel):
    range_limit: float
    rate: float

class RangesModel(BaseModel):
    ranges: conlist(RangeRatePair, min_items=1)  # Ensure at least one range-rate pair is provided

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
            max_rates: List[RangeRatePair],
            coil_constant: float,
            terminator: str = "\r\n", 
            reset: bool = False,  # Add reset parameter with a default value
            **kwargs
        ) -> None:

        """
        Initialize the Cryomagnetics 4G instrument driver.

        Args:
            name (str): Name of the instrument.
            address (str): VISA address of the instrument.
            max_rates (List[RangeRatePair]): List of maximum ramp rates as RangeRatePair objects.
            coil_constant (float): Coil constant for unit conversion.
            terminator (str, optional): Command terminator. Defaults to "\r\n".
        """
        super().__init__(name, address, terminator=terminator, **kwargs)

        # Store the coil configuration
        self.max_rates = max_rates
        self.coil_constant = coil_constant

        self._parent_instrument = None

        # Add reset function
        self.add_function("reset", call_cmd="*RST")
        if reset:
            self.reset()


        self.add_parameter(name='units',
                           set_cmd='UNITS {}',
                           get_cmd='UNITS?',
                           get_parser=str,
                           vals=Enum('A', 'G','T'),
                           docstring="Field Units"
                           )

        self.add_parameter(name='llim',
                           unit="T",
                           set_cmd=self.set_llim,
                           get_cmd=self.get_llim,
                           get_parser=float,
                           vals=Numbers(-90.001, 0),
                           docstring="Lower Ramp Limit"
                           )
        
        self.add_parameter(name='ulim',
                           unit="T",
                           set_cmd=self.set_ulim,
                           get_cmd=self.get_ulim,
                           get_parser=float,
                           vals=Numbers(0,90.001),
                           docstring="Upper Ramp Limit"
                           )
        self.add_parameter(name='field',
                           unit="T",
                           set_cmd=self.set_field,
                           get_cmd=self.get_field,
                           get_parser=float,
                           vals=Numbers(-90.001, 90.001),
                           docstring="Field"
                           )


        self.add_parameter(name='Vmag',
                           unit="V",
                           get_cmd=self.get_vmag,
                           get_parser=float,
                           vals=Numbers(-10, 10),
                           docstring="Magnet sense voltage"
                           )
        
        self.add_parameter(name='Vout',
                           unit="V",
                           get_cmd=self.get_vout,
                           get_parser=float,
                           vals=Numbers(-12.8, 12.8),
                           docstring="Magnet output voltage"
                           )
        
        self.add_parameter(name='Iout',
                           unit="kG",
                           get_cmd=self.get_Iout,
                           get_parser=float,
                           vals=Numbers(-90.001, 90.001),
                           docstring="Magnet output field/current"
                           )

        self.add_parameter(name='ranges',
                           unit="A",
                           get_cmd=self.get_ranges,
                           set_cmd=self.set_ranges,
                           get_parser=list,
                           #vals=Numbers(-90.001, 90.001),
                           docstring="Ramp rate ranges"
                           )
                           
        self.add_parameter(name='rate',
                           unit="T/s",
                           get_cmd=self.get_rate,
                           set_cmd=self.set_rate,
                           get_parser=list,
                           #vals=Numbers(-90.001, 90.001),
                           docstring="Ramp rate ranges"
                           )
       

        # Add persistent switch
        switch_heater = Cryo4GSwitchHeater(self)
        self.add_submodule("switch_heater", switch_heater)

        # Add interaction functions
        self.add_function("get_error", call_cmd="SYST:ERR?")
        # Add function to reset quench
        self.add_function('QReset', call_cmd='QRESET')
        # Add function to set to remote mode
        self.add_function('remote', call_cmd='REMOTE')
        # Set to remote mode
        self.remote()
        # Set units to tesla by default
        self.units('T')
        #Set rates to max by default
        self.ranges(self.max_rates)
        self.connect_message()

    def set_llim(self, value: float) -> None:
        """
        Sets the lower limit for the magnetic field sweep.
    
        Args:
            value: The lower limit value in the current unit. If the unit is Tesla, it will be converted to kilogauss for the instrument.
        """
        self._set_limit("LLIM", value)

    def get_llim(self) -> float:
        """
        Get lower limit.

        Returns:
            The lower limit value in Tesla, converted from kilogauss.
        """
        return self._get_limit("LLIM?")


    def set_ulim(self, value: float) -> None:
        """
        Sets the upper limit for the magnetic field sweep.

        Args:
            value: The upper limit value in the current unit. If the unit is Tesla, it will be converted to kilogauss for the instrument.
        """
        self._set_limit("ULIM", value)


    def get_ulim(self) -> float:
        """
        Get upper limit.

        Returns:
            The upper limit value in Tesla, converted from kilogauss.
        """
        return self._get_limit("ULIM?")

    def set_field(self, value: float, block: bool = True) -> None:
        """
        Ramp to a certain field in kilogauss (kG).

        Args:
            value: Field setpoint in Tesla (T), to be converted to kilogauss for the instrument.
            block: Whether to wait until the field has finished setting.
        """
        # Check if the current units are not appropriate for magnetic field (e.g., Amperes)
        if self.units() == "A":
            raise ValueError("Current units are set to Amperes (A). Cannot retrieve magnetic field in these units.")

        # Convert the value to kilogauss
        value_in_kG = value * 10  # Convert Tesla to kilogauss

        # Determine the direction of the sweep based on the value
        if value_in_kG == 0:
            self.sweep('ZERO')
        elif value_in_kG < 0:
            self.set_llim(value_in_kG)
            self.sweep('DOWN')
        elif value_in_kG > 0:
            self.set_ulim(value_in_kG)
            self.sweep('UP')

        # Wait for the field to reach the setpoint if blocking is enabled
        if block:
            self._wait_for_field_setpoint(value_in_kG)

    def get_field(self) -> float:
        """
        Queries and returns the current magnetic field value in Tesla.

        Returns:
            The current magnetic field in Tesla, converted from kilogauss.
        """

        # Check if the current units are not appropriate for magnetic field (e.g., Amperes)
        if self.units() == "A":
            raise ValueError("Current units are set to Amperes (A). Cannot retrieve magnetic field in these units.")
        
        raw_output = self.ask("IMAG?")
        filtered_output = ''.join(filter(lambda x: x.isdigit() or x == '.', raw_output))

        try:
            field_value_in_kG = float(filtered_output)
        except ValueError:
            raise ValueError("Failed to convert the instrument response to a float.")

        # Convert the field value from kilogauss to Tesla
        return field_value_in_kG / 10  # Convert kilogauss to Tesla

    def sweep(self, direction: str) -> None:
        """
        Initiates a magnetic field sweep in the specified direction.    

        Args:
            direction: The direction of the sweep. Expected values are 'UP', 'DOWN', or 'ZERO'.
        """
        # Assuming the command to start a sweep is "SWEEP <direction>"
        # Validate the direction before sending the command
        if direction.upper() in ['UP', 'DOWN', 'ZERO']:
            self.write(f"SWEEP {direction.upper()}")
        else:
             raise ValueError("Invalid sweep direction. Expected 'UP', 'DOWN', or 'ZERO'.")
    
    def _wait_for_field_setpoint(self, setpoint: float, tolerance: float = 0.002) -> None:
        """
        Wait until the field reaches the setpoint within a specified tolerance.

        Args:
            setpoint: The field setpoint.
            tolerance: The tolerance within which the setpoint is considered reached.
        """
        current_field = self.get_field()
        while abs(setpoint - current_field) > tolerance:
            time.sleep(5)
            current_field = self.get_field()


    def get_ranges(self) -> List[RangeRatePair]:
        """
        Queries the instrument for its current ramp rate ranges.

        Returns:
            A list of RangeRatePair instances, where each contains the range limit and rate for each range.
        """
        ranges = []
        for range_index in range(5):  # Assuming there are 5 ranges to query
            range_limit = float(self.ask(f"RANGE? {range_index}"))
            rate = float(self.ask(f"RATE? {range_index}"))
            ranges.append(RangeRatePair(range_limit=range_limit, rate=rate))
        return ranges

    def set_ranges(self, ranges_input: List[dict]) -> None:
        """
        Sets the ramp rate ranges on the instrument using a Pydantic model for validation.

        Args:
            ranges_input: A list of dictionaries, where each dictionary contains the keys 'range_limit' and 'rate'.
        """
        try:
            ranges_model = RangesModel(ranges=[RangeRatePair(**range_rate) for range_rate in ranges_input])
        except ValueError as e:
            raise ValueError(f"Invalid input for ranges: {e}")

        for range_index, range_rate_pair in enumerate(ranges_model.ranges):
            self.write(f"RANGE {range_index}, {range_rate_pair.range_limit}")
            self.write(f"RATE {range_index}, {range_rate_pair.rate}")

    def _set_limit(self, command: str, value: float) -> None:
        """
        Helper function to set a limit (lower or upper) on the instrument, with unit check.

        Args:
            command (str): The command prefix to send to the instrument to set the limit (e.g., "LLIM" or "ULIM").
            value (float): The limit value. If the current unit is Tesla, the value is converted to kilogauss for the instrument.
        """
        # Check if the current unit is Tesla and convert if necessary
        if self.units() == "T":
            value_in_kG = value * 10  # Convert Tesla to kilogauss
        else:
            value_in_kG = value

        # Send the command with the value in the appropriate unit
        self.write(f"{command} {value_in_kG}")

    def _get_limit(self, command: str) -> float:
        """
        Helper function to get a limit (lower or upper) from the instrument.

        Args:
            command (str): The command to send to the instrument to query the limit.

        Returns:
            The limit value in Tesla if the units are set to Tesla, otherwise in the instrument's default unit.
        """
        output = self.ask(command)
        # Remove any ASCII letters from the response to isolate the numeric value
        letters = str.maketrans('', '', ascii_letters)
        output = output.translate(letters)

        try:
            numeric_output = float(output)
        except ValueError:
            raise ValueError(f"Failed to convert the instrument response to a float: '{output}'")

        # Convert from kilogauss to Tesla if the current unit is set to Tesla
        if self.units() == "T":
            return numeric_output / 10  # Convert kilogauss to Tesla
        else:
            return numeric_output

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


