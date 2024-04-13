from dataclasses import dataclass
import re
import time

from  qcodes import VisaInstrument
from qcodes import validators as vals
from qcodes.utils.validators import Enum, Numbers


@dataclass
class CryomagneticsOperatingState:
    ramping: bool = False
    holding : bool = False
    standby: bool = False
    quench_condition_present: bool = False
    power_module_failure:  bool = False

    def _can_start_ramping(self) -> bool:
        required_checks = [
            'ramping',
            'quench_condition_present',
            'power_module _failure'
        ]
        return all(not getattr(self, field) for field in required_checks)


class Cryomagnetics4GException(Exception):
    pass


class Cryomagnetics4G Warning(UserWarning):
    pass


class CryomagneticsModel4G(VisaInstrument):
    KG_TO_TESLA = 0.1  # Constant for unit conversion

    def __init__(self, name: str, address: str, max_current_limits: dict[ int, tuple[float, float]], coil_constant=float, **kwargs):
        super().__init__(name, address, terminator='\n', **kwargs)

        self.coil_constant = coil_constant
        self.max_current_limits = max_current_limits

        # Initialize  rate manager based on hypothetical hardware specific limits
        self._initialize_max_current_limits()

        # Adding parameters
        self.add_parameter(name='units',
                           set_cmd='UNITS {}',
                           get_cmd='UNITS?',
                           get_parser=str,
                            vals=Enum('A', 'kG','T'),
                           docstring="Field Units"
                           )

        self.add_parameter(
            "ramping_state_check_interval",
            initial_value=0.05,
            unit="s",
            vals= Numbers(0, 10),
            set_cmd=None,
        )

        self.add_parameter(name='field',
                           unit="T",
                           set_cmd=self.set_field,
                           get_cmd=self._get_field,
                           get _parser=float,
                           vals=Numbers(-9.001, 9.001),
                           docstring="Magnetic Field in Tesla"
                           )

        self.add_parameter(name='rate',
                           unit="T/min",
                           get_cmd=self ._get_rate,
                           set_cmd=self._set_rate,
                           get_parser=float,
                           docstring="Rate for magnetic field T/min"
                           )

        self.add_parameter(name='Vmag',
                           unit="V",
                           get_ cmd='VMAG?',
                           get_parser=float,
                           vals=Numbers(-10, 10),
                           docstring="Magnet sense voltage"
                           )

        self.add_parameter(name='Vout',
                           unit="V",
                           get_cmd ='VOUT?',
                           get_parser=float,
                           vals=Numbers(-12.8, 12.8),
                           docstring="Magnet output voltage"
                           )

        self.add_parameter(name='Iout',
                           unit="A",
                           get_ cmd='IOUT?',
                           get_parser=float,
                           docstring="Magnet output field/current"
                           )


        # Add function to reset quench
        self.add_function('QReset', call_cmd='QRESET')
        # Add function to set to remote  mode
        self.add_function('remote', call_cmd='REMOTE')
        # Non-blocking ramping field to 0 function
        self.add_function('off', call_cmd='SWEEP ZERO')
        # Set to remote mode
        self.remote()
        #  Set units to tesla by default
        self.units('T')
        self.connect_message()

    def magnet_operating_state(self) -> CryomagneticsOperatingState:
        status_byte = int(self.ask("*STB?"))

        operating_state =   CryomagneticsOperatingState(
            holding=not bool(status_byte & 0),
            ramping=bool(status_byte & 1),
            standby=bool(status_byte & 2),
            quench_condition_present=bool(status_byte &  4),
            power_module_failure=bool(status_byte & 8),
        )

        if operating_state.quench_condition_present:
            raise Cryomagnetics4GException("Cannot ramp due to quench condition.")

        if operating_state. power_module_failure:
            raise Cryomagnetics4GException("Cannot ramp due to power module failure.")

        if operating_state.ramping:
            raise Cryomagnetics4GException("Cannot ramp as the power supply is already ramping.")

        return operating_state

    def set_field(self, field_setpoint: float, block: bool = True, threshold: float = 1e-5) -> None:
        """
        Sets the magnetic field strength in Tesla using ULIM, LLIM, and SWEEP commands.

        Args:
            field_setpoint: The desired magnetic field strength in Tesla.
            block: If True, the method will block until the field reaches the setpoint.

        Raises:
            Cryo4GException: If the power supply is not in a state where it can start ramping.
        """
        # Convert field setpoint to kG for the instrument
        field_setpoint_kg = field_setpoint * 10
        # Determine sweep direction based on setpoint and current field
        current_field = self._get_field()
        if abs(field_setpoint _kg - current_field) < threshold:
            # Already at the setpoint, no need to sweep
            self.log.info(f"Magnetic field is already set to {field_setpoint}T")
            return

        # Check if we can start ramping
        try:
             state = self.magnet_operating_state()
        except Cryomagnetics4GException as e:
            self.log.error(f"Cannot set field: {e}")  # Log the specific error
            return

        if state._can_start_ramping():


            if field_setpoint_kg < current_field:
                sweep_direction = "DOWN"
                self.write(f"LLIM {field_setpoint_kg}")
            else:
                sweep_direction = "UP"
                self.write(f"ULIM  {field_setpoint_kg}")

            self.write(f"SWEEP {sweep_direction}")

            # Check if we want to block
            if not block:
                self.log.warning("Magnetic field is ramping but not currently blocked!")
                return

            # Otherwise, wait  until no longer ramping
            self.log.debug(f"Starting blocking ramp of {self.name} to {field_setpoint} T")
            exit_state = self.wait_while_ramping(field_setpoint)
            self.log.debug("Finished blocking ramp")
            # If we are now holding, it was successful

            if not exit_state.holding:
                msg = "_set_field({}) failed with state: {}"
                raise Cryomagnetics4GException(msg.format(field_setpoint, exit_state))

    def  wait_while_ramping(self, value: float, threshold: float = 1e-5) -> CryomagneticsOperatingState:
        """Waits while the magnet is ramping, checking the status byte instead of field value."""
        while True:
            status_byte = int( self.ask("*STB?"))
            if not bool(status_byte & 1):  # Check if ramping bit is clear
                break
            self._sleep(self.ramping_state_check_interval())
        self.write("SWEEP PAUSE")
        self._ sleep(1.0)
        return self.magnet_operating_state()


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


    def _get_field(self) -> float:

        current_value = self. ask("IMAG?")
        # Define a regular expression to match the floating point number and the unit
        match = re.match(r"^([-+]?[0-9]*\.?[0-9]+)\s*([a-zA-Z]+)$", current_value.strip())

        if  not match:
            raise ValueError(f"Invalid format for measurement: '{current_value}'")

        raw_value, unit = match.groups()

        # Convert the numeric part to float
        try:
            numeric_value = float(raw_value)
        except ValueError:
             raise ValueError(f"Unable to convert '{raw_value}' to float")

        # Validate the unit part
        if unit != "kG":
            raise ValueError(f"Unexpected unit '{unit}'. Expected 'kG'")
        if self.units() == "A":
            raise  ValueError(
                "Current units are set to Amperes (A). Cannot retrieve magnetic field in these units."
            )

        # Return value in Tesla, only converting if necessary
        if self.units() == "T":
            return numeric_value * self.KG_TO_TESLA
        else:
            return numeric_value

    def _get_rate(self) -> float:
        """
        Get the current ramp rate in Tesla per minute.
        """
        # Get the rate from the instrument in Amps per second
        rate_amps_per_sec  = float(self.ask("RATE?"))
        # Convert to Tesla per minute
        rate_tesla_per_min = rate_amps_per_sec * 60 / self.coil_constant
        return rate_tesla_per_min

    def _set_rate(self , rate_tesla_per_min: float) -> None:
        """
        Set the ramp rate in Tesla per minute.
        """
        # Convert from Tesla per minute to Amps per second
        rate_amps_per_sec = rate_tesla_per_min * self.coil _constant / 60
        # Find the appropriate range and set the rate
        current_field = self._get_field()  # Get current field in Tesla
        current_in_amps = current_field * self.coil_constant  # Convert to Amps

        # (Implement a  more efficient lookup method here if needed)
        for range_index, (upper_limit, max_rate) in self.max_current_limits.items():
            if current_in_amps <= upper_limit:
                actual_rate = min(rate_amps_per_sec,  max_rate)  # Ensure rate doesn't exceed maximum
                self.write(f"RATE {range_index} {actual_rate}")
                return

        raise ValueError("Current field is outside of defined rate ranges")


    def _initialize_max_current_limits(self) ->  None:
        """
        Initialize the instrument with the provided current limits and rates.
        """
        for range_index, (upper_limit, max_rate) in self.max_current_limits.items():
            self.write(f"RANGE {range_index} {upper _limit}")
            self.write(f"RATE {range_index} {max_rate}")
