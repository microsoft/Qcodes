from __future__ import annotations

import logging
import time
from enum import Enum
from string import ascii_letters
from typing import Callable, ClassVar, list, TypeVar

from pyvisa import VisaIOError
from dataclasses import dataclass

from qcodes.instrument import VisaInstrument
from qcodes.validators import Numbers

log = logging.getLogger(__name__)
T = TypeVar("T")

@dataclass
class StatusByte:
    sweep_mode_active: bool = False
    standby_mode_active: bool = False
    quench_condition_present: bool = False
    power_module_failure: bool = False
    message_available: bool = False
    extended_status_byte: bool = False
    master_summary_status: bool = False
    menu_mode: bool = False

class SweepMode(Enum):
    UP = "sweep up"
    DOWN = "sweep down"
    PAUSED = "sweep paused"
    ZEROING = "zeroing"

@dataclass
class SweepState:
    mode: SweepMode = SweepMode.PAUSED
    fast: bool = False

class Cryo4GException(Exception):
    pass

class Cryo4GWarning(UserWarning):
    pass

#class RangeRatePair(BaseModel):
#    range_limit: float
#    rate: float
#
#
#class RangesModel(BaseModel):
#    ranges: conlist(RangeRatePair, min_items=1)


class CryomagneticsModel4G(VisaInstrument):
    _SHORT_UNITS: ClassVar[dict[str, str]] = {
        "seconds": "s",
        "minutes": "min",
        "tesla": "T",
        "gauss": "G",
        "amps": "A",
    }

    def __init__(
        self,
        name: str,
        address: str,
        reset: bool = False,
        terminator: str = "\r\n",
        current_ramp_limit: float | None = None,
        current_ramp_limits_per_range: list[float] | None = None,
        **kwargs,
    ):
        super().__init__(
            name,
            address,
            terminator=terminator,
            **kwargs,
        )
        self._parent_instrument = None

        self.add_function("reset", call_cmd="*RST")
        if reset:
            self.reset()

        self.add_parameter(
            name="units",
            set_cmd="UNITS {}",
            get_cmd="UNITS?",
            get_parser=str,
            vals=Enum("A", "G", "T"),
            docstring="Field Units",
        )

        self.add_parameter(
            name="llim",
            unit="T",
            set_cmd=self.set_llim,
            get_cmd=self.get_llim,
            get_parser=float,
            vals=Numbers(-90.001, 0),
            docstring="Lower Ramp Limit",
        )

        self.add_parameter(
            name="ulim",
            unit="T",
            set_cmd=self.set_ulim,
            get_cmd=self.get_ulim,
            get_parser=float,
            vals=Numbers(0, 90.001),
            docstring="Upper Ramp Limit",
        )

        self.add_parameter(
            name="field",
            unit="T",
            set_cmd=self.set_field,
            get_cmd=self.get_field,
            get_parser=float,
            vals=Numbers(-90.001, 90.001),
            docstring="Field",
        )

        self.add_parameter(
            name="Vmag",
            unit="V",
            get_cmd=self.get_vmag,
            get_parser=float,
            vals=Numbers(-10, 10),
            docstring="Magnet sense voltage",
        )

        self.add_parameter(
            name="Vout",
            unit="V",
            get_cmd=self.get_vout,
            get_parser=float,
            vals=Numbers(-12.8, 12.8),
            docstring="Magnet output voltage",
        )

        self.add_parameter(
            name="Iout",
            unit="kG",
            get_cmd=self.get_Iout,
            get_parser=float,
            vals=Numbers(-90.001, 90.001),
            docstring="Magnet output field/current",
        )

        self.add_parameter(
            name="ranges",
            unit="A",
            get_cmd=self.get_ranges,
            set_cmd=self.set_ranges,
            get_parser=list,
            vals=Numbers(-90.001, 90.001),
            docstring="Ramp rate ranges",
        )

        self.add_parameter(
            name="rate",
            unit="T/s",
            get_cmd=self.get_rate,
            set_cmd=self.set_rate,
            get_parser=list,
            vals=Numbers(-90.001, 90.001),
            docstring="Ramp rate ranges",
        )

        self.status = self._get_status_byte()

        self.add_function("get_error", call_cmd="SYST:ERR?")
        self.add_function('QReset', call_cmd='QRESET')
        self.add_function('remote', call_cmd='REMOTE')

        self.remote()
        self.units('T')

    def _get_status_byte(self, status_byte: int) -> StatusByte:
        return StatusByte(
            sweep_mode_active=bool(status_byte & 1),
            standby_mode_active=bool(status_byte & 2),
            quench_condition_present=bool(status_byte & 4),
            power_module_failure=bool(status_byte & 8),
            message_available=bool(status_byte & 16),
            extended_status_byte=bool(status_byte & 32),
            master_summary_status=bool(status_byte & 64),
            menu_mode=bool(status_byte & 128),
        )

    def _can_start_ramping(self) -> bool:
        if self.status.quench_condition_present:
            logging.error(f"{__name__}: Could not ramp because of quench")
            return False
        if self.status.standby_mode_active:
            logging.error(f"{__name__}: Standby mode active, cannot ramp")
            return False
        if self.status.power_module_failure:
            logging.error(f"{__name__}: Could not ramp power module failure detected")
            return False

        state = self.ramping_state()
        if state == "ramping":
            if not self.switch_heater.enabled():
                return True
            elif self.switch_heater.state():
                return True
        elif state in ["holding", "paused", "at zero current"]:
            return True

        logging.error(f"{__name__}: Could not ramp, state: {state}")
        return False

    def set_llim(self, value: float) -> None:
        self._set_limit("LLIM", value)

    def get_llim(self) -> float:
        return self._get_limit("LLIM?")

    def set_ulim(self, value: float) -> None:
        self._set_limit("ULIM", value)

    def get_ulim(self) -> float:
        return self._get_limit("ULIM?")

    def set_field(self, value: float, block: bool = True) -> None:
        if self.units() == "A":
            raise ValueError("Current units are set to Amperes (A). Cannot retrieve magnetic field in these units.")

        value_in_kG = value * 10

        if value_in_kG == 0:
            self.sweep('ZERO')
        elif value_in_kG < 0:
            self.set_llim(value_in_kG)
            self.sweep('DOWN')
        elif value_in_kG > 0:
            self.set_ulim(value_in_kG)
            self.sweep('UP')

        if block:
            self._wait_for_field_setpoint(value_in_kG)

    def get_field(self) -> float:
        if self.units() == "A":
            raise ValueError("Current units are set to Amperes (A). Cannot retrieve magnetic field in these units.")

        raw_output = self.ask_raw("IMAG?")
        filtered_output = ''.join(filter(lambda x: x.isdigit() or x == '.', raw_output))
        try:
            field_value_in_kG = float(filtered_output)
        except ValueError:
            raise ValueError("Failed to convert the instrument response to a float.")

        return field_value_in_kG / 10

    def sweep(self, direction: str) -> None:
        if direction.upper() in ["UP", "DOWN", "ZERO"]:
            self.write_raw(f"SWEEP {direction.upper()}")
        else:
            raise ValueError(
                "Invalid sweep direction. Expected 'UP', 'DOWN', or 'ZERO'."
            )

    def _wait_for_field_setpoint(self, setpoint: float, tolerance: float = 0.002) -> None:
        current_field = self.get_field()
        while abs(setpoint - current_field) > tolerance:
            time.sleep(5)
            current_field = self.get_field()

    def get_ranges(self) -> list[RangeRatePair]:
        ranges = []
        for range_index in range(5):
            range_limit = float(self.ask_raw(f"RANGE? {range_index}"))
            rate = float(self.ask_raw(f"RATE? {range_index}"))
            ranges.append(RangeRatePair(range_limit=range_limit, rate=rate))
        return ranges

    def set_ranges(self, ranges_input: list[dict]) -> None:
        try:
            ranges_model = RangesModel(ranges=[RangeRatePair(**range_rate) for range_rate in ranges_input])
        except ValueError as e:
            raise ValueError(f"Invalid input for ranges: {e}")

        for range_index, range_rate_pair in enumerate(ranges_model.ranges):
            self.write_raw(f"RANGE {range_index}, {range_rate_pair.range_limit}")
            self.write_raw(f"RATE {range_index}, {range_rate_pair.rate}")

    def _set_limit(self, command: str, value: float) -> None:
        if self.units() == "T":
            value_in_kG = value * 10
        else:
            value_in_kG = value

        self.write_raw(f"{command} {value_in_kG}")

    def _get_limit(self, command: str) -> float:
        output = self.ask_raw(command)
        letters = str.maketrans('', '', ascii_letters)
        output = output.translate(letters)
        try:
            numeric_output = float(output)
        except ValueError:
            raise ValueError(f"Failed to convert the instrument response to a float: '{output}'")

        if self.units() == "T":
            return numeric_output / 10
        else:
            return numeric_output

    _RETRY_COUNT = 3
    _RETRY_TIME = 1

    def _retry_communication(
        self, communication_method: Callable, cmd: str
    ) -> int | str | None:
        for attempt in range(self._RETRY_COUNT):
            try:
                return communication_method(cmd)
            except VisaIOError as err:
                if attempt < self._RETRY_COUNT - 1:
                    self.log.warning(f"Attempt {attempt + 1} failed for command {cmd}. "
                                     f"Retrying in {self._RETRY_TIME} seconds...")
                    time.sleep(self._RETRY_TIME)
                    self.device_clear()
                else:
                    self.log.exception(f"All {self._RETRY_COUNT} attempts failed for command {cmd}.")
                    raise err

    def write_raw(self, cmd: str) -> None:
        self._retry_communication(super().write_raw, cmd)

    def ask_raw(self, cmd: str) -> str | int | None:
        return self._retry_communication(super().ask_raw, cmd)

class Cryo4G(CryomagneticsModel4G):
    pass
