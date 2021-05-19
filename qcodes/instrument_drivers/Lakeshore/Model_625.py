import time
from typing import Tuple, Optional


from qcodes import VisaInstrument
from qcodes.utils.validators import Numbers, Enum


class Lakeshore625(VisaInstrument):
    """
    Driver for the Lakeshore Model 625 superconducting magnet power supply.

    This class uses T/A and A/s as units.

    Args:
        name (str): a name for the instrument
        coil_constant (float): Coil contant of magnet, in untis of T/A
        field_ramp_rate (float): Magnetic field ramp rate, in units of T/min
        address (str): VISA address of the device
    """

    def __init__(
        self,
        name: str,
        address: str,
        coil_constant: Optional[float] = None,
        field_ramp_rate: Optional[float] = None,
        reset: bool = False,
        terminator: str = "",
        **kwargs,
    ) -> None:

        super().__init__(name, address, terminator=terminator, **kwargs)

        self._create_parameters()

        self.add_function("reset", call_cmd="*RST")
        self.add_function("clear", call_cmd="*CLS")

        if reset:
            self.reset()

        # Defaults
        self.persistent_switch_heater("disabled")
        self.ramp_segments("disabled")
        self.coil_constant_unit("T/A")

        # assign init parameters
        if coil_constant is not None:
            self.coil_constant(coil_constant)
        if field_ramp_rate is not None:
            self.field_ramp_rate(field_ramp_rate)

        # print connect message
        self.connect_message()

    def _create_parameters(self) -> None:
        # Add power supply parameters
        self.add_parameter(
            name="current_limit",
            unit="A",
            set_cmd=self._set_current_limit,
            get_cmd=self._get_current_limit,
            get_parser=float,
            vals=Numbers(0, 60.1),
            docstring="Maximum output current",
        )

        self.add_parameter(
            name="voltage_limit",
            unit="V",
            set_cmd=self._set_voltage_limit,
            get_cmd=self._get_voltage_limit,
            get_parser=float,
            vals=Numbers(0, 5),
            docstring="Maximum compliance voltage",
        )

        self.add_parameter(
            name="current_rate_limit",
            unit="A/s",
            set_cmd=self._set_current_rate_limit,
            get_cmd=self._get_current_rate_limit,
            get_parser=float,
            vals=Numbers(0.0001, 99.999),
            docstring="Maximum current ramp rate",
        )

        self.add_parameter(
            name="voltage",
            unit="V",
            set_cmd="SETV {}",
            get_cmd="RDGV?",
            get_parser=float,
            vals=Numbers(-5, 5),
        )

        self.add_parameter(
            name="current",
            unit="A",
            set_cmd="SETI {}",
            get_cmd="RDGI?",
            get_parser=float,
            vals=Numbers(-60, 60),
        )

        self.add_parameter(
            name="current_ramp_rate",
            unit="A/s",
            set_cmd="RATE {}",
            get_cmd="RATE?",
            get_parser=float,
        )

        self.add_parameter(
            name="ramp_segments",
            set_cmd="RSEG {}",
            get_cmd="RSEG?",
            get_parser=int,
            val_mapping={"disabled": 0, "enabled": 1},
        )

        self.add_parameter(
            name="persistent_switch_heater",
            set_cmd=self._set_persistent_switch_heater_status,
            get_cmd=self._get_persistent_switch_heater_status,
            get_parser=int,
            val_mapping={"disabled": 0, "enabled": 1},
        )

        self.add_parameter(
            name="quench_detection",
            set_cmd=self._set_quench_detection_status,
            get_cmd=self._get_quench_detection_status,
            get_parser=int,
            val_mapping={"disabled": 0, "enabled": 1},
        )

        self.add_parameter(
            name="quench_current_step_limit",
            unit="A/s",
            set_cmd=self._set_quench_current_step_limit,
            get_cmd=self._get_quench_current_step_limit,
            get_parser=float,
            vals=Numbers(0.01, 10),
        )

        self.add_parameter(
            name="ramping_state",
            get_cmd=self._get_ramping_state,
            vals=Enum("ramping", "not ramping"),
        )

        self.add_parameter(
            name="operational_error_status",
            get_cmd=self._get_operational_errors,
            get_parser=str,
        )

        self.add_parameter(
            name="oer_quench",
            get_cmd=self._get_operr_quench_bit,
            get_parser=int,
            val_mapping={"no quench detected": 0, "quench detected": 1},
        )

        # Add solenoid parameters
        self.add_parameter(
            name="coil_constant_unit",
            set_cmd=self._set_coil_constant_unit,
            get_cmd=self._get_coil_constant_unit,
            get_parser=int,
            val_mapping={"T/A": 0, "kG/A": 1},
            docstring="unit of the coil constant, either T/A (default) or kG/A",
        )

        self.add_parameter(
            name="coil_constant",
            unit=self.coil_constant_unit,
            set_cmd=self._update_coil_constant,
            get_cmd=self._get_coil_constant,
            get_parser=float,
            vals=Numbers(0.001, 999.99999),  # what are good numbers here?
        )

        self.add_parameter(
            name="field",
            unit="T",
            set_cmd=self.set_field,
            get_cmd="RDGF?",
            get_parser=float,
        )

        self.add_parameter(
            name="field_ramp_rate",
            unit="T/min",
            set_cmd=self._set_field_ramp_rate,
            get_cmd=self._get_field_ramp_rate,
            get_parser=float,
            docstring="Field ramp rate (T/min)",
        )

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

    # get functions returning several values
    def _get_limit(self) -> Tuple[float, float, float]:
        """
        Limit Output Settings Query

        Returns
        -------
            <current>, <voltage>, <rate>
        """
        raw_string = self.ask("LIMIT?")
        current_limit, voltage_limit, current_rate_limit = raw_string.split(",")
        return (float(current_limit), float(voltage_limit), float(current_rate_limit))

    def _get_persistent_switch_heater_setup(self) -> Tuple[int, float, float]:
        """
        Persistent Switch Heater Parameter Query

        Returns
        -------
            <enable>, <current>, <delay>
        """
        raw_string = self.ask("PSHS?")
        status, psh_current, psh_delay = raw_string.split(",")
        return int(status), float(psh_current), float(psh_delay)

    def _get_quench_detection_setup(self) -> Tuple[int, float]:
        """
        Quench Parameter Query

        Returns
        -------
            <enable>, <rate>
        """
        raw_string = self.ask("QNCH?")
        status, current_step_limit = raw_string.split(",")
        return int(status), float(current_step_limit)

    def _get_field_setup(self) -> Tuple[str, float]:
        """
        Computed Magnetic Field Parameter Query

        Returns
        -------
            <units>, <constant>
        """
        raw_string = self.ask("FLDS?")
        unit, coil_constant = raw_string.split(",")
        return str(unit), float(coil_constant)

    # get functions for parameters
    def _get_current_limit(self) -> float:
        """
        Get maximum allowed output current setting.

        Returns
        -------
            <current>
        """
        current_limit, _, _ = self._get_limit()
        return current_limit

    def _get_voltage_limit(self) -> float:
        """
        Gets maximum allowed compliance voltage setting

        Returns
        -------
            <voltage>
        """
        _, voltage_limit, _ = self._get_limit()
        return voltage_limit

    def _get_current_rate_limit(self) -> float:
        """
        Gets maximum allowed output current ramp rate setting

        Returns
        -------
            <rate>
        """
        _, _, current_rate_limit = self._get_limit()
        return current_rate_limit

    def _get_persistent_switch_heater_status(self) -> int:
        """
        Queries if there is a persistent switch: 0 = Disabled (no PSH), 1 = Enabled

        Returns
        -------
            status
        """
        status, _, _ = self._get_persistent_switch_heater_setup()
        return status

    def _get_quench_detection_status(self) -> int:
        """
        Queries if quench detection is to be used: 0 = Disabled, 1 = Enabled

        Returns
        -------
            status
        """
        status, _ = self._get_quench_detection_setup()
        return status

    def _get_quench_current_step_limit(self) -> float:
        """
        Gets current step limit for quench detection

        Returns
        -------
            <rate>
        """
        _, current_step_limit = self._get_quench_detection_setup()
        return current_step_limit

    def _get_coil_constant(self) -> float:
        """
        Gets magnetic field constant in either T/A or kG/A depending on units

        Returns
        -------
            <constant>
        """
        _, coil_constant = self._get_field_setup()
        return coil_constant

    def _get_coil_constant_unit(self) -> str:
        """
        Gets the units of the magnetic field constant: 0 = T/A, 1 = kG/A

        Returns
        -------
            <units>
        """
        coil_constant_unit, _ = self._get_field_setup()
        return coil_constant_unit

    def _get_field_ramp_rate(self) -> float:
        """
        Gets the field ramp rate in units of T/min

        Returns
        -------
            field_ramp_rate (T/min)
        """
        _, coil_constant = self._get_field_setup()  # in T/A by default
        current_ramp_rate = self.current_ramp_rate()  # in A/s
        field_ramp_rate = current_ramp_rate * coil_constant * 60  # in T/min
        return field_ramp_rate

    def _get_ramping_state(self) -> str:
        """
        Gets the ramping state of the power supply (corresponds to blue LED on panel)
        Is inferred from the status bit register

        Returns
        -------
            ramping state
        """
        operation_condition_register = self.ask("OPST?")
        bin_opst = bin(int(operation_condition_register))[2:]
        if len(bin_opst) < 2:
            rampbit = 1
        else:
            # read second bit, 0 = ramping, 1 = not ramping
            rampbit = int(bin_opst[-2])
        if rampbit == 1:
            return "not ramping"
        else:
            return "ramping"

    def _get_operational_errors(self) -> str:
        """
        Error Status Query

        Returns
        -------
            error status
        """
        error_status_register = self.ask("ERST?")
        # three bytes are read at the same time, the middle one is the operational error status
        operational_error_register = error_status_register.split(",")[1]

        # prepend zeros to bit-string such that it always has length 9
        operr_bit_str = bin(int(operational_error_register))[2:].zfill(9)
        return operr_bit_str

    def _get_operr_quench_bit(self) -> int:
        """
        Returns the operr quench bit

        Returns
        -------
            quench bit
        """
        return int(self._get_operational_errors()[3])

    # set functions for parameters
    def _set_current_limit(self, current_limit_setpoint: float) -> None:
        """
        Sets maximum allowed output current
        """
        _, voltage_limit, current_rate_limit = self._get_limit()
        self.write_raw(
            "LIMIT {}, {}, {}".format(
                current_limit_setpoint, voltage_limit, current_rate_limit
            )
        )

    def _set_voltage_limit(self, voltage_limit_setpoint: float) -> None:
        """
        Sets maximum allowed compliance voltage
        """
        current_limit, _, current_rate_limit = self._get_limit()
        self.write_raw(
            "LIMIT {}, {}, {}".format(
                current_limit, voltage_limit_setpoint, current_rate_limit
            )
        )

    def _set_current_rate_limit(self, current_rate_limit_setpoint: float) -> None:
        """
        Sets maximum allowed output current ramp rate
        """
        current_limit, voltage_limit, _ = self._get_limit()
        self.write_raw(
            "LIMIT {}, {}, {}".format(
                current_limit, voltage_limit, current_rate_limit_setpoint
            )
        )

    def _set_persistent_switch_heater_status(self, status_setpoint: int) -> None:
        """
        Specifies if there is a persistent switch: 0 = Disabled (no PSH), 1 = Enabled
        """
        _, psh_current, psh_delay = self._get_persistent_switch_heater_setup()
        self.write_raw(
            "PSHS {}, {}, {}".format(status_setpoint, psh_current, psh_delay)
        )

    def _set_quench_detection_status(self, status_setpoint: int) -> None:
        """
        Specifies if quench detection is to be used: 0 = Disabled, 1 = Enabled
        """
        _, current_step_limit = self._get_quench_detection_setup()
        self.write_raw("QNCH {}, {}".format(status_setpoint, current_step_limit))

    def _set_quench_current_step_limit(
        self, current_step_limit_setpoint: float
    ) -> None:
        """
        Specifies the current step limit for quench detection
        """
        status, _ = self._get_quench_detection_setup()
        self.write_raw("QNCH {}, {}".format(status, current_step_limit_setpoint))

    def _set_coil_constant(self, coil_constant_setpoint: float) -> None:
        """
        Specifies the magnetic field constant in either T/A or kG/A depending on units
        """
        coil_constant_unit, _ = self._get_field_setup()
        self.write_raw("FLDS {}, {}".format(coil_constant_unit, coil_constant_setpoint))

    def _set_coil_constant_unit(self, coil_constant_unit_setpoint: str) -> None:
        """
        Specifies the units of the magnetic field constant: 0 = T/A, 1 = kG/A
        """
        _, coil_constant = self._get_field_setup()
        self.write_raw("FLDS {}, {}".format(coil_constant_unit_setpoint, coil_constant))

    def _update_coil_constant(self, coil_constant_setpoint: float) -> None:
        """
        Updates the coil_constant and with it all linked parameters
        """
        # read field_ramp_rate before chaning coil constant
        field_ramp_rate = self.field_ramp_rate()
        # set the coil constant
        self._set_coil_constant(coil_constant_setpoint)
        # update the current ramp rate, leaving the field ramp rate unchanged
        current_ramp_rate_setpoint = (
            field_ramp_rate / coil_constant_setpoint / 60
        )  # current_ramp_rate is in A/s
        self.current_ramp_rate(current_ramp_rate_setpoint)

    def _set_field_ramp_rate(self, field_ramp_rate_setpoint: float) -> None:
        """
        Sets the field ramp rate in units of T/min by setting the corresponding current_ramp_rate
        """
        _, coil_constant = self._get_field_setup()  # in T/A by default
        current_ramp_rate_setpoint = (
            field_ramp_rate_setpoint / coil_constant / 60
        )  # current_ramp_rate is in A/s
        self.current_ramp_rate(current_ramp_rate_setpoint)

    def set_field(self, value: float, block: bool = True) -> None:
        """
        Ramp to a certain field

        Args:
            value: field setpoint
            block: Whether to wait until the field has finished setting
        """

        self.write("SETF {}".format(value))
        # Check if we want to block
        if not block:
            return

        # Otherwise, wait until no longer ramping
        self.log.debug(f"Starting blocking ramp of {self.name} to {value}")
        self._sleep(
            0.5
        )  # wait for a short time for the power supply to fall into the ramping state
        while self.ramping_state() == "ramping":
            self._sleep(0.3)
        self._sleep(2.0)
        self.log.debug("Finished blocking ramp")
        return
