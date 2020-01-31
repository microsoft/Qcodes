from typing import cast

from qcodes import VisaInstrument, InstrumentChannel
from qcodes.utils.validators import Enum, Numbers, Ints
from qcodes.utils.helpers import create_on_off_val_mapping


class Sense7510(InstrumentChannel):
    """
    The sense module of the Keithley 7510 DMM, based on the sense module of
    Keithley 2450 SMU

    Args:
        parent
        name
        proper_function: This can be one of modes listed in the dictionary
            "function_modes", e.g.,  "current", "voltage", or "resistance".
            "voltage"/"current" is for DC voltage/current.
            "Avoltage"/"Acurrent" is for AC voltage/current.
            "resistance" is for two-wire measurement of resistance.
            "Fresistance" is for Four-wire measurement of resistance.

            All parameters and methods in this submodule should only be
            accessible to the user if
            self.parent.sense_function.get() == self._proper_function. We
            ensure this through the 'sense' property on the main driver class
            which returns the proper submodule for any given function mode
    """

    function_modes = {
        "voltage": {
            "name": '"VOLT:DC"',
            "unit": 'V',
            "range_vals": Numbers(0.1, 1000),
        },
        "Avoltage": {
            "name": '"VOLT:AC"',
            "unit": 'V',
            "range_vals": Numbers(0.1, 700),
        },
        "current": {
            "name": '"CURR:DC"',
            "unit": 'V',
            "range_vals": Numbers(10e-6, 10),
        },
        "Acurrent": {
            "name": '"CURR:AC"',
            "unit": 'A',
            "range_vals": Numbers(1e-3, 10),
        },
        "resistance": {
            "name": '"RES"',
            "unit": 'V',
            "range_vals": Numbers(10, 1e9),
        },
        "Fresistance": {
            "name": '"FRES"',
            "unit": 'V',
            "range_vals": Numbers(1, 1e9),
        },
    }

    def __init__(self, parent: '', name: str, proper_function: str) -> None:
        super().__init__(parent, name)
        self._proper_function = proper_function
        range_vals = self.function_modes[self._proper_function]["range_vals"]
        unit = self.function_modes[self._proper_function]["unit"]

        self.function = self.parent.sense_function

        self.add_parameter(
            self._proper_function,
            get_cmd=":MEASure?",
            get_parser=float,
            unit=unit,
        )

        self.add_parameter(
            "auto_range",
            get_cmd=f":SENSe:{self._proper_function}:RANGe:AUTO?",
            set_cmd=f":SENSe:{self._proper_function}:RANGe:AUTO {{}}",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0")
        )

        self.add_parameter(
            "range",
            get_cmd=f":SENSe:{self._proper_function}:RANGe?",
            set_cmd=f":SENSe:{self._proper_function}:RANGe {{}}",
            vals=range_vals,
            get_parser=float,
            unit=unit
        )

        self.add_parameter(
            "nplc",
            get_cmd=f":SENSe:{self._proper_function}:NPLCycles?",
            set_cmd=f":SENSe:{self._proper_function}:NPLCycles {{}}",
            vals=Numbers(0.01, 10),
            get_parser=float,
        )

        self.add_parameter(
            "auto_delay",
            get_cmd=f":SENSe:{self._proper_function}:DELay:AUTO?",
            set_cmd=f":SENSe:{self._proper_function}:DELay:AUTO {{}}",
            val_mapping=create_on_off_val_mapping(on_val="ON", off_val="OFF")
        )

        self.add_parameter(
            'user_number',
            get_cmd=None,
            set_cmd=None,
            vals=Ints(1, 5)
        )

        self.add_parameter(
            "user_delay",
            get_cmd=self._get_user_delay,
            set_cmd=self._set_user_delay,
            vals=Numbers(0, 1e4),
            unit='second'
        )

        self.add_parameter(
            "auto_zero",
            get_cmd=f":SENSe:{self._proper_function}:AZERo?",
            set_cmd=f":SENSe:{self._proper_function}:AZERo {{}}",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0")
        )

        self.add_parameter(
            "auto_zero_once",
            set_cmd=f":SENSe:AZERo:ONCE",
        )

        self.add_parameter(
            "average",
            get_cmd=f":SENSe:{self._proper_function}:AVERage?",
            set_cmd=f":SENSe:{self._proper_function}:AVERage {{}}",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0")
        )

        self.add_parameter(
            "average_count",
            get_cmd=f":SENSe:{self._proper_function}:AVERage:COUNt?",
            set_cmd=f":SENSe:{self._proper_function}:AVERage:COUNt {{}}",
            vals=Numbers(1, 100)
        )

        self.add_parameter(
            "average_type",
            get_cmd=f":SENSe:{self._proper_function}:AVERage:TCONtrol?",
            set_cmd=f":SENSe:{self._proper_function}:AVERage:TCONtrol {{}}",
            vals=Enum('REP', 'rep', 'MOV', 'mov')
        )

    def _get_user_delay(self) -> str:
        get_cmd = f":SENSe:{self._proper_function}:DELay:USER" \
                  f"{self.user_number()}?"
        return self.ask(get_cmd)

    def _set_user_delay(self, value: float) -> None:
        set_cmd = f":SENSe:{self._proper_function}:DELay:USER" \
                  f"{self.user_number()} {value}"
        self.write(set_cmd)


class Keithley7510(VisaInstrument):
    """
    The QCoDeS driver for the Keithley 7510 DMM
    """
    def __init__(self, name: str, address: str, terminator='\n', **kwargs):
        """
        Create an instance of the instrument.

        Args:
            name: Name of the instrument instance
            address: Visa-resolvable instrument address.
        """
        super().__init__(name, address, terminator=terminator, **kwargs)

        self.add_parameter(
            "sense_function",
            set_cmd=self._set_sense_function,
            get_cmd=":SENSe:FUNCtion?",
            val_mapping={
                key: value["name"]
                for key, value in Sense7510.function_modes.items()
            }
        )

        for proper_sense_function in Sense7510.function_modes:
            self.add_submodule(
                f"_sense_{proper_sense_function}",
                Sense7510(self, "sense", proper_sense_function)
            )

        self.connect_message()

    def _set_sense_function(self, value: str) -> None:
        """
        Change the sense function. The property 'sense' will return the
        sense module appropriate for this function setting.

        Args:
            value: functions in sense.function_modes
        """
        self.write(f":SENSe:FUNCtion {value}")

    @property
    def sense(self) -> Sense7510:
        """
        We have different sense modules depending on the sense function.

        Return the correct source module based on the sense function
        """
        sense_function = \
            self.sense_function.get_latest() or self.sense_function()
        submodule = self.submodules[f"_sense_{sense_function}"]
        return cast(Sense7510, submodule)

    def clear_status(self) -> None:
        """
        This command clears the event registers of the Questionable Event and
        Operation Event Register set. It does not affect the Questionable Event
        Enable or Operation Event Enable registers.
        """
        self.write('*CLS')

    def reset(self) -> None:
        """
        Returns the instrument to default settings, cancels all pending commands
        """
        self.write('*RST')
