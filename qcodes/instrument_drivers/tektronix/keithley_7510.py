import numpy as np
from typing import cast, Optional, List, Union

from qcodes import VisaInstrument, InstrumentChannel
from qcodes.instrument.parameter import invert_val_mapping
from qcodes.utils.validators import Enum, Numbers, Ints, Lists
from qcodes.utils.helpers import create_on_off_val_mapping


class UnimplementedError(Exception):
    pass


class Buffer7510(InstrumentChannel):
    """
    Treat the reading buffer as a submodule, similar to Sense
    """
    default_buffer = {"defbuffer1", "defbuffer2"}

    buffer_elements = {
        "date": "DATE",
        "measurement_formatted": "FORMatted",
        "fractional_seconds": "FRACtional",
        "measurement": "READing",
        "relative_time": "RELative",
        "seconds": "SEConds",
        "source_value": "SOURce",
        "source_value_formatted": "SOURFORMatted",
        "source_value_status": "SOURSTATus",
        "source_value_unit": "SOURUNIT",
        "measurement_status": "STATus",
        "time": "TIME",
        "timestamp": "TSTamp",
        "measurement_unit": "UNIT"
    }

    inverted_buffer_elements = invert_val_mapping(buffer_elements)

    def __init__(
            self,
            parent: 'Keithley7510',
            name: str,
            size: Optional[int] = None,
            style: str = ''
    ) -> None:
        super().__init__(parent, name)
        self._size = size
        self.style = style
        self.data_start = 1  # first index of the data to be returned
        self.data_end = 1    # last index of the data to be returned

        if self.name not in self.default_buffer:
            # when making a new buffer, the "size" parameter is required.
            if size is None:
                raise TypeError(
                    "buffer() missing 1 required positional argument: 'size'"
                )
            self.write(
                f":TRACe:MAKE '{self.name}', {self._size}, {self.style}"
            )
        else:
            # when referring to default buffer, "size" parameter is not needed.
            if size is not None:
                self.log.warning(
                    f"Please use method 'size()' to resize default buffer "
                    f"{self.name} size to {self._size}."
                )

        self.add_parameter(
            "size",
            get_cmd=f":TRACe:POINts? '{self.name}'",
            set_cmd=f":TRACe:POINts {{}}, '{self.name}'",
            get_parser=int,
            docstring="The number of readings a buffer can store."
        )

        self.add_parameter(
            "number_of_readings",
            get_cmd=f":TRACe:ACTual? '{self.name}'",
            get_parser=int,
            docstring="Get the number of readings in the reading buffer."
        )

        self.add_parameter(
            "last_index",
            get_cmd=f":TRACe:ACTual:END? '{self.name}'",
            get_parser=int,
            docstring="Get the last index of readings in the reading buffer."
        )

        self.add_parameter(
            "first_index",
            get_cmd=f":TRACe:ACTual:STARt? '{self.name}'",
            get_parser=int,
            docstring="Get the starting index of readings in the reading "
                      "buffer."
        )

        self.add_parameter(
            "elements",
            get_cmd=None,
            get_parser=self._from_scpi_to_name,
            set_cmd=None,
            set_parser=self._from_name_to_scpi,
            vals=Lists(Enum(*list(self.buffer_elements.keys()))),
            docstring="List of buffer elements to read."
        )

        self.add_parameter(
            "data",
            get_cmd=self._get_data,
            docstring="Gets the data with user-defined start, end, and fields."
        )

        self.add_parameter(
            "fill_mode",
            get_cmd=":TRACe:FILL:MODE?",
            set_cmd=":TRACe:FILL:MODE {}",
            vals=Enum('CONT', 'continuous', 'ONCE', 'once'),
            docstring="if a reading buffer is filled continuously or is filled"
                      " once and stops"
        )

    def _from_name_to_scpi(self, element_names: List[str]) -> List[str]:
        return [self.buffer_elements[element] for element in element_names]

    def _from_scpi_to_name(self, element_scpis: List[str]) -> List[str]:
        if element_scpis is None:
            return []
        return [
            self.inverted_buffer_elements[element] for element in element_scpis
        ]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.delete()

    @property
    def available_elements(self) -> set:
        return set(self.buffer_elements.keys())

    def get_last_reading(self) -> str:
        """
        This method requests the latest reading from a reading buffer.

        """
        if not self.elements():
            return self.ask(f":FETCh? '{self.name}'")
        fetch_elements = [
            self.buffer_elements[element] for element in self.elements()
        ]
        return self.ask(
            f":FETCh? '{self.name}', {','.join(fetch_elements)}"
        )

    def _get_data(self) -> np.ndarray:
        """
        This command returns specified data elements from reading buffer.

        Returns:
            data elements from the reading buffer
        """
        start_idx = self.data_start
        end_idx = self.data_end
        npts = end_idx - start_idx + 1

        if not self.elements():
            raw_data = self.ask(f":TRACe:DATA? {start_idx}, {end_idx}, "
                                f"'{self.name}'")
            return np.array([float(i) for i in raw_data.split(",")])
        elements = \
            [self.buffer_elements[element] for element in self.elements()]
        raw_data_with_extra = self.ask(f":TRACe:DATA? {start_idx}, "
                                       f"{end_idx}, "
                                       f"'{self.name}', "
                                       f"{','.join(elements)}")
        all_data = np.array(raw_data_with_extra.split(","))
        return all_data.reshape(npts, len(elements)).T

    def clear_buffer(self) -> None:
        """
        Clear the data in the buffer
        """
        self.write(f":TRACe:CLEar '{self.name}'")

    def trigger_start(self) -> None:
        """
        This method makes readings using the active measure function and
        stores them in a reading buffer.
        """
        self.write(f":TRACe:TRIGger '{self.name}'")

    def delete(self) -> None:
        if self.name not in self.default_buffer:
            self.parent.submodules.pop(f"_buffer_{self.name}")
            self.parent.buffer_name("defbuffer1")
            self.write(f":TRACe:DELete '{self.name}'")


class Sense7510(InstrumentChannel):
    """
    The sense module of the Keithley 7510 DMM, based on the sense module of
    Keithley 2450 SMU.

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
            which returns the proper submodule for any given function mode.
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
        }
    }

    def __init__(
            self,
            parent: VisaInstrument,
            name: str,
            proper_function: str
    ) -> None:

        super().__init__(parent, name)

        self._proper_function = proper_function
        range_vals = self.function_modes[self._proper_function]["range_vals"]
        unit = self.function_modes[self._proper_function]["unit"]

        self.function = self.parent.sense_function

        self.add_parameter(
            self._proper_function,
            get_cmd=self._measure,
            get_parser=float,
            unit=unit,
            docstring="Make measurements, place them in a reading buffer, and "
                      "return the last reading."
        )

        self.add_parameter(
            "auto_range",
            get_cmd=f":SENSe:{self._proper_function}:RANGe:AUTO?",
            set_cmd=f":SENSe:{self._proper_function}:RANGe:AUTO {{}}",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0"),
            docstring="Determine if the measurement range is set manually or "
                      "automatically for the selected measure function."
        )

        self.add_parameter(
            "range",
            get_cmd=f":SENSe:{self._proper_function}:RANGe?",
            set_cmd=f":SENSe:{self._proper_function}:RANGe {{}}",
            vals=range_vals,
            get_parser=float,
            unit=unit,
            docstring="Determine the positive full-scale measure range."
        )

        self.add_parameter(
            "nplc",
            get_cmd=f":SENSe:{self._proper_function}:NPLCycles?",
            set_cmd=f":SENSe:{self._proper_function}:NPLCycles {{}}",
            vals=Numbers(0.01, 10),
            get_parser=float,
            docstring="Set the time that the input signal is measured for the "
                      "selected function.(NPLC = number of power line cycles)"
        )

        self.add_parameter(
            "auto_delay",
            get_cmd=f":SENSe:{self._proper_function}:DELay:AUTO?",
            set_cmd=f":SENSe:{self._proper_function}:DELay:AUTO {{}}",
            val_mapping=create_on_off_val_mapping(on_val="ON", off_val="OFF"),
            docstring="Enable or disable the automatic delay that occurs "
                      "before each measurement."
        )

        self.add_parameter(
            'user_number',
            get_cmd=None,
            set_cmd=None,
            vals=Ints(1, 5),
            docstring="Set the user number for user-defined delay."
        )

        self.add_parameter(
            "user_delay",
            get_cmd=self._get_user_delay,
            set_cmd=self._set_user_delay,
            vals=Numbers(0, 1e4),
            unit='second',
            docstring="Set a user-defined delay that you can use in the "
                      "trigger model."
        )

        self.add_parameter(
            "auto_zero",
            get_cmd=f":SENSe:{self._proper_function}:AZERo?",
            set_cmd=f":SENSe:{self._proper_function}:AZERo {{}}",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0"),
            docstring="Enable or disable automatic updates to the internal "
                      "reference measurements (autozero) of the instrument."
        )

        self.add_parameter(
            "auto_zero_once",
            set_cmd=f":SENSe:AZERo:ONCE",
            docstring="Cause the instrument to refresh the reference and "
                      "zero measurements once"
        )

        self.add_parameter(
            "average",
            get_cmd=f":SENSe:{self._proper_function}:AVERage?",
            set_cmd=f":SENSe:{self._proper_function}:AVERage {{}}",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0"),
            docstring="Enable or disable the averaging filter for measurements "
                      "of the selected function."
        )

        self.add_parameter(
            "average_count",
            get_cmd=f":SENSe:{self._proper_function}:AVERage:COUNt?",
            set_cmd=f":SENSe:{self._proper_function}:AVERage:COUNt {{}}",
            vals=Numbers(1, 100),
            docstring="Set the number of measurements that are averaged when "
                      "filtering is enabled."
        )

        self.add_parameter(
            "average_type",
            get_cmd=f":SENSe:{self._proper_function}:AVERage:TCONtrol?",
            set_cmd=f":SENSe:{self._proper_function}:AVERage:TCONtrol {{}}",
            vals=Enum('REP', 'rep', 'MOV', 'mov'),
            docstring="Set the type of averaging filter that is used for the "
                      "selected measure function when the measurement filter "
                      "is enabled."
        )

    def _get_user_delay(self) -> str:
        get_cmd = f":SENSe:{self._proper_function}:DELay:USER" \
                  f"{self.user_number()}?"
        return self.ask(get_cmd)

    def _set_user_delay(self, value: float) -> None:
        set_cmd = f":SENSe:{self._proper_function}:DELay:USER" \
                  f"{self.user_number()} {value}"
        self.write(set_cmd)

    def _measure(self) -> Union[float, str]:
        if not self.parent.output_enabled():
            raise RuntimeError("Output needs to be on for a measurement")
        buffer_name = self.parent.buffer_name()
        return float(self.ask(f":MEASure? '{buffer_name}'"))

    def clear_trace(self, buffer_name: str = "defbuffer1") -> None:
        """
        Clear the data buffer
        """
        self.write(f":TRACe:CLEar '{buffer_name}'")


class DigitizeSense7510(InstrumentChannel):
    """
    The Digitize sense module of the Keithley 7510 DMM.
    """
    function_modes = {
        "voltage": {
            "name": '"VOLT"',
            "unit": 'V',
            "range_vals": Numbers(0.1, 1000),
        },
        "current": {
            "name": '"CURR"',
            "unit": 'V',
            "range_vals": Numbers(10e-6, 10),
        }
    }

    def __init__(
            self,
            parent: VisaInstrument,
            name: str,
            proper_function: str
    ) -> None:

        super().__init__(parent, name)

        self._proper_function = proper_function
        range_vals = self.function_modes[self._proper_function]["range_vals"]
        unit = self.function_modes[self._proper_function]["unit"]

        self.function = self.parent.digi_sense_function

        self.add_parameter(
            self._proper_function,
            get_cmd=self._measure,
            unit=unit,
            docstring="Make measurements, place them in a reading buffer, and "
                      "return the last reading."
        )

        self.add_parameter(
            "range",
            get_cmd=f":SENSe:DIGitize:{self._proper_function}:RANGe?",
            set_cmd=f":SENSe:DIGitize:{self._proper_function}:RANGe {{}}",
            vals=range_vals,
            get_parser=float,
            unit=unit,
            docstring="Determine the positive full-scale measure range."
        )

        self.add_parameter(
            "input_impedance",
            get_cmd=":SENSe:DIGitize:VOLTage:INPutimpedance?",
            set_cmd=":SENSe:DIGitize:VOLTage:INPutimpedance {}",
            vals=Enum("AUTO", "MOHM10"),
            docstring="Determine when the 10 MΩ input divider is enabled. "
                      "'MOHM10' means 10 MΩ for all ranges."
        )

        self.add_parameter(
            'acq_rate',
            get_cmd=f":SENSe:DIGitize:{self._proper_function}:SRATE?",
            set_cmd=f":SENSe:DIGitize:{self._proper_function}:SRATE {{}}",
            vals=Ints(1000, 1000000),
            docstring="Define the precise acquisition rate at which the "
                      "digitizing measurements are made."
        )

        self.add_parameter(
            "aperture",
            get_cmd=f":SENSe:DIGitize:{self._proper_function}:APERture?",
            set_cmd=f":SENSe:DIGitize:{self._proper_function}:APERture {{}}",
            unit="us",
            docstring="Determine the aperture setting."
        )

        self.add_parameter(
            "count",
            get_cmd="SENSe:DIGitize:COUNt?",
            set_cmd="SENSe:DIGitize:COUNt {}",
            vals=Ints(1, 55000000),
            docstring="Set the number of measurements to digitize when a "
                      "measurement is requested"
        )

    def _measure(self) -> Union[float, str]:
        if not self.parent.output_enabled():
            raise RuntimeError("Output needs to be on for a measurement")
        buffer_name = self.parent.buffer_name()
        return float(self.ask(f":MEASure:DIGitize? '{buffer_name}'"))


class Keithley7510(VisaInstrument):
    """
    The QCoDeS driver for the Keithley 7510 DMM
    """
    def __init__(self, name: str, address: str, terminator='\n', **kwargs):
        """
        Create an instance of the instrument.

        Args:
            name: Name of the instrument instance
            address: Visa-resolvable instrument address
        """
        super().__init__(name, address, terminator=terminator, **kwargs)

        self.add_parameter(
            "sense_function",
            set_cmd=":SENSe:FUNCtion {}",
            get_cmd=":SENSe:FUNCtion?",
            val_mapping={
                key: value["name"]
                for key, value in Sense7510.function_modes.items()
            },
            docstring="Add sense functions listed in the function modes."
        )

        self.add_parameter(
            "digi_sense_function",
            set_cmd=":DIGitize:FUNCtion {}",
            get_cmd=":DIGitize:FUNCtion?",
            val_mapping={
                key: value["name"]
                for key, value in DigitizeSense7510.function_modes.items()
            },
            docstring="Make readings using the active digitize function."
        )

        self.add_parameter(
            "buffer_name",
            get_cmd=None,
            set_cmd=None,
            docstring="Name of the reading buffer in use."
        )

        self.add_parameter(
            "trigger_block_list",
            get_cmd=":TRIGger:BLOCk:LIST?",
            docstring="Return the settings for all trigger model blocks."
        )

        self.add_parameter(
            "trigger_in_ext_clear",
            set_cmd=":TRIGger:EXTernal:IN:CLEar",
            docstring="Clear the trigger event on the external in line."
        )

        self.add_parameter(
            "trigger_in_ext_edge",
            get_cmd=":TRIGger:EXTernal:IN:EDGE?",
            set_cmd=":TRIGger:EXTernal:IN:EDGE {}",
            vals=Enum("FALL", "RIS", "falling", "rising", "EITH", "either"),
            docstring="Type of edge that is detected as an input on the "
                      "external trigger in line"
        )

        self.add_parameter(
            "overrun_status",
            get_cmd=":TRIGger:EXTernal:IN:OVERrun?",
            docstring="Return the event detector overrun status."
        )

        self.add_parameter(
            "digitize_trigger",
            get_cmd=":TRIGger:DIGitize:STIMulus?",
            set_cmd=":TRIGger:DIGitize:STIMulus {}",
            vals=Enum("EXT", "external", "NONE"),
            docstring="Set the instrument to digitize a measurement the next "
                      "time it detects the specified trigger event."
        )

        self.add_parameter(
            "system_errors",
            get_cmd=":SYSTem:ERRor?",
            docstring="Return the oldest unread error message from the event "
                      "log and removes it from the log."
        )

        for proper_sense_function in Sense7510.function_modes:
            self.add_submodule(
                f"_sense_{proper_sense_function}",
                Sense7510(self, "sense", proper_sense_function)
            )

        for proper_sense_function in DigitizeSense7510.function_modes:
            self.add_submodule(
                f"_digi_sense_{proper_sense_function}",
                DigitizeSense7510(self, "digi_sense", proper_sense_function)
            )

        self.buffer_name('defbuffer1')
        self.buffer(name=self.buffer_name())
        self.connect_message()

    @property
    def sense(self) -> Sense7510:
        """
        We have different sense modules depending on the sense function.

        Return the correct source module based on the sense function.
        """
        sense_function = \
            self.sense_function.get_latest() or self.sense_function()
        submodule = self.submodules[f"_sense_{sense_function}"]
        return cast(Sense7510, submodule)

    @property
    def digi_sense(self) -> DigitizeSense7510:
        """
        We have different sense modules depending on the sense function.

        Return the correct source module based on the sense function.
        """
        sense_function = \
            self.sense_function.get_latest() or self.sense_function()
        submodule = self.submodules[f"_digi_sense_{sense_function}"]
        return cast(DigitizeSense7510, submodule)

    def buffer(
            self,
            name: str,
            size: Optional[int] = None,
            style: str = ''
    ) -> Buffer7510:
        self.buffer_name(name)
        if f"_buffer_{name}" in self.submodules:
            return cast(Buffer7510, self.submodules[f"_buffer_{name}"])
        new_buffer = Buffer7510(parent=self, name=name, size=size, style=style)
        self.add_submodule(f"_buffer_{name}", new_buffer)
        return new_buffer

    def initiate(self) -> None:
        """
        This command starts the trigger model.
        """
        self.write(":INITiate")

    def wait(self) -> None:
        """
        This command postpones the execution of subsequent commands until all
        previous overlapped commands are finished.
        """
        self.write("*WAI")

    def clear_status(self) -> None:
        """
        This command clears the event registers of the Questionable Event and
        Operation Event Register set. It does not affect the Questionable Event
        Enable or Operation Event Enable registers.
        """
        self.write('*CLS')

    def reset(self) -> None:
        """
        Returns the instrument to default settings, cancels all pending
        commands.
        """
        self.write('*RST')
