import numpy as np
from typing import cast, Optional, List, Union, Sequence, Any, Tuple, Dict, Type
from types import TracebackType

from qcodes import VisaInstrument, InstrumentChannel
from qcodes.instrument.parameter import invert_val_mapping, Parameter, \
    DelegateParameter, MultiParameter, ParamRawDataType
from qcodes.utils.validators import Enum, Numbers, Ints, Lists, Arrays
from qcodes.utils.helpers import create_on_off_val_mapping


class DataArray7510(MultiParameter):
    """
    Data class when user selected more than one field for data output.
    """
    _data: Tuple[Tuple[Any, ...], ...] = ((), ())

    def __init__(self,
                 names: Sequence[str],
                 shapes: Sequence[Sequence[int]],
                 setpoints: Optional[Sequence[Sequence]],
                 **kwargs: Any):
        super().__init__(name='data_array_7510',
                         names=names,
                         shapes=shapes,
                         setpoints=setpoints,
                         **kwargs)
        for param_name in self.names:
            self.__dict__.update({param_name: []})

    def get_raw(self) -> Optional[tuple]:
        return self._data


class GeneratedSetPoints(Parameter):
    """
    A parameter that generates a setpoint array from start, stop and num points
    parameters.
    """
    def __init__(self,
                 start: Parameter,
                 stop: Parameter,
                 n_points: Parameter,
                 *args: Any,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._start = start
        self._stop = stop
        self._n_points = n_points

    def get_raw(self) -> np.ndarray:
        return np.linspace(self._start(), self._stop(), self._n_points())


class Buffer7510(InstrumentChannel):
    """
    Treat the reading buffer as a submodule, similar to Sense.
    """
    default_buffer = {"defbuffer1", "defbuffer2"}

    buffer_elements = {
        "date": "DATE",
        "measurement_formatted": "FORMatted",
        "fractional_seconds": "FRACtional",
        "measurement": "READing",
        "relative_time": "RELative",
        "seconds": "SEConds",
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

        if self.short_name not in self.default_buffer:
            # when making a new buffer, the "size" parameter is required.
            if size is None:
                raise TypeError(
                    "buffer() missing 1 required positional argument: 'size'"
                )
            self.write(
                f":TRACe:MAKE '{self.short_name}', {self._size}, {self.style}"
            )
        else:
            # when referring to default buffer, "size" parameter is not needed.
            if size is not None:
                self.log.warning(
                    f"Please use method 'size()' to resize default buffer "
                    f"{self.short_name} size to {self._size}."
                )

        self.add_parameter(
            "size",
            get_cmd=f":TRACe:POINts? '{self.short_name}'",
            set_cmd=f":TRACe:POINts {{}}, '{self.short_name}'",
            get_parser=int,
            docstring="The number of readings a buffer can store."
        )

        self.add_parameter(
            "number_of_readings",
            get_cmd=f":TRACe:ACTual? '{self.short_name}'",
            get_parser=int,
            docstring="Get the number of readings in the reading buffer."
        )

        self.add_parameter(
            "last_index",
            get_cmd=f":TRACe:ACTual:END? '{self.short_name}'",
            get_parser=int,
            docstring="Get the last index of readings in the reading buffer."
        )

        self.add_parameter(
            "first_index",
            get_cmd=f":TRACe:ACTual:STARt? '{self.short_name}'",
            get_parser=int,
            docstring="Get the starting index of readings in the reading "
                      "buffer."
        )

        self.add_parameter(
            "data_start",
            initial_value=1,
            get_cmd=None,
            set_cmd=None,
            docstring="First index of the data to be returned."
        )

        self.add_parameter(
            "data_end",
            initial_value=1,
            get_cmd=None,
            set_cmd=None,
            docstring="Last index of the data to be returned."
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
            "setpoints_start",
            label="start value for the setpoints",
            source=None,
            parameter_class=DelegateParameter
        )

        self.add_parameter(
            "setpoints_stop",
            label="stop value for the setpoints",
            source=None,
            parameter_class=DelegateParameter
        )

        self.add_parameter(
            "n_pts",
            label="total n for the setpoints",
            get_cmd=self._get_n_pts
        )

        self.add_parameter(
            "setpoints",
            parameter_class=GeneratedSetPoints,
            start=self.setpoints_start,
            stop=self.setpoints_stop,
            n_points=self.n_pts,
            vals=Arrays(shape=(self.n_pts.get_latest,))
        )

        self.add_parameter(
            "t_start",
            label="start time",
            unit="s",
            initial_value=0,
            get_cmd=None,
            set_cmd=None,
            set_parser=float
        )

        self.add_parameter(
            "t_stop",
            label="stop time",
            unit="s",
            initial_value=1,
            get_cmd=None,
            set_cmd=None,
            set_parser=float
        )

        self.add_parameter(
            "fill_mode",
            get_cmd=f":TRACe:FILL:MODE? '{self.short_name}'",
            set_cmd=f":TRACe:FILL:MODE {{}}, '{self.short_name}'",
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

    def _get_n_pts(self) -> int:
        return self.data_end() - self.data_start() + 1

    def set_setpoints(self,
                      start: Parameter,
                      stop: Parameter,
                      label: Optional[str] = None) -> None:
        self.setpoints_start.source = start
        self.setpoints_stop.source = stop
        self.setpoints.unit = start.unit
        if label is not None:
            self.setpoints.label = label

    def __enter__(self) -> "Buffer7510":
        return self

    def __exit__(self, exception_type: Optional[Type[BaseException]],
                 value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        self.delete()

    @property
    def available_elements(self) -> set:
        return set(self.buffer_elements.keys())

    @property
    def n_elements(self) -> int:
        return max(1, len(self.elements()))

    @property
    def data(self) -> DataArray7510:
        return self._get_data()

    def get_last_reading(self) -> str:
        """
        This method requests the latest reading from a reading buffer.

        """
        if not self.elements():
            return self.ask(f":FETCh? '{self.short_name}'")
        fetch_elements = [
            self.buffer_elements[element] for element in self.elements()
        ]
        return self.ask(
            f":FETCh? '{self.short_name}', {','.join(fetch_elements)}"
        )

    def _get_data(self) -> DataArray7510:
        """
        This command returns the data in the buffer, depends on the user
        selected elements.
        """
        try:
            _ = self.setpoints()
        except NotImplementedError:
            # if the "setpionts" has not been implemented, use a time series
            # with parameters "t_start" and "t_stop":
            self.set_setpoints(self.t_start, self.t_stop)

        if self.parent.digi_sense_function() == "None":
            # when current sense is not digitize sense
            sense_function = self.parent.sense_function()
            unit = Sense7510.function_modes[sense_function]["unit"]
        else:
            # when current sense is digitize sense
            sense_function = self.parent.digi_sense_function()
            unit = DigitizeSense7510.function_modes[sense_function]["unit"]

        elements_units = {
            "date": "str",
            "measurement_formatted": "str",
            "fractional_seconds": "s",
            "measurement": unit,
            "relative_time": "s",
            "seconds": "s",
            "measurement_status": "",
            "time": "str",
            "timestamp": "str",
            "measurement_unit": "str"
        }

        if not self.elements():
            raw_data = self.ask(f":TRACe:DATA? "
                                f"{self.data_start()}, "
                                f"{self.data_end()}, "
                                f"'{self.short_name}'")
        else:
            elements = \
                [self.buffer_elements[element] for element in self.elements()]
            raw_data = self.ask(f":TRACe:DATA? {self.data_start()}, "
                                f"{self.data_end()}, "
                                f"'{self.short_name}', "
                                f"{','.join(elements)}")

        all_data = raw_data.split(",")

        if len(self.elements()) == 0:
            elements = ['measurement']
        else:
            elements = self.elements()
        n_elements = len(elements)

        units = tuple(elements_units[element] for element in elements)
        processed_data = dict.fromkeys(elements)
        for i, (element, unit) in enumerate(zip(elements, units)):
            if unit == 'str':
                processed_data[element] = np.array(all_data[i::n_elements])
            else:
                processed_data[element] = np.array([
                    float(v) for v in all_data[i::n_elements]
                ])

        data = DataArray7510(
            names=tuple(elements),
            shapes=((self.n_pts(),),) * n_elements,
            units=units,
            setpoints=((self.setpoints(),),) * n_elements,
            setpoint_units=((self.setpoints.unit,),) * n_elements,
            setpoint_names=((self.setpoints.label,),) * n_elements
        )
        data._data = tuple(
            tuple(processed_data[element]) for element in elements
        )
        for i in range(len(data.names)):
            setattr(data, data.names[i], tuple(processed_data[data.names[i]]))
        return data

    def clear_buffer(self) -> None:
        """
        Clear the data in the buffer
        """
        self.write(f":TRACe:CLEar '{self.short_name}'")

    def trigger_start(self) -> None:
        """
        This method makes readings using the active measure function and
        stores them in a reading buffer.
        """
        self.write(f":TRACe:TRIGger '{self.short_name}'")

    def delete(self) -> None:
        if self.short_name not in self.default_buffer:
            self.parent.submodules.pop(f"_buffer_{self.short_name}")
            self.parent.buffer_name("defbuffer1")
            self.write(f":TRACe:DELete '{self.short_name}'")


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
            "unit": 'Ohm',
            "range_vals": Numbers(10, 1e9),
        },
        "Fresistance": {
            "name": '"FRES"',
            "unit": 'Ohm',
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
    function_modes: Dict[str, Dict[str, Any]] = {
        "None": {
            "name": '"NONE"',
            "unit": '',
            "range_vals": None
        },
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
        buffer_name = self.parent.buffer_name()
        return float(self.ask(f":MEASure:DIGitize? '{buffer_name}'"))


class Keithley7510(VisaInstrument):
    """
    The QCoDeS driver for the Keithley 7510 DMM
    """
    def __init__(self, name: str, address: str,
                 terminator: str = '\n', **kwargs: Any):
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
        if self.digi_sense_function() == 'None':
            raise AttributeError("Please use 'digi_sense_function()' to select"
                                 " a digitize function first")
        sense_function = \
            self.digi_sense_function.get_latest() or self.digi_sense_function()
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
