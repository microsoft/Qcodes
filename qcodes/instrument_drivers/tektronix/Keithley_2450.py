from types import TracebackType
from typing import Any, Dict, List, Optional, Type, Union, cast

import numpy as np
from qcodes import InstrumentChannel, ParameterWithSetpoints, VisaInstrument
from qcodes.instrument.parameter import invert_val_mapping
from qcodes.utils.helpers import create_on_off_val_mapping
from qcodes.utils.validators import Arrays, Enum, Ints, Lists, Numbers


class ParameterWithSetpointsCustomized(ParameterWithSetpoints):
    """
    While the parent class ParameterWithSetpoints only support numerical data
    (in the format of "Arrays"), the newly added "_user_selected_data" will
    include extra fields which may contain string type, in addition to the
    numerical values, which can be obtained by the get_cmd of the parent class.

    This customized class is used for the "sweep" parameter.
    """
    _user_selected_data: Optional[list] = None

    def get_selected(self) -> Optional[list]:
        return self._user_selected_data


class Buffer2450(InstrumentChannel):
    """
    Treat the reading buffer as a submodule, similar to Sense and Source
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
            parent: 'Keithley2450',
            name: str,
            size: Optional[int] = None,
            style: str = ''
    ) -> None:
        super().__init__(parent, name)
        self.buffer_name = name
        self._size = size
        self.style = style

        if self.buffer_name not in self.default_buffer:
            # when making a new buffer, the "size" parameter is required.
            if size is None:
                raise TypeError(
                    "buffer() missing 1 required positional argument: 'size'"
                )
            self.write(
                f":TRACe:MAKE '{self.buffer_name}', {self._size}, {self.style}"
            )
        else:
            # when referring to default buffer, "size" parameter is not needed.
            if size is not None:
                self.log.warning(
                    f"Please use method 'size()' to resize default buffer "
                    f"{self.buffer_name} size to {self._size}."
                )

        self.add_parameter(
            "size",
            get_cmd=f":TRACe:POINts? '{self.buffer_name}'",
            set_cmd=f":TRACe:POINts {{}}, '{self.buffer_name}'",
            get_parser=int,
            docstring="The number of readings a buffer can store."
        )

        self.add_parameter(
            "number_of_readings",
            get_cmd=f":TRACe:ACTual? '{self.buffer_name}'",
            get_parser=int,
            docstring="To get the number of readings in the reading buffer."
        )

        self.add_parameter(
            "elements",
            get_cmd=None,
            get_parser=self.from_scpi_to_name,
            set_cmd=None,
            set_parser=self.from_name_to_scpi,
            vals=Lists(Enum(*list(self.buffer_elements.keys()))),
            docstring="List of buffer elements to read."
        )

    def from_name_to_scpi(self, element_names: List[str]) -> List[str]:
        return [self.buffer_elements[element] for element in element_names]

    def from_scpi_to_name(self, element_scpis: List[str]) -> List[str]:
        if element_scpis is None:
            return []
        return [
            self.inverted_buffer_elements[element] for element in element_scpis
        ]

    def __enter__(self) -> "Buffer2450":
        return self

    def __exit__(self, exception_type: Optional[Type[BaseException]],
                 value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        self.delete()

    @property
    def available_elements(self) -> set:
        return set(self.buffer_elements.keys())

    def get_last_reading(self) -> str:
        """
        This method requests the latest reading from a reading buffer.

        """
        if not self.elements():
            return self.ask(f":FETCh? '{self.buffer_name}'")
        fetch_elements = [
            self.buffer_elements[element] for element in self.elements()
        ]
        return self.ask(
            f":FETCh? '{self.buffer_name}', {','.join(fetch_elements)}"
        )

    def get_data(
            self,
            start_idx: int,
            end_idx: int,
            readings_only: bool = False
    ) -> list:
        """
        This command returns specified data elements from reading buffer.

        Args:
            start_idx: beginning index of the buffer to return
            end_idx: ending index of the buffer to return
            readings_only: a flag to temporarily disable the elements and
                output only the numerical readings

        Returns:
            data elements from the reading buffer

        """
        if (not self.elements()) or readings_only:
            raw_data = self.ask(f":TRACe:DATA? {start_idx}, {end_idx}, "
                                f"'{self.buffer_name}'")
            return [float(i) for i in raw_data.split(",")]
        elements = \
            [self.buffer_elements[element] for element in self.elements()]
        raw_data_with_extra = self.ask(f":TRACe:DATA? {start_idx}, "
                                       f"{end_idx}, "
                                       f"'{self.buffer_name}', "
                                       f"{','.join(elements)}")
        return raw_data_with_extra.split(",")

    def clear_buffer(self) -> None:
        """
        Clear the data in the buffer
        """
        self.write(f":TRACe:CLEar '{self.buffer_name}'")

    def trigger_start(self) -> None:
        """
        This method makes readings using the active measure function and
        stores them in a reading buffer.
        """
        self.write(f":TRACe:TRIGger '{self.buffer_name}'")

    def delete(self) -> None:
        if self.buffer_name not in self.default_buffer:
            self.parent.submodules.pop(f"_buffer_{self.buffer_name}")
            self.parent.buffer_name("defbuffer1")
            self.write(f":TRACe:DELete '{self.buffer_name}'")


class Sense2450(InstrumentChannel):
    """
    The sense module of the Keithley 2450 SMU.

    Args:
        parent
        name
        proper_function: This can be one of either "current", "voltage"
            or "resistance". All parameters and methods in this submodule
            should only be accessible to the user if
            self.parent.sense_function.get() == self._proper_function. We
            ensure this through the 'sense' property on the main driver class
            which returns the proper submodule for any given function mode
    """

    function_modes = {
        "current": {
            "name": '"CURR:DC"',
            "unit": "A",
            "range_vals": Numbers(10E-9, 1)
        },
        "resistance": {
            "name": '"RES"',
            "unit": "Ohm",
            "range_vals": Numbers(20, 200E6)
        },
        "voltage": {
            "name": '"VOLT:DC"',
            "unit": "V",
            "range_vals": Numbers(0.02, 200)
        }
    }

    def __init__(self, parent: 'Keithley2450', name: str, proper_function: str) -> None:
        super().__init__(parent, name)
        self._proper_function = proper_function
        range_vals = self.function_modes[self._proper_function]["range_vals"]
        unit = self.function_modes[self._proper_function]["unit"]

        self.function = self.parent.sense_function

        self.add_parameter(
            "four_wire_measurement",
            set_cmd=f":SENSe:{self._proper_function}:RSENse {{}}",
            get_cmd=f":SENSe:{self._proper_function}:RSENse?",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0")
        )

        self.add_parameter(
            "range",
            set_cmd=f":SENSe:{self._proper_function}:RANGe {{}}",
            get_cmd=f":SENSe:{self._proper_function}:RANGe?",
            vals=range_vals,
            get_parser=float,
            unit=unit
        )

        self.add_parameter(
            "auto_range",
            set_cmd=f":SENSe:{self._proper_function}:RANGe:AUTO {{}}",
            get_cmd=f":SENSe:{self._proper_function}:RANGe:AUTO?",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0")
        )

        self.add_parameter(
            self._proper_function,
            get_cmd=self._measure,
            get_parser=float,
            unit=unit,
            snapshot_value=False
        )

        self.add_parameter(
            "sweep",
            label=self._proper_function,
            get_cmd=self._measure_sweep,
            unit=unit,
            vals=Arrays(shape=(self.parent.npts,)),
            parameter_class=ParameterWithSetpointsCustomized
        )

        self.add_parameter(
            "nplc",
            get_cmd=f":SENSe:{self._proper_function}:NPLCycles?",
            set_cmd=f":SENSe:{self._proper_function}:NPLCycles {{}}",
            vals=Numbers(0.001, 10)
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
            get_parser=float,
            vals=Numbers(0, 1e4)
        )

        self.add_parameter(
            'auto_zero_enabled',
            get_cmd=f":SENSe:{self._proper_function}:AZERo?",
            set_cmd=f":SENSe:{self._proper_function}:AZERo {{}}",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0"),
            docstring="This command enables or disables automatic updates to"
                      "the internal reference measurements (autozero) of the"
                      "instrument."
        )

        self.add_parameter(
            'count',
            get_cmd=":SENSe:COUNt?",
            set_cmd=":SENSe:COUNt {}",
            docstring="The number of measurements to make when a measurement "
                      "is requested."
        )

    def _measure(self) -> Union[float, str]:
        if not self.parent.output_enabled():
            raise RuntimeError("Output needs to be on for a measurement")
        buffer_name = self.parent.buffer_name()
        return float(self.ask(f":MEASure? '{buffer_name}'"))

    def _measure_sweep(self) -> np.ndarray:

        source = cast(Source2450, self.parent.source)
        source.sweep_start()
        buffer_name = self.parent.buffer_name()
        buffer = cast(
            Buffer2450, self.parent.submodules[f"_buffer_{buffer_name}"]
        )
        end_idx = self.parent.npts()
        raw_data = buffer.get_data(1, end_idx, readings_only=True)
        raw_data_with_extra = buffer.get_data(1, end_idx)
        self.parent.sense.sweep._user_selected_data = raw_data_with_extra
        # Clear the trace so we can be assured that a subsequent measurement
        # will not be contaminated with data from this run.
        buffer.clear_buffer()
        return np.array([float(i) for i in raw_data])

    def auto_zero_once(self) -> None:
        """
        This command causes the instrument to refresh the reference and zero
        measurements once.
        """
        self.write(":SENSe:AZERo:ONCE")

    def clear_trace(self, buffer_name: str = "defbuffer1") -> None:
        """
        Clear the data buffer
        """
        self.write(f":TRACe:CLEar '{buffer_name}'")

    def _get_user_delay(self) -> str:
        get_cmd = f":SENSe:{self._proper_function}:DELay:USER" \
                  f"{self.user_number()}?"
        return self.ask(get_cmd)

    def _set_user_delay(self, value: float) -> None:
        set_cmd = f":SENSe:{self._proper_function}:DELay:USER" \
                  f"{self.user_number()} {value}"
        self.write(set_cmd)


class Source2450(InstrumentChannel):
    """
    The source module of the Keithley 2450 SMU.

    Args:
        parent
        name
        proper_function: This can be one of either "current" or "voltage"
            All parameters and methods in this submodule should only be
            accessible to the user if
            self.parent.source_function.get() == self._proper_function. We
            ensure this through the 'source' property on the main driver class
            which returns the proper submodule for any given function mode
    """
    function_modes = {
        "current": {
            "name": "CURR",
            "unit": "A",
            "range_vals": Numbers(-1, 1)
        },
        "voltage": {
            "name": "VOLT",
            "unit": "V",
            "range_vals": Numbers(-200, 200)
        }
    }

    def __init__(self, parent: 'Keithley2450', name: str, proper_function: str) -> None:
        super().__init__(parent, name)
        self._proper_function = proper_function
        range_vals = self.function_modes[self._proper_function]["range_vals"]
        unit = self.function_modes[self._proper_function]["unit"]

        self.function = self.parent.source_function
        self._sweep_arguments: Dict[str, Union[float, int, str]] = {}

        self.add_parameter(
            "range",
            set_cmd=f":SOUR:{self._proper_function}:RANGe {{}}",
            get_cmd=f":SOUR:{self._proper_function}:RANGe?",
            vals=range_vals,
            get_parser=float,
            unit=unit
        )

        self.add_parameter(
            "auto_range",
            set_cmd=f":SOURce:{self._proper_function}:RANGe:AUTO {{}}",
            get_cmd=f":SOURce:{self._proper_function}:RANGe:AUTO?",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0")
        )

        limit_cmd = {"current": "VLIM", "voltage": "ILIM"}[self._proper_function]
        self.add_parameter(
            "limit",
            set_cmd=f"SOUR:{self._proper_function}:{limit_cmd} {{}}",
            get_cmd=f"SOUR:{self._proper_function}:{limit_cmd}?",
            get_parser=float,
            unit=unit
        )

        self.add_parameter(
            "limit_tripped",
            get_cmd=f":SOUR:{self._proper_function}:{limit_cmd}:TRIPped?",
            val_mapping={True: 1, False: 0}
        )

        self.add_parameter(
            self._proper_function,
            set_cmd=f"SOUR:{self._proper_function} {{}}",
            get_cmd=f"SOUR:{self._proper_function}?",
            get_parser=float,
            unit=unit,
            snapshot_value=False
        )

        self.add_parameter(
            "sweep_axis",
            label=self._proper_function,
            get_cmd=self.get_sweep_axis,
            vals=Arrays(shape=(self.parent.npts,)),
            unit=unit
        )

        self.add_parameter(
            "delay",
            get_cmd=f":SOURce:{self._proper_function}:DELay?",
            set_cmd=f":SOURce:{self._proper_function}:DELay {{}}",
            vals=Numbers(0, 1e4)
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
            vals=Numbers(0, 1e4)
        )

        self.add_parameter(
            "auto_delay",
            get_cmd=f":SOURce:{self._proper_function}:DELay:AUTO?",
            set_cmd=f":SOURce:{self._proper_function}:DELay:AUTO {{}}",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0")
        )

        self.add_parameter(
            "read_back_enabled",
            get_cmd=f":SOURce:{self._proper_function}:READ:BACK?",
            set_cmd=f":SOURce:{self._proper_function}:READ:BACK {{}}",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0"),
            docstring="This command determines if the instrument records the "
                      "measured source value or the configured source value "
                      "when making a measurement."
        )

    def get_sweep_axis(self) -> np.ndarray:
        if self._sweep_arguments == {}:
            raise ValueError(
                "Please setup the sweep before getting values of this parameter"
            )

        return np.linspace(
            start=self._sweep_arguments["start"],
            stop=self._sweep_arguments["stop"],
            num=self._sweep_arguments["step_count"]
        )

    def sweep_setup(
            self,
            start: float,
            stop: float,
            step_count: int,
            delay: float = 0,
            sweep_count: int = 1,
            range_mode: str = "AUTO",
            fail_abort: str = "ON",
            dual: str = "OFF",
            buffer_name: str = "defbuffer1"
    ) -> None:

        self._sweep_arguments = dict(
            start=start,
            stop=stop,
            step_count=step_count,
            delay=delay,
            sweep_count=sweep_count,
            range_mode=range_mode,
            fail_abort=fail_abort,
            dual=dual,
            buffer_name=buffer_name
        )

    def sweep_start(self) -> None:
        """
        Start a sweep and return when the sweep has finished.
        Note: This call is blocking
        """
        cmd_args = dict(self._sweep_arguments)
        cmd_args["function"] = self._proper_function

        cmd = ":SOURce:SWEep:{function}:LINear {start},{stop}," \
              "{step_count},{delay},{sweep_count},{range_mode}," \
              "{fail_abort},{dual},'{buffer_name}'".format(**cmd_args)

        self.write(cmd)
        self.write(":INITiate")
        self.write("*WAI")

    def sweep_reset(self) -> None:
        self._sweep_arguments = {}

    def _get_user_delay(self) -> float:
        get_cmd = f":SOURce:{self._proper_function}:DELay:USER" \
                  f"{self.user_number()}?"
        return float(self.ask(get_cmd))

    def _set_user_delay(self, value: float) -> None:
        set_cmd = f":SOURce:{self._proper_function}:DELay:USER" \
                  f"{self.user_number()} {value}"
        self.write(set_cmd)


class Keithley2450(VisaInstrument):
    """
    The QCoDeS driver for the Keithley 2450 SMU
    """

    def __init__(self, name: str, address: str, **kwargs: Any) -> None:

        super().__init__(name, address, terminator='\n', **kwargs)

        if not self._has_correct_language_mode():
            self.log.warning(
                f"The instrument is in an unsupported language mode. "
                f"Please run `instrument.set_correct_language()` and try to "
                f"initialize the driver again after an instrument power cycle. "
                f"No parameters/sub modules will be available on this driver "
                f"instance"
            )
            return

        self.add_parameter(
            "source_function",
            set_cmd=self._set_source_function,
            get_cmd=":SOUR:FUNC?",
            val_mapping={
                key: value["name"]
                for key, value in Source2450.function_modes.items()
            }
        )

        self.add_parameter(
            "sense_function",
            set_cmd=self._set_sense_function,
            get_cmd=":SENS:FUNC?",
            val_mapping={
                key: value["name"]
                for key, value in Sense2450.function_modes.items()
            }
        )

        self.add_parameter(
            "terminals",
            set_cmd="ROUTe:TERMinals {}",
            get_cmd="ROUTe:TERMinals?",
            vals=Enum("rear", "front")
        )

        self.add_parameter(
            "output_enabled",
            initial_value="0",
            set_cmd=":OUTP {}",
            get_cmd=":OUTP?",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0")
        )

        self.add_parameter(
            "line_frequency",
            get_cmd=":SYSTem:LFRequency?",
            unit='Hz',
            docstring="returns the power line frequency setting that is used "
                      "for NPLC calculations"
        )

        self.add_parameter(
            "buffer_name",
            get_cmd=None,
            set_cmd=None,
            docstring="name of the reading buffer in using"
        )

        # Make a source module for every source function ('current' and 'voltage')
        for proper_source_function in Source2450.function_modes:
            self.add_submodule(
                f"_source_{proper_source_function}",
                Source2450(self, "source", proper_source_function)
            )

        # Make a sense module for every sense function ('current', voltage' and 'resistance')
        for proper_sense_function in Sense2450.function_modes:
            self.add_submodule(
                f"_sense_{proper_sense_function}",
                Sense2450(self, "sense", proper_sense_function)
            )

        self.buffer_name('defbuffer1')
        self.buffer(name=self.buffer_name())
        self.connect_message()

    def _set_sense_function(self, value: str) -> None:
        """
        Change the sense function. The property 'sense' will return the
        sense module appropriate for this function setting.

        We need to ensure that the setpoints of the sweep parameter in the
        active sense module is correctly set. Normally we would do that
        with 'self.sense.sweep.setpoints = (self.source.sweep_axis,)'

        However, we cannot call the property 'self.sense', because that property
        will call `get_latest` on the parameter for which this function
        (that is '_set_sense_function') is the setter
        """
        self.write(f":SENS:FUNC {value}",)
        sense_function = self.sense_function.inverse_val_mapping[value]
        sense = self.submodules[f"_sense_{sense_function}"]
        if not isinstance(sense, Sense2450):
            raise RuntimeError(f"Expect Sense Module to be of type "
                               f"Sense2450 got {type(sense)}")
        sense.sweep.setpoints = (self.source.sweep_axis,)

    def _set_source_function(self, value: str) -> None:
        """
        Change the source function. The property 'source' will return the
        source module appropriate for this function setting.

        We need to ensure that the setpoints of the sweep parameter in the
        active sense module reflects the change in the source module.
        Normally we would do that with
        'self.sense.sweep.setpoints = (self.source.sweep_axis,)'

        However, we cannot call the property 'self.source', because that property
        will call `get_latest` on the parameter for which this function
        (that is '_set_source_function') is the setter
        """

        if self.sense_function() == "resistance":
            raise RuntimeError(
                "Cannot change the source function while sense function is in 'resistance' mode"
            )

        self.write(f":SOUR:FUNC {value}")
        source_function = self.source_function.inverse_val_mapping[value]
        source = self.submodules[f"_source_{source_function}"]
        self.sense.sweep.setpoints = (source.sweep_axis,)
        if not isinstance(source, Source2450):
            raise RuntimeError(f"Expect Sense Module to be of type "
                               f"Source2450 got {type(source)}")
        # Once the source function has changed,
        # we cannot trust the sweep setup anymore
        source.sweep_reset()

    @property
    def source(self) -> Source2450:
        """
        We have different source modules depending on the source function, which can be
        'current' or 'voltage'

        Return the correct source module based on the source function
        """
        source_function = self.source_function.get_latest() or self.source_function()
        submodule = self.submodules[f"_source_{source_function}"]
        return cast(Source2450, submodule)

    @property
    def sense(self) -> Sense2450:
        """
        We have different sense modules depending on the sense function, which can be
        'current', 'voltage' or 'resistance'

        Return the correct source module based on the sense function
        """
        sense_function = self.sense_function.get_latest() or self.sense_function()
        submodule = self.submodules[f"_sense_{sense_function}"]
        return cast(Sense2450, submodule)

    def buffer(
            self,
            name: str,
            size: Optional[int] = None,
            style: str = ''
    ) -> Buffer2450:
        self.buffer_name(name)
        if f"_buffer_{name}" in self.submodules:
            return cast(Buffer2450, self.submodules[f"_buffer_{name}"])
        new_buffer = Buffer2450(parent=self, name=name, size=size, style=style)
        self.add_submodule(f"_buffer_{name}", new_buffer)
        return new_buffer

    def npts(self) -> int:
        """
        Get the number of points in the sweep axis
        """
        return len(self.source.get_sweep_axis())

    def set_correct_language(self) -> None:
        """
        The correct communication protocol is SCPI, make sure this is set
        """
        self.write("*LANG SCPI")
        self.log.warning("Please power cycle the instrument to make the change take effect")
        # We want the user to be able to instantiate a driver with the same name
        self.close()

    def _has_correct_language_mode(self) -> bool:
        """
        Query if we have the correct language mode
        """
        return self.ask("*LANG?") == "SCPI"

    def abort(self) -> None:
        """
        This command stops all trigger model commands on the instrument.
        """
        self.write(":ABORt")

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

    def clear_event_register(self) -> None:
        """
        This function clears event registers.
        """
        self.write(":STATus:CLEar")

    def clear_event_log(self) -> None:
        """
        This command clears the event log.
        """
        self.write(":SYSTem:CLEar")

    def reset(self) -> None:
        """
        Returns instrument to default settings, cancels all pending commands.
        """
        self.write("*RST")
