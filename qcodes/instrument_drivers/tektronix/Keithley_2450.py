import numpy as np
from typing import Any, Dict, Callable, cast, Sequence

from qcodes import VisaInstrument, InstrumentChannel, ParameterWithSetpoints
from qcodes.instrument.parameter import Parameter, _BaseParameter
from qcodes.utils.validators import Enum, Numbers, Arrays


class ParameterError(Exception):
    """Raise this error if a parameter has an unexpected value"""
    pass


def assert_parameter_values(required_parameter_values: Dict[Parameter, Any]) -> None:
    """
    Assert that the parameters specified in the input dictionary have the required
    values.

    Args:
        required_parameter_values
    """
    for parameter, required_value in required_parameter_values.items():

        actual_value = parameter.get_latest() or parameter.get()
        if required_value != actual_value:

            instrument = cast(VisaInstrument, parameter.instrument)
            instrument_name = instrument.name.replace("_", '.')

            raise ParameterError(
                f"'{instrument_name}.{parameter.name}' has value '{actual_value}'. "
                f"This should be '{required_value}'"
            )


class Sense2450(InstrumentChannel):
    function_modes = {
        "current": {
            "instrument_name": "\"CURR:DC\"",
            "unit": "A",
            "range_vals": Numbers(10E-9, 1)
        },
        "resistance": {
            "instrument_name": "\"RES\"",
            "unit": "Ohm",
            "range_vals": Numbers(20, 200E6)
        },
        "voltage": {
            "instrument_name": "\"VOLT:DC\"",
            "unit": "V",
            "range_vals": Numbers(0.02, 200)
        }
    }

    def __init__(self, parent: 'Keithley2450', name: str) -> None:
        super().__init__(parent, name)

        self.add_parameter(
            "function",
            set_cmd=":SENS:FUNC {}",
            get_cmd=":SENS:FUNC?",
            val_mapping={
                key: self.function_modes[key]["instrument_name"]
                for key in self.function_modes
            }
        )

        for function, args in Sense2450.function_modes.items():
            self.add_parameter(
                f"_four_wire_measurement_{function}",
                set_cmd=f":SENSe:{function}:RSENse {{}}",
                get_cmd=f":SENSe:{function}:RSENse?",
                val_mapping={
                    True: "1",
                    False: "0"
                },
            )

            self.add_parameter(
                f"_range_{function}",
                set_cmd=f":SENSe:{function}:RANGe {{}}",
                get_cmd=f":SENSe:{function}:RANGe?",
                vals=args["range_vals"],
                get_parser=float,
                unit=args["unit"]
            )

            self.add_parameter(
                f"_auto_range_{function}",
                set_cmd=f":SENSe:{function}:RANGe:AUTO {{}}",
                get_cmd=f":SENSe:{function}:RANGe:AUTO?",
                val_mapping={
                    True: "1",
                    False: "0"
                }
            )

            self.add_parameter(
                function,
                get_cmd=self._measure(function),
                get_parser=float,
                unit=args["unit"],
                snapshot_value=False
            )

    def _measure(self, function: str) -> Callable:
        def measurer():
            assert_parameter_values({self.function: function, self.parent.output: True})
            return self.ask(":MEASure?")
        return measurer

    @property
    def four_wire_measurement(self) -> _BaseParameter:
        """
        Return the appropriate parameter based on the current function
        """
        function = self.function.get_latest() or self.function()
        param_name = f"_four_wire_measurement_{function}"
        return self.parameters[param_name]

    @property
    def auto_range(self) -> _BaseParameter:
        """
        Return the appropriate parameter based on the current function
        """
        function = self.function.get_latest() or self.function()
        param_name = f"_auto_range_{function}"
        return self.parameters[param_name]

    @property
    def range(self) -> _BaseParameter:
        """
        Return the appropriate parameter based on the current function
        """
        function = self.function.get_latest() or self.function()
        param_name = f"_range_{function}"
        return self.parameters[param_name]


class Source2450(InstrumentChannel):
    function_modes = {
        "current": {
            "instrument_name": "CURR",
            "unit": "A",
            "range_vals": Numbers(-1, 1)
        },
        "voltage": {
            "instrument_name": "VOLT",
            "unit": "V",
            "range_vals": Numbers(-200, 200)
        }
    }

    def __init__(self, parent: 'Keithley2450', name: str) -> None:
        super().__init__(parent, name)

        self._sweep_arguments: dict = {}

        self.add_parameter(
            "function",
            set_cmd=self._set_source_function,
            get_cmd=":SOUR:FUNC?",
            val_mapping={
                key: self.function_modes[key]["instrument_name"]
                for key in self.function_modes
            }
        )

        for source_function, args in Source2450.function_modes.items():
            self.add_parameter(
                f"_range_{source_function}",
                set_cmd=f":SOUR:{source_function}:RANGe {{}}",
                get_cmd=f":SOUR:{source_function}:RANGe?",
                vals=args["range_vals"],
                get_parser=float,
                unit=args["unit"]
            )

            self.add_parameter(
                f"_auto_range_{source_function}",
                set_cmd=f":SOURce:{source_function}:RANGe:AUTO {{}}",
                get_cmd=f":SOURce:{source_function}:RANGe:AUTO?",
                val_mapping={
                    True: "1",
                    False: "0"
                }
            )

            limit_cmd = {"current": "VLIM", "voltage": "ILIM"}[source_function]
            self.add_parameter(
                f"_limit_{source_function}",
                set_cmd=f"SOUR:{source_function}:{limit_cmd} {{}}",
                get_cmd=f"SOUR:{source_function}:{limit_cmd}?",
                get_parser=float,
                unit=args["unit"]
            )

            self.add_parameter(
                f"_sweep_axis_{source_function}",
                get_cmd=self._get_sweep_axis,
                vals=Arrays(shape=(self.npts,)),
                unit=args["unit"]
            )

            for sense_function, sense_args in Sense2450.function_modes.items():
                self.add_parameter(
                    f"_sweep_{source_function}_{sense_function}",
                    get_cmd=self._measure_sweep(source_function),
                    unit=sense_args["unit"],
                    vals=Arrays(shape=(self.npts,)),
                    setpoints=(self.parameters[f"_sweep_axis_{source_function}"],),
                    parameter_class=ParameterWithSetpoints
                )

        self.add_parameter(
            "current",
            set_cmd=self._setpoint_setter("current"),
            get_cmd=self._setpoint_getter("current"),
            get_parser=float,
            unit="A",
            snapshot_value=False
        )

        self.add_parameter(
            "voltage",
            set_cmd=self._setpoint_setter("voltage"),
            get_cmd=self._setpoint_getter("voltage"),
            get_parser=float,
            unit="V",
            snapshot_value=False
        )

    def _set_source_function(self, value: str) -> None:

        if self.parent.sense.function() == "resistance":
            raise RuntimeError(
                "Cannot change the source function while sense function is in 'resistance' mode"
            )

        self.write(f":SOUR:FUNC {value}")
        # If the source function changes, we cannot trust the
        # sweep setup anymore
        self._sweep_arguments: dict = {}

    def _setpoint_setter(self, function: str) -> Callable:
        def setter(value):
            assert_parameter_values({self.function: function})
            return self.write_raw(f"SOUR:{function} {value}")

        return setter

    def _setpoint_getter(self, function: str) -> Callable:
        def getter():
            assert_parameter_values({self.function: function})
            return self.ask_raw(f"SOUR:{function}?")

        return getter

    def _measure_sweep(self, function: str) -> Callable:
        def measurer() -> np.ndarray:
            assert_parameter_values({self.function: function, self.parent.output: True})

            cmd_args = dict(self._sweep_arguments)
            cmd_args["function"] = self.function()

            cmd = ":SOURce:SWEep:{function}:LINear {start},{stop}," \
                  "{step_count},{delay},{sweep_count},{range_mode}".format(**cmd_args)

            self.write(cmd)
            self.write(":INITiate")
            self.write("*WAI")
            raw_data = self.ask(f":TRACe:DATA? 1, {self.npts()}")
            self.write(":TRACe:CLEar")

            return np.array([float(i) for i in raw_data.split(",")])

        return measurer

    def sweep_setup(
            self,
            start: float,
            stop: float,
            step_count: int,
            delay: float = 0,
            sweep_count: int = 1,
            range_mode: str = "AUTO"
    ) -> None:

        self._sweep_arguments = dict(
            start=start,
            stop=stop,
            step_count=step_count,
            delay=delay,
            sweep_count=sweep_count,
            range_mode=range_mode
        )

    def _get_sweep_axis(self):
        if self._sweep_arguments == {}:
            raise ValueError(
                "Before starting a sweep, please call 'sweep_setup'"
            )

        return np.linspace(
            start=self._sweep_arguments["start"],
            stop=self._sweep_arguments["stop"],
            num=self._sweep_arguments["step_count"],
        )

    @property
    def range(self) -> _BaseParameter:
        """
        Return the appropriate parameter based on the current function
        """
        function = self.function.get_latest() or self.function()
        param_name = f"_range_{function}"
        return self.parameters[param_name]

    @property
    def auto_range(self) -> _BaseParameter:
        """
        Return the appropriate parameter based on the current function
        """
        function = self.function.get_latest() or self.function()
        param_name = f"_auto_range_{function}"
        return self.parameters[param_name]

    @property
    def limit(self) -> _BaseParameter:
        """
        Return the appropriate parameter based on the current function
        """
        function = self.function.get_latest() or self.function()
        param_name = f"_limit_{function}"
        return self.parameters[param_name]

    @property
    def sweep_axis(self) -> _BaseParameter:
        function = self.function.get_latest() or self.function()
        param_name = f"_sweep_axis_{function}"
        return self.parameters[param_name]

    def npts(self):
        return len(self.sweep_axis())

    @property
    def sweep(self) -> _BaseParameter:
        """
        Return the appropriate parameter based on the current function
        """
        source_function = self.function.get_latest() or self.function()
        sense_module = cast(Sense2450, self.parent.sense)
        sense_function = sense_module.function.get_latest() or sense_module.function()
        param_name = f"_sweep_{source_function}_{sense_function}"
        return self.parameters[param_name]


class Keithley2450(VisaInstrument):
    """
    The QCoDeS driver for the Keithley 2450 source meter
    """

    def __init__(self, name: str, address: str, **kwargs) -> None:

        super().__init__(name, address, terminator='\n', **kwargs)

        if not self._has_correct_language_mode():
            self.log.warning(
                f"The instrument is an unsupported language mode."
                f"Please run `instrument.set_correct_language()` and try to "
                f"initialize the driver again after an instrument power cycle. "
                f"No parameters/sub modules shall be available on this driver "
                f"instance"
            )
            return

        self.add_parameter(
            "terminals",
            set_cmd="ROUTe:TERMinals {}",
            get_cmd="ROUTe:TERMinals?",
            vals=Enum("REAR", "FRONT")
        )

        self.add_parameter(
            "output",
            set_cmd=":OUTP {}",
            get_cmd=":OUTP?",
            val_mapping={
                True: "1",
                False: "0"
            }
        )

        self.add_submodule(
            "source",
            Source2450(self, "source")
        )

        self.add_submodule(
            "sense",
            Sense2450(self, "sense")
        )

        self.connect_message()

    def set_correct_language(self) -> None:
        """
        The correct communication protocol is SCPI, make sure this is set
        """
        self.raw_write("*LANG SCPI")
        self.log.warning("PLease power cycle the instrument to make the change take effect")
        # We want the user to be able to instantiate a driver with the same name
        self.close()

    def _has_correct_language_mode(self) -> bool:
        """
        Query if we have the correct language mode
        """
        return self.ask_raw("*LANG?") == "SCPI"
