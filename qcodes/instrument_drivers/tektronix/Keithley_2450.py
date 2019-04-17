import visa
import numpy as np
from typing import Any, Dict

from qcodes import VisaInstrument, InstrumentChannel, ArrayParameter
from qcodes.instrument.parameter import Parameter
from qcodes.utils.validators import Enum, Numbers


class ParameterError(Exception):
    """Raise this error if a parameter has an unexpected value"""
    pass


def assert_parameters(parameter_values: Dict[Parameter, Any]) -> None:
    """
    Args:
        parameter_values
    Returns:
        Callable: a decorator which asserts that the parameters specified in the
        input dictionary have the required values before calling the original method
    """
    for parameter, required_value in parameter_values.items():

        actual_value = parameter.get_latest() or parameter.get()
        if required_value != actual_value:

            instrument_name = parameter.instrument.name.replace("_", '.')

            raise ParameterError(
                f"'{instrument_name}.{parameter.name}()={actual_value}' "
                f"Please run '{instrument_name}.{parameter.name}({required_value})' and try again"
            )


class Keithley2450Sweep(ArrayParameter):
    """
    Array parameter to capture sweeps defined in the Keithley 2450 instrument
    """

    def __init__(self, name: str, instrument: 'Keithley2450') -> None:
        placeholder_shape = (0,)
        super().__init__(name, placeholder_shape, instrument)
        self._has_performed_setup = False

    def setup(
            self,
            start: float,
            stop: float,
            step_count: int,
            delay: float = 0,
            sweep_count: int = 1,
            range_mode: str = "AUTO"
    ) -> None:
        source_function_p = self.instrument.source.function
        source_function = source_function_p.get_latest() or source_function_p.function()
        source_unit = Source2450.function_modes[source_function]["unit"]

        self.setpoint_names = (source_function,)
        self.setpoint_labels = (source_function,)
        self.setpoint_units = (source_unit,)

        sense_function_p = self.instrument.sense.function
        sense_function = sense_function_p.get_latest() or sense_function_p.function()
        self.label = sense_function
        self.unit = Sense2450.function_modes[sense_function]["unit"]

        setpoints = np.linspace(start, stop, step_count)

        self.shape = (len(setpoints),)
        self.setpoints = (tuple(setpoints),)

        self.instrument.write_raw(
            f":SOURce:SWEep:{source_function}:LINear {start},{stop},{step_count},{delay},{sweep_count},{range_mode}"
        )
        self._has_performed_setup = True

    def get_raw(self) -> np.array:

        if not self._has_performed_setup:
            raise RuntimeError("Please setup the sweep before calling this function")

        self.instrument.write_raw(":INITiate")
        self.instrument.write_raw("*WAI")
        raw_data = self.instrument.ask_raw(f":TRACe:DATA? 1, {self.shape[0]}")
        self.instrument.write_raw(":TRACe:CLEar")
        self._has_performed_setup = False

        return np.array([float(i) for i in raw_data.split(",")])


class Sense2450(InstrumentChannel):
    function_modes = {
        "current": {
            "instrument_name": "\"CURR:DC\"",
            "unit": "A",
            "range_vals": Numbers(10E-9, 1)
        },
        "resistance": {
            "instrument_name": "RES",
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
                f"_measure_{function}",
                get_cmd=":MEASure?",
                get_parser=float,
                unit=args["unit"],
                snapshot_value=False
            )

    @property
    def four_wire_measurement(self) -> Parameter:
        """
        Return the appropriate parameter based on the current function
        """
        function = self.function.get_latest() or self.function()
        param_name = f"_four_wire_measurement_{function}"
        return self.parameters[param_name]

    @property
    def auto_range(self) -> Parameter:
        """
        Return the appropriate parameter based on the current function
        """
        function = self.function.get_latest() or self.function()
        param_name = f"_auto_range_{function}"
        return self.parameters[param_name]

    @property
    def range(self) -> Parameter:
        """
        Return the appropriate parameter based on the current function
        """
        function = self.function.get_latest() or self.function()
        param_name = f"_range_{function}"
        return self.parameters[param_name]

    @property
    def current(self) -> Parameter:
        """
        Return the parameter which performs a current measurement. Assert that we
        are in current mode first
        """

        assert_parameters({self.function: "current", self.parent.output: True})
        return self.parameters["_measure_current"]

    @property
    def voltage(self) -> Parameter:
        assert_parameters({self.function: "voltage", self.parent.output: True})
        return self.parameters["_measure_voltage"]


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

        self.add_parameter(
            "function",
            set_cmd="SOUR:FUNC {}",
            get_cmd="SOUR:FUNC?",
            val_mapping={
                key: self.function_modes[key]["instrument_name"]
                for key in self.function_modes
            }
        )

        for function, args in Source2450.function_modes.items():
            self.add_parameter(
                f"_range_{function}",
                set_cmd=f":SOUR:{function}:RANGe {{}}",
                get_cmd=f":SOUR:{function}:RANGe?",
                vals=args["range_vals"],
                get_parser=float,
                unit=args["unit"]
            )

            self.add_parameter(
                f"_auto_range_{function}",
                set_cmd=f":SOURce:{function}:RANGe:AUTO {{}}",
                get_cmd=f":SOURce:{function}:RANGe:AUTO?",
                val_mapping={
                    True: "1",
                    False: "0"
                }
            )

            self.add_parameter(
                f"_setpoint_{function}",
                set_cmd=f"SOUR:{function} {{}}",
                get_cmd=f"SOUR:{function}?",
                get_parser=float,
                unit=args["unit"]
            )

            limit_cmd = {"current": "VLIM", "voltage": "ILIM"}[function]
            self.add_parameter(
                f"_limit_{function}",
                set_cmd=f"SOUR:{function}:{limit_cmd} {{}}",
                get_cmd=f"SOUR:{function}:{limit_cmd}?",
                get_parser=float,
                unit=args["unit"]
            )

    @property
    def range(self) -> Parameter:
        """
        Return the appropriate parameter based on the current function
        """
        function = self.function.get_latest() or self.function()
        param_name = f"_range_{function}"
        return self.parameters[param_name]

    @property
    def auto_range(self) -> Parameter:
        """
        Return the appropriate parameter based on the current function
        """
        function = self.function.get_latest() or self.function()
        param_name = f"_auto_range_{function}"
        return self.parameters[param_name]

    @property
    def limit(self):
        """
        Return the appropriate parameter based on the current function
        """
        function = self.function.get_latest() or self.function()
        param_name = f"_limit_{function}"
        return self.parameters[param_name]

    @property
    def current(self) -> Parameter:
        """
        Return the parameter that sets/gets the current set point
        """

        assert_parameters({self.function: "current"})
        return self.parameters["_setpoint_current"]

    @property
    def voltage(self) -> Parameter:
        """
        Return the parameter that sets/gets the voltage set point
        """
        assert_parameters({self.function: "voltage"})
        return self.parameters["_setpoint_voltage"]


class Keithley2450(VisaInstrument):
    """
    The QCoDeS driver for the Keithley 2450 source meter
    """

    @staticmethod
    def set_correct_language(address: str) -> None:
        """
        The correct communication protocol is SCPI, make sure this is set

        Args:
            address: Visa resource address
        """
        Keithley2450._check_scpi_mode(address, raise_on_incorrect_setting=False)

    @staticmethod
    def _check_scpi_mode(address: str, raise_on_incorrect_setting: bool = True) -> None:
        """
        Args:
            address: Visa resource address
            raise_on_incorrect_setting: If True and the language mode is anything other then "SCPI",
                raise a runtime error. If False and the language mode is anything other then "SCPI"
                adjust the language to "SCPI"
        """

        resource_manager = visa.ResourceManager()
        raw_instrument = resource_manager.open_resource(address)
        language = raw_instrument.query("*LANG?").strip()

        if language != "SCPI":
            if raise_on_incorrect_setting:
                raise RuntimeError(
                    f"The instrument is in {language} mode which is not supported."
                    f"Please run `Keithley2450.set_correct_language(address)` and try to "
                    f"initialize the driver again"
                )
            else:
                raw_instrument.write("*LANG SCPI")
                print("PLease power cycle the instrument to make the change take effect")

        raw_instrument.close()

    def __init__(self, name: str, address: str, **kwargs) -> None:

        Keithley2450._check_scpi_mode(address)

        super().__init__(name, address, terminator='\n', **kwargs)

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
            "sense",
            Sense2450(self, "sense")
        )

        self.add_submodule(
            "source",
            Source2450(self, "source")
        )

        self.add_parameter(
            name="sweep",
            parameter_class=Keithley2450Sweep
        )

        self.connect_message()

