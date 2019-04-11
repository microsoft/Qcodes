import visa
import numpy as np

from qcodes import VisaInstrument, InstrumentChannel, ArrayParameter
from qcodes.instrument.parameter import Parameter
import qcodes.utils.validators as vals


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

        sense_function_p = self.instrument.sens.function
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
        "CURR": {
            "unit": "A",
            "vals": vals.Numbers(10E-9, 1)
        },
        "RES": {
            "unit": "Ohm",
            "vals": vals.Numbers(20, 200E6)
        },
        "VOLT": {
            "unit": "V",
            "vals": vals.Numbers(0.02, 200)
        }
    }

    def __init__(self, parent: 'Keithley2450', name: str) -> None:
        super().__init__(parent, name)

        self.add_parameter(
            "function",
            set_cmd=":SENS:FUNC \"{}\"",
            get_cmd=":SENS:FUNC?",
            vals=vals.Enum(*Sense2450.function_modes.keys()),
            get_parser=self._function_get_parser
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
                vals=args["vals"],
                get_parser=float,
                unit=args["unit"]
            )

    def _function_get_parser(self, get_value: str) -> str:
        """
        Args:
            get_value: ""CURR:DC"" (Note the quotation marks in the string)

        Returns:
            str: "CURR"
        """
        return_value = get_value.strip("\"")
        if return_value.endswith(":DC"):
            return_value = return_value.split(":")[0]

        return return_value

    @property
    def four_wire_measurement(self) -> Parameter:
        """
        Return the appropriate parameter based on the current function
        """
        function = self.function.get_latest() or self.function()
        param_name = f"_four_wire_measurement_{function}"
        return self.parameters[param_name]

    @property
    def range(self) -> Parameter:
        """
        Return the appropriate parameter based on the current function
        """
        function = self.function.get_latest() or self.function()
        param_name = f"_range_{function}"
        return self.parameters[param_name]


class Source2450(InstrumentChannel):
    function_modes = {
        "CURR": {
            "unit": "A",
            "vals": vals.Numbers(-1, 1)
        },
        "VOLT": {
            "unit": "V",
            "vals": vals.Numbers(-200, 200)
        }
    }

    def __init__(self, parent: 'Keithley2450', name: str) -> None:
        super().__init__(parent, name)

        self.add_parameter(
            "function",
            set_cmd="SOUR:FUNC {}",
            get_cmd="SOUR:FUNC?",
            vals=vals.Enum(*Source2450.function_modes.keys())
        )

        for function, args in Sense2450.function_modes.items():
            self.add_parameter(
                f"_range_{function}",
                set_cmd=f":SOUR:{function}:RANGe {{}}",
                get_cmd=f":SOUR:{function}:RANGe?",
                vals=args["vals"],
                get_parser=float,
                unit=args["unit"]
            )

            self.add_parameter(
                f"_setpoint_{function}",
                set_cmd=f"SOUR:{function} {{}}",
                get_cmd=f"SOUR:{function}?",
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
    def setpoint(self) -> Parameter:
        """
        Return the appropriate parameter based on the current function
        """
        function = self.function.get_latest() or self.function()
        param_name = f"_setpoint_{function}"
        return self.parameters[param_name]


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
            vals=vals.Enum("REAR", "FRONT")
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

        for function, args in Sense2450.function_modes.items():
            parameter_name = f"_measure_{function}"

            self.add_parameter(
                parameter_name,
                get_cmd=":MEASure?",
                get_parser=float,
                unit=args["unit"]
            )

        self.add_parameter(
            name="sweep",
            parameter_class=Keithley2450Sweep
        )

        self.connect_message()

    @property
    def measure(self) -> Parameter:
        """
        Return the appropriate parameter based on the current function
        """
        function = self.sense.function.get_latest() or self.sense.function()
        param_name = f"_measure_{function}"
        return self.parameters[param_name]
