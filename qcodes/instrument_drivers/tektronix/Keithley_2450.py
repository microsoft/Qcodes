import re
from typing import Dict, Union, Any
import visa

from qcodes import VisaInstrument, InstrumentChannel
from qcodes.instrument.parameter import Parameter, _BaseParameter
import qcodes.utils.validators as vals


class JustInTimeParameter(_BaseParameter):
    """
    Part of the set and get command strings are resolved right before the
    `ask_raw` and `write_raw` are called.

    For instance, if the get_cmd is ":SENS:{function}:RANG?", then at
    call time we look up what the latest value of the
    "instrument.function" parameter is. If this is `CURR`, we send
    the string ":SENS:CURR:RANG?" to the instrument.

    Everything between the curly brackets needs to be a Parameter attached
    to the parent instrument.
    """

    def __init__(
            self,
            name: str,
            instrument: Union[VisaInstrument, InstrumentChannel],
            get_cmd: str,
            set_cmd: str,
            unit: str = None,
            **kwargs: Dict[str, Any]
    ) -> None:

        super().__init__(name, instrument, **kwargs)
        self.unit = unit
        self._get_cmd = get_cmd
        self._set_cmd = set_cmd

    def _resolve_kwargs(self, command_string: str) -> Dict[str, Any]:
        """
        Args:
            command_string: E.g. ":SENS:{function}:RANG?"

        Returns:
            dict: E.g. {"function": "current"} if self.instrument.current.get() = "current"
        """
        def get_param_value(parameter_name):
            param = self.instrument.parameters[parameter_name]
            value = param.get_latest()
            if value is None:
                value = param.get()

            return value

        parameter_names = re.findall(r"{(\w+)}", command_string)

        return {
            param_name: get_param_value(param_name)
            for param_name in parameter_names
        }

    def get_raw(self) -> Any:
        kwargs = self._resolve_kwargs(self._get_cmd)
        get_cmd = self._get_cmd.format(**kwargs)

        return self.instrument.ask_raw(get_cmd)

    def set_raw(self, value: Any) -> None:
        kwargs = self._resolve_kwargs(self._set_cmd)
        set_cmd = self._set_cmd.format(**kwargs)

        self.instrument.write_raw(set_cmd.format(value))


class Sense2450(InstrumentChannel):
    """Sense sub-module for the Keithley 2450 source meter"""
    range_limits = {
        "CURR": vals.Numbers(10E-9, 1),
        "RES": vals.Numbers(20, 200E6),
        "VOLT": vals.Numbers(0.02, 200)
    }

    def __init__(self, parent: 'Keithley2450', name: str) -> None:
        super().__init__(parent, name)

        self.add_parameter(
            "function",
            set_cmd=":SENS:FUNC \"{}\"",
            get_cmd=":SENS:FUNC?",
            vals=vals.Enum(
                "CURR",
                "VOLT",
                "RES"
            ),
            get_parser=self._function_get_parser
        )

        self.add_parameter(
            "four_wire_measurement",
            set_cmd=":SENSe:{function}:RSENse {{}}",
            get_cmd=":SENSe:{function}:RSENse?",
            val_mapping={
                True: "1",
                False: "0"
            },
            parameter_class=JustInTimeParameter
        )

        self.add_parameter(
            "range",
            set_cmd=":SENSe:{function}:RANGe {{}}",
            get_cmd=":SENSe:{function}:RANGe?",
            set_parser=self._range_validator,
            get_parser=float,
            parameter_class=JustInTimeParameter
        )

    @staticmethod
    def _function_get_parser(get_value: str) -> str:
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

    def _range_validator(self, range_value: float) -> float:
        """Validate that the range, given the function, is correct"""
        validator = Sense2450.range_limits[self.function()]
        validator.validate(range_value)
        return range_value


class Source2450(InstrumentChannel):
    """Source sub-module for the Keithley 2450 source meter"""
    range_limits = {
        "CURR": vals.Numbers(-1, 1),
        "VOLT": vals.Numbers(-200, 200)
    }

    def __init__(self, parent: 'Keithley2450', name: str):
        super().__init__(parent, name)

        self.add_parameter(
            "function",
            set_cmd="SOUR:FUNC {}",
            get_cmd="SOUR:FUNC?",
            vals=vals.Enum(
                "CURR",
                "VOLT"
            )
        )

        self.add_parameter(
            "range",
            set_cmd=":SOUR:{function}:RANGe {{}}",
            get_cmd=":SOUR:{function}:RANGe?",
            set_parser=self._range_validator,
            get_parser=float,
            parameter_class=JustInTimeParameter
        )

        self.add_parameter(
            "read_back",
            set_cmd=":SOUR:{function}:READ:BACK {{}}",
            get_cmd=":SOUR:{function}:READ:BACK?",
            vals=vals.Enum(
                "ON",
                "OFF"
            ),
            parameter_class=JustInTimeParameter
        )

        for source_name, unit in [("current", "A"), ("voltage", "V")]:
            self.add_parameter(
                f"{source_name}_setpoint",
                set_cmd=f"SOUR:{source_name} {{}}",
                get_cmd=f"SOUR:{source_name}?",
                unit=unit,
                get_parser=float
            )

    def _range_validator(self, range_value: float) -> float:
        """Validate that the range, given the function, is correct"""
        validator = Source2450.range_limits[self.function()]
        validator.validate(range_value)
        return range_value


class Keithley2450(VisaInstrument):
    """
    The QCoDeS driver for the Keithley 2450 source meter
    """
    @staticmethod
    def set_correct_protocol(address: str) -> None:
        """
        The correct communication protocol is SCPI, make sure this is set

        Args:
            address: Visa resource address
        """
        Keithley2450._check_and_adjust_protocol(address)

    @staticmethod
    def _check_and_adjust_protocol(address, raise_on_fail=False) -> None:
        """
        Args:
            address: Visa resource address
            raise_on_fail: If True and the language mode is anything other then "SCPI", raise a runtime error.
                If 'raise_on_fail' is False and the language mode is anything other then "SCPI"
                adjust the language to "SCPI"
        """
        resource_manager = visa.ResourceManager()
        raw_instrument = resource_manager.open_resource(address)
        language = raw_instrument.query("*LANG?")

        if language != "SCPI":
            if raise_on_fail:
                raise RuntimeError(
                    f"The instrument is in {language} mode which is not supported."
                    f"Please run `Keithley2450.set_correct_protocol(address)` and try to "
                    f"initialize the driver again"
                )
            else:
                raw_instrument.write("*LANG SCPI")
                print("PLease power cycle the instrument to make the change take effect")

        raw_instrument.close()

    def __init__(self, name: str, address: str, **kwargs) -> None:

        # Before doing anything else, make sure the instrument has the correct
        # protocol mode (SCPI), else raise a runtime error
        Keithley2450._check_and_adjust_protocol(address, raise_on_fail=True)

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

        self.connect_message()

    @property
    def measure(self) -> Parameter:

        units = {
            "CURR": "A",
            "VOLT": "V",
            "RES": "Ohm"
        }
        sense_function = self.sense.function()
        unit = units[sense_function]

        return Parameter(
            "measure",
            instrument=self,
            get_cmd=":MEASure?",
            unit=unit,
            get_parser=float
        )
