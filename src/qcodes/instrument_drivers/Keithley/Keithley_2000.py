from functools import partial
from typing import TYPE_CHECKING, Any

from qcodes.instrument import VisaInstrument, VisaInstrumentKWArgs
from qcodes.validators import Bool, Enum, Ints, MultiType, Numbers

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Unpack

    from qcodes.parameters import Parameter


def _parse_output_string(s: str) -> str:
    """Parses and cleans string outputs of the Keithley"""
    # Remove surrounding whitespace and newline characters
    s = s.strip()

    # Remove surrounding quotes
    if (s[0] == s[-1]) and s.startswith(("'", '"')):
        s = s[1:-1]

    s = s.lower()

    # Convert some results to a better readable version
    conversions = {
        "mov": "moving",
        "rep": "repeat",
    }

    if s in conversions.keys():
        s = conversions[s]

    return s


def _parse_output_bool(value: str) -> bool:
    return True if int(value) == 1 else False


class Keithley2000(VisaInstrument):
    """
    Driver for the Keithley 2000 multimeter.
    """

    default_terminator = "\n"

    def __init__(
        self,
        name: str,
        address: str,
        reset: bool = False,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ):
        super().__init__(name, address, **kwargs)

        self._trigger_sent = False

        # Unfortunately the strings have to contain quotation marks and a
        # newline character, as this is how the instrument returns it.
        self._mode_map = {
            "ac current": '"CURR:AC"',
            "dc current": '"CURR:DC"',
            "ac voltage": '"VOLT:AC"',
            "dc voltage": '"VOLT:DC"',
            "2w resistance": '"RES"',
            "4w resistance": '"FRES"',
            "temperature": '"TEMP"',
            "frequency": '"FREQ"',
        }

        self.mode: Parameter = self.add_parameter(
            "mode",
            get_cmd="SENS:FUNC?",
            set_cmd="SENS:FUNC {}",
            val_mapping=self._mode_map,
        )
        """Parameter mode"""

        # Mode specific parameters
        self.nplc: Parameter = self.add_parameter(
            "nplc",
            get_cmd=partial(self._get_mode_param, "NPLC", float),
            set_cmd=partial(self._set_mode_param, "NPLC"),
            vals=Numbers(min_value=0.01, max_value=10),
        )
        """Parameter nplc"""

        # TODO: validator, this one is more difficult since different modes
        # require different validation ranges
        self.range: Parameter = self.add_parameter(
            "range",
            get_cmd=partial(self._get_mode_param, "RANG", float),
            set_cmd=partial(self._set_mode_param, "RANG"),
            vals=Numbers(),
        )
        """Parameter range"""

        self.auto_range_enabled: Parameter = self.add_parameter(
            "auto_range_enabled",
            get_cmd=partial(self._get_mode_param, "RANG:AUTO", _parse_output_bool),
            set_cmd=partial(self._set_mode_param, "RANG:AUTO"),
            vals=Bool(),
        )
        """Parameter auto_range_enabled"""

        self.digits: Parameter = self.add_parameter(
            "digits",
            get_cmd=partial(self._get_mode_param, "DIG", int),
            set_cmd=partial(self._set_mode_param, "DIG"),
            vals=Ints(min_value=4, max_value=7),
        )
        """Parameter digits"""

        self.averaging_type: Parameter = self.add_parameter(
            "averaging_type",
            get_cmd=partial(self._get_mode_param, "AVER:TCON", _parse_output_string),
            set_cmd=partial(self._set_mode_param, "AVER:TCON"),
            vals=Enum("moving", "repeat"),
        )
        """Parameter averaging_type"""

        self.averaging_count: Parameter = self.add_parameter(
            "averaging_count",
            get_cmd=partial(self._get_mode_param, "AVER:COUN", int),
            set_cmd=partial(self._set_mode_param, "AVER:COUN"),
            vals=Ints(min_value=1, max_value=100),
        )
        """Parameter averaging_count"""

        self.averaging_enabled: Parameter = self.add_parameter(
            "averaging_enabled",
            get_cmd=partial(self._get_mode_param, "AVER:STAT", _parse_output_bool),
            set_cmd=partial(self._set_mode_param, "AVER:STAT"),
            vals=Bool(),
        )
        """Parameter averaging_enabled"""

        # Global parameters
        self.display_enabled: Parameter = self.add_parameter(
            "display_enabled",
            get_cmd="DISP:ENAB?",
            get_parser=_parse_output_bool,
            set_cmd="DISP:ENAB {}",
            set_parser=int,
            vals=Bool(),
        )
        """Parameter display_enabled"""

        self.trigger_continuous: Parameter = self.add_parameter(
            "trigger_continuous",
            get_cmd="INIT:CONT?",
            get_parser=_parse_output_bool,
            set_cmd="INIT:CONT {}",
            set_parser=int,
            vals=Bool(),
        )
        """Parameter trigger_continuous"""

        self.trigger_count: Parameter = self.add_parameter(
            "trigger_count",
            get_cmd="TRIG:COUN?",
            get_parser=int,
            set_cmd="TRIG:COUN {}",
            vals=MultiType(
                Ints(min_value=1, max_value=9999),
                Enum("inf", "default", "minimum", "maximum"),
            ),
        )
        """Parameter trigger_count"""

        self.trigger_delay: Parameter = self.add_parameter(
            "trigger_delay",
            get_cmd="TRIG:DEL?",
            get_parser=float,
            set_cmd="TRIG:DEL {}",
            unit="s",
            vals=Numbers(min_value=0, max_value=999999.999),
        )
        """Parameter trigger_delay"""

        self.trigger_source: Parameter = self.add_parameter(
            "trigger_source",
            get_cmd="TRIG:SOUR?",
            set_cmd="TRIG:SOUR {}",
            val_mapping={
                "immediate": "IMM",
                "timer": "TIM",
                "manual": "MAN",
                "bus": "BUS",
                "external": "EXT",
            },
        )
        """Parameter trigger_source"""

        self.trigger_timer: Parameter = self.add_parameter(
            "trigger_timer",
            get_cmd="TRIG:TIM?",
            get_parser=float,
            set_cmd="TRIG:TIM {}",
            unit="s",
            vals=Numbers(min_value=0.001, max_value=999999.999),
        )
        """Parameter trigger_timer"""

        self.amplitude: Parameter = self.add_parameter(
            "amplitude", unit="arb.unit", get_cmd=self._read_next_value
        )
        """Parameter amplitude"""

        self.add_function("reset", call_cmd="*RST")

        if reset:
            self.reset()

        # Set the data format to have only ascii data without units and channels
        self.write("FORM:DATA ASCII")
        self.write("FORM:ELEM READ")

        self.connect_message()

    def trigger(self) -> None:
        if not self.trigger_continuous():
            self.write("INIT")
            self._trigger_sent = True

    def _read_next_value(self) -> float:
        # Prevent a timeout when no trigger has been sent
        if not self.trigger_continuous() and not self._trigger_sent:
            return 0.0

        self._trigger_sent = False

        return float(self.ask("SENSE:DATA:FRESH?"))

    def _get_mode_param(
        self, parameter: str, parser: "Callable[[str], Any]"
    ) -> float | str | bool:
        """Read the current Keithley mode and ask for a parameter"""
        mode = _parse_output_string(self._mode_map[self.mode()])
        cmd = f"{mode}:{parameter}?"

        return parser(self.ask(cmd))

    def _set_mode_param(self, parameter: str, value: float | str | bool) -> None:
        """Read the current Keithley mode and set a parameter"""
        if isinstance(value, bool):
            value = int(value)

        mode = _parse_output_string(self._mode_map[self.mode()])
        cmd = f"{mode}:{parameter} {value}"

        self.write(cmd)
