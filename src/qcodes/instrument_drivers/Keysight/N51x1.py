from typing import TYPE_CHECKING, Optional

from typing_extensions import deprecated

from qcodes.instrument import VisaInstrument, VisaInstrumentKWArgs
from qcodes.parameters import Parameter, create_on_off_val_mapping
from qcodes.utils import QCoDeSDeprecationWarning
from qcodes.validators import Numbers

if TYPE_CHECKING:
    from typing_extensions import Unpack


class KeysightN51x1(VisaInstrument):
    """
    This is the qcodes driver for Keysight/Agilent scalar RF sources.
    It has been tested with N5171B, N5181A, N5173B, N5183B
    """

    default_terminator = "\n"

    def __init__(
        self,
        name: str,
        address: str,
        min_power: int = -144,
        max_power: int = 19,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ):
        super().__init__(name, address, **kwargs)

        self._options = self.ask("*OPT?")
        # Determine installed frequency option
        freq_dict = {
            "501": 1e9,
            "503": 3e9,
            "506": 6e9,
            "513": 13e9,
            "520": 20e9,
            "532": 31.8e9,
            "540": 40e9,
        }

        frequency_option = None
        for f_option in freq_dict.keys():
            if f_option in self._options:
                frequency_option = f_option
        if frequency_option is None:
            raise RuntimeError("Could not determine the frequency option")

        max_freq = freq_dict[frequency_option]

        self.power: Parameter = self.add_parameter(
            "power",
            label="Power",
            get_cmd="SOUR:POW?",
            get_parser=float,
            set_cmd="SOUR:POW {:.2f}",
            unit="dBm",
            vals=Numbers(min_value=min_power, max_value=max_power),
        )
        """Parameter power"""

        self.frequency: Parameter = self.add_parameter(
            "frequency",
            label="Frequency",
            get_cmd="SOUR:FREQ?",
            get_parser=float,
            set_cmd="SOUR:FREQ {:.2f}",
            unit="Hz",
            vals=Numbers(min_value=9e3, max_value=max_freq),
        )
        """Parameter frequency"""

        self.phase_offset: Parameter = self.add_parameter(
            "phase_offset",
            label="Phase Offset",
            get_cmd="SOUR:PHAS?",
            get_parser=float,
            set_cmd="SOUR:PHAS {:.2f}",
            unit="rad",
        )
        """Parameter phase_offset"""

        self.auto_freq_ref: Parameter = self.add_parameter(
            "auto_freq_ref",
            get_cmd=":ROSC:SOUR:AUTO?",
            set_cmd=":ROSC:SOUR:AUTO {}",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
        )
        """Parameter auto_freq_ref"""

        self.rf_output: Parameter = self.add_parameter(
            "rf_output",
            get_cmd="OUTP:STAT?",
            set_cmd="OUTP:STAT {}",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
        )
        """Parameter rf_output"""

        if "UNW" in self._options:
            self.pulse_modulation: Parameter = self.add_parameter(
                "pulse_modulation",
                get_cmd="PULM:STAT?",
                set_cmd="PULM:STAT {}",
                val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
            )
            """Parameter pulse_modulation"""

            self.pulse_modulation_source: Parameter = self.add_parameter(
                "pulse_modulation_source",
                get_cmd="PULM:SOUR?",
                set_cmd="PULM:SOUR {}",
                val_mapping={"internal": "INT", "external": "EXT"},
            )
            """Parameter pulse_modulation_source"""

        self.connect_message()

    def get_idn(self) -> dict[str, Optional[str]]:
        IDN_str = self.ask_raw('*IDN?')
        vendor, model, serial, firmware = map(str.strip, IDN_str.split(','))
        IDN: dict[str, Optional[str]] = {
            'vendor': vendor, 'model': model,
            'serial': serial, 'firmware': firmware}
        return IDN


@deprecated("Base class is renamed KeysightN51x1", category=QCoDeSDeprecationWarning)
class N51x1(KeysightN51x1):
    pass
