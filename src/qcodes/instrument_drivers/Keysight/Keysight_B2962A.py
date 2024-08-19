from typing import TYPE_CHECKING

from qcodes.instrument import (
    Instrument,
    InstrumentBaseKWArgs,
    InstrumentChannel,
    VisaInstrument,
    VisaInstrumentKWArgs,
)

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.parameters import Parameter


class KeysightB2962AChannel(InstrumentChannel):
    def __init__(
        self,
        parent: Instrument,
        name: str,
        chan: int,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        """
        InstrumentChannel that represents a singe channel of a
        KeysightB2962A.

        Args:
            parent: The instrument to which the channel is attached.
            name: The name of the channel
            chan: The number of the channel in question (1-2)
            **kwargs: Forwarded to base class.
        """
        # Sanity Check inputs
        if name not in ["ch1", "ch2"]:
            raise ValueError(f"Invalid Channel: {name}, expected 'ch1' or 'ch2'")
        if chan not in [1, 2]:
            raise ValueError(f"Invalid Channel: {chan}, expected '1' or '2'")

        super().__init__(parent, name, **kwargs)

        self.source_voltage: Parameter = self.add_parameter(
            "source_voltage",
            label=f"Channel {chan} Voltage",
            get_cmd=f"SOURCE{chan:d}:VOLT?",
            get_parser=float,
            set_cmd=f"SOURCE{chan:d}:VOLT {{:.8G}}",
            unit="V",
        )
        """Parameter source_voltage"""

        self.source_current: Parameter = self.add_parameter(
            "source_current",
            label=f"Channel {chan} Current",
            get_cmd=f"SOURCE{chan:d}:CURR?",
            get_parser=float,
            set_cmd=f"SOURCE{chan:d}:CURR {{:.8G}}",
            unit="A",
        )
        """Parameter source_current"""

        self.voltage: Parameter = self.add_parameter(
            "voltage",
            get_cmd=f"MEAS:VOLT? (@{chan:d})",
            get_parser=float,
            label=f"Channel {chan} Voltage",
            unit="V",
        )
        """Parameter voltage"""

        self.current: Parameter = self.add_parameter(
            "current",
            get_cmd=f"MEAS:CURR? (@{chan:d})",
            get_parser=float,
            label=f"Channel {chan} Current",
            unit="A",
        )
        """Parameter current"""

        self.resistance: Parameter = self.add_parameter(
            "resistance",
            get_cmd=f"MEAS:RES? (@{chan:d})",
            get_parser=float,
            label=f"Channel {chan} Resistance",
            unit="ohm",
        )
        """Parameter resistance"""

        self.voltage_limit: Parameter = self.add_parameter(
            "voltage_limit",
            get_cmd=f"SENS{chan:d}:VOLT:PROT?",
            get_parser=float,
            set_cmd=f"SENS{chan:d}:VOLT:PROT {{:.8G}}",
            label=f"Channel {chan} Voltage Limit",
            unit="V",
        )
        """Parameter voltage_limit"""

        self.current_limit: Parameter = self.add_parameter(
            "current_limit",
            get_cmd=f"SENS{chan:d}:CURR:PROT?",
            get_parser=float,
            set_cmd=f"SENS{chan:d}:CURR:PROT {{:.8G}}",
            label="Channel {} Current Limit",
            unit="A",
        )
        """Parameter current_limit"""

        self.enable: Parameter = self.add_parameter(
            "enable",
            get_cmd=f"OUTP{chan:d}?",
            set_cmd=f"OUTP{chan:d} {{:d}}",
            val_mapping={"on": 1, "off": 0},
        )
        """Parameter enable"""

        self.source_mode: Parameter = self.add_parameter(
            "source_mode",
            get_cmd=f":SOUR{chan:d}:FUNC:MODE?",
            set_cmd=f":SOUR{chan:d}:FUNC:MODE {{:s}}",
            val_mapping={"current": "CURR", "voltage": "VOLT"},
        )
        """Parameter source_mode"""

        self.channel = chan


B2962AChannel = KeysightB2962AChannel


class KeysightB2962A(VisaInstrument):
    """
    This is the qcodes driver for the Keysight B2962A 6.5 Digit Low Noise
    Power Source

    Status: alpha-version.

    Todo:
        - Implement any remaining parameters supported by the device
        - Similar drivers have special handlers to map return values of
          9.9e+37 to inf, is this needed?
    """

    default_terminator = "\n"

    def __init__(
        self, name: str, address: str, **kwargs: "Unpack[VisaInstrumentKWArgs]"
    ):
        super().__init__(name, address, **kwargs)

        # The B2962A supports two channels
        for ch_num in [1, 2]:
            ch_name = f"ch{ch_num:d}"
            channel = KeysightB2962AChannel(self, ch_name, ch_num)
            self.add_submodule(ch_name, channel)

        self.connect_message()

    def get_idn(self) -> dict[str, str | None]:
        IDN_str = self.ask_raw("*IDN?")
        vendor, model, serial, firmware = map(str.strip, IDN_str.split(","))
        IDN: dict[str, str | None] = {
            "vendor": vendor,
            "model": model,
            "serial": serial,
            "firmware": firmware,
        }
        return IDN


class B2962A(KeysightB2962A):
    """
    Alias for backwards compatibility
    """
