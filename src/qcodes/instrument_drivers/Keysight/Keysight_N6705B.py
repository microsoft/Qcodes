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


class KeysightN6705BChannel(InstrumentChannel):
    def __init__(
        self,
        parent: Instrument,
        name: str,
        chan: int,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        if chan not in [1, 2, 3, 4]:
            raise ValueError("Invalid channel specified")

        super().__init__(parent, name, **kwargs)

        self.source_voltage: Parameter = self.add_parameter(
            "source_voltage",
            label=f"Channel {chan} Voltage",
            get_cmd=f"SOURCE:VOLT? (@{chan})",
            get_parser=float,
            set_cmd=f"SOURCE:VOLT {{:.8G}}, (@{chan})",
            unit="V",
        )
        """Parameter source_voltage"""

        self.source_current: Parameter = self.add_parameter(
            "source_current",
            label=f"Channel {chan} Current",
            get_cmd=f"SOURCE:CURR? (@{chan})",
            get_parser=float,
            set_cmd=f"SOURCE:CURR {{:.8G}}, (@{chan})",
            unit="A",
        )
        """Parameter source_current"""

        self.voltage_limit: Parameter = self.add_parameter(
            "voltage_limit",
            get_cmd=f"SOUR:VOLT:PROT? (@{chan})",
            get_parser=float,
            set_cmd=f"SOUR:VOLT:PROT {{:.8G}}, @({chan})",
            label=f"Channel {chan} Voltage Limit",
            unit="V",
        )
        """Parameter voltage_limit"""

        self.voltage: Parameter = self.add_parameter(
            "voltage",
            get_cmd=f"MEAS:VOLT? (@{chan})",
            get_parser=float,
            label=f"Channel {chan} Voltage",
            unit="V",
        )
        """Parameter voltage"""

        self.current: Parameter = self.add_parameter(
            "current",
            get_cmd=f"MEAS:CURR? (@{chan})",
            get_parser=float,
            label=f"Channel {chan} Current",
            unit="A",
        )
        """Parameter current"""

        self.enable: Parameter = self.add_parameter(
            "enable",
            get_cmd=f"OUTP:STAT? (@{chan})",
            set_cmd=f"OUTP:STAT {{:d}}, (@{chan})",
            val_mapping={"on": 1, "off": 0},
        )
        """Parameter enable"""

        self.source_mode: Parameter = self.add_parameter(
            "source_mode",
            get_cmd=f":OUTP:PMOD? (@{chan})",
            set_cmd=f":OUTP:PMOD {{:s}}, (@{chan})",
            val_mapping={"current": "CURR", "voltage": "VOLT"},
        )
        """Parameter source_mode"""

        self.channel = chan
        self.ch_name = name


N6705BChannel = KeysightN6705BChannel
"""Alias for backwards compatibility"""


class KeysightN6705B(VisaInstrument):
    default_terminator = "\n"

    def __init__(
        self, name: str, address: str, **kwargs: "Unpack[VisaInstrumentKWArgs]"
    ) -> None:
        super().__init__(name, address, **kwargs)
        self.channels: list[KeysightN6705BChannel] = []
        for ch_num in [1, 2, 3, 4]:
            ch_name = f"ch{ch_num}"
            channel = KeysightN6705BChannel(self, ch_name, ch_num)
            self.add_submodule(ch_name, channel)
            self.channels.append(channel)

        self.connect_message()

    def get_idn(self) -> dict[str, str | None]:
        IDNstr = self.ask_raw("*IDN?")
        vendor, model, serial, firmware = map(str.strip, IDNstr.split(","))
        IDN: dict[str, str | None] = {
            "vendor": vendor,
            "model": model,
            "serial": serial,
            "firmware": firmware,
        }
        return IDN


class N6705B(KeysightN6705B):
    """
    Alias for backwards compatibility.
    """
