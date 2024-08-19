import logging
from functools import partial
from typing import TYPE_CHECKING

from typing_extensions import deprecated

from qcodes import validators as vals
from qcodes.instrument import (
    Instrument,
    InstrumentBaseKWArgs,
    InstrumentChannel,
    VisaInstrument,
    VisaInstrumentKWArgs,
)
from qcodes.utils import QCoDeSDeprecationWarning

from .private.error_handling import KeysightErrorQueueMixin

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.parameters import Parameter

log = logging.getLogger(__name__)


# This is to be the grand unified driver superclass for
# The Keysight/Agilent/HP Waveform generators of series
# 33200, 33500, and 33600


class Keysight33xxxOutputChannel(InstrumentChannel):
    """
    Class to hold the output channel of a Keysight 33xxxx waveform generator.
    """

    def __init__(
        self,
        parent: Instrument,
        name: str,
        channum: int,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        """
        Args:
            parent: The instrument to which the channel is
                attached.
            name: The name of the channel
            channum: The number of the channel in question (1-2)
            **kwargs: kwargs are forwarded to base class.
        """
        super().__init__(parent, name, **kwargs)

        def val_parser(parser: type, inputstring: str) -> float | int:
            """
            Parses return values from instrument. Meant to be used when a query
            can return a meaningful finite number or a numeric representation
            of infinity

            Args:
                parser: Either int or float, what to return in finite
                    cases
                inputstring: The raw return value
            """

            inputstring = inputstring.strip()

            if float(inputstring) == 9.9e37:
                output = float("inf")
            else:
                output = float(inputstring)
                if parser is int:
                    output = parser(output)

            return output

        self.model = self._parent.model

        self.function_type: Parameter = self.add_parameter(
            "function_type",
            label=f"Channel {channum} function type",
            set_cmd=f"SOURce{channum}:FUNCtion {{}}",
            get_cmd=f"SOURce{channum}:FUNCtion?",
            get_parser=str.rstrip,
            vals=vals.Enum(
                "SIN", "SQU", "TRI", "RAMP", "PULS", "PRBS", "NOIS", "ARB", "DC"
            ),
        )
        """Parameter function_type"""

        self.frequency_mode: Parameter = self.add_parameter(
            "frequency_mode",
            label=f"Channel {channum} frequency mode",
            set_cmd=f"SOURce{channum}:FREQuency:MODE {{}}",
            get_cmd=f"SOURce{channum}:FREQuency:MODE?",
            get_parser=str.rstrip,
            vals=vals.Enum("CW", "LIST", "SWEEP", "FIXED"),
        )
        """Parameter frequency_mode"""

        max_freq = self._parent._max_freqs[self.model]
        self.frequency: Parameter = self.add_parameter(
            "frequency",
            label=f"Channel {channum} frequency",
            set_cmd=f"SOURce{channum}:FREQuency {{}}",
            get_cmd=f"SOURce{channum}:FREQuency?",
            get_parser=float,
            unit="Hz",
            # TODO: max. freq. actually really tricky
            vals=vals.Numbers(1e-6, max_freq),
        )
        """Parameter frequency"""

        self.phase: Parameter = self.add_parameter(
            "phase",
            label=f"Channel {channum} phase",
            set_cmd=f"SOURce{channum}:PHASe {{}}",
            get_cmd=f"SOURce{channum}:PHASe?",
            get_parser=float,
            unit="deg",
            vals=vals.Numbers(0, 360),
        )
        """Parameter phase"""
        self.amplitude_unit: Parameter = self.add_parameter(
            "amplitude_unit",
            label=f"Channel {channum} amplitude unit",
            set_cmd=f"SOURce{channum}:VOLTage:UNIT {{}}",
            get_cmd=f"SOURce{channum}:VOLTage:UNIT?",
            vals=vals.Enum("VPP", "VRMS", "DBM"),
            get_parser=str.rstrip,
        )
        """Parameter amplitude_unit"""

        self.amplitude: Parameter = self.add_parameter(
            "amplitude",
            label=f"Channel {channum} amplitude",
            set_cmd=f"SOURce{channum}:VOLTage {{}}",
            get_cmd=f"SOURce{channum}:VOLTage?",
            unit="",  # see amplitude_unit
            get_parser=float,
        )
        """Parameter amplitude"""

        self.offset: Parameter = self.add_parameter(
            "offset",
            label=f"Channel {channum} voltage offset",
            set_cmd=f"SOURce{channum}:VOLTage:OFFSet {{}}",
            get_cmd=f"SOURce{channum}:VOLTage:OFFSet?",
            unit="V",
            get_parser=float,
        )
        """Parameter offset"""
        self.output: Parameter = self.add_parameter(
            "output",
            label=f"Channel {channum} output state",
            set_cmd=f"OUTPut{channum} {{}}",
            get_cmd=f"OUTPut{channum}?",
            val_mapping={"ON": 1, "OFF": 0},
        )
        """Parameter output"""

        self.ramp_symmetry: Parameter = self.add_parameter(
            "ramp_symmetry",
            label=f"Channel {channum} ramp symmetry",
            set_cmd=f"SOURce{channum}:FUNCtion:RAMP:SYMMetry {{}}",
            get_cmd=f"SOURce{channum}:FUNCtion:RAMP:SYMMetry?",
            get_parser=float,
            unit="%",
            vals=vals.Numbers(0, 100),
        )
        """Parameter ramp_symmetry"""

        self.pulse_width: Parameter = self.add_parameter(
            "pulse_width",
            label=f"Channel {channum} pulse width",
            set_cmd=f"SOURce{channum}:FUNCtion:PULSE:WIDTh {{}}",
            get_cmd=f"SOURce{channum}:FUNCtion:PULSE:WIDTh?",
            get_parser=float,
            unit="S",
        )
        """Parameter pulse_width"""

        # TRIGGER MENU
        self.trigger_source: Parameter = self.add_parameter(
            "trigger_source",
            label=f"Channel {channum} trigger source",
            set_cmd=f"TRIGger{channum}:SOURce {{}}",
            get_cmd=f"TRIGger{channum}:SOURce?",
            vals=vals.Enum("IMM", "EXT", "TIM", "BUS"),
            get_parser=str.rstrip,
        )
        """Parameter trigger_source"""

        self.trigger_slope: Parameter = self.add_parameter(
            "trigger_slope",
            label=f"Channel {channum} trigger slope",
            set_cmd=f"TRIGger{channum}:SLOPe {{}}",
            get_cmd=f"TRIGger{channum}:SLOPe?",
            vals=vals.Enum("POS", "NEG"),
            get_parser=str.rstrip,
        )
        """Parameter trigger_slope"""

        # Older models do not have all the fancy trigger options
        if self._parent.model[2] in ["5", "6"]:
            self.trigger_count: Parameter = self.add_parameter(
                "trigger_count",
                label=f"Channel {channum} trigger count",
                set_cmd=f"TRIGger{channum}:COUNt {{}}",
                get_cmd=f"TRIGger{channum}:COUNt?",
                vals=vals.Ints(1, 1000000),
                get_parser=partial(val_parser, int),
            )
            """Parameter trigger_count"""

            self.trigger_delay: Parameter = self.add_parameter(
                "trigger_delay",
                label=f"Channel {channum} trigger delay",
                set_cmd=f"TRIGger{channum}:DELay {{}}",
                get_cmd=f"TRIGger{channum}:DELay?",
                vals=vals.Numbers(0, 1000),
                get_parser=float,
                unit="s",
            )
            """Parameter trigger_delay"""

            self.trigger_timer: Parameter = self.add_parameter(
                "trigger_timer",
                label=f"Channel {channum} trigger timer",
                set_cmd=f"TRIGger{channum}:TIMer {{}}",
                get_cmd=f"TRIGger{channum}:TIMer?",
                vals=vals.Numbers(1e-6, 8000),
                get_parser=float,
            )
            """Parameter trigger_timer"""

        # TODO: trigger level doesn't work remotely. Why?

        # output menu
        self.output_polarity: Parameter = self.add_parameter(
            "output_polarity",
            label=f"Channel {channum} output polarity",
            set_cmd=f"OUTPut{channum}:POLarity {{}}",
            get_cmd=f"OUTPut{channum}:POLarity?",
            get_parser=str.rstrip,
            vals=vals.Enum("NORM", "INV"),
        )
        """Parameter output_polarity"""
        # BURST MENU
        self.burst_state: Parameter = self.add_parameter(
            "burst_state",
            label=f"Channel {channum} burst state",
            set_cmd=f"SOURce{channum}:BURSt:STATe {{}}",
            get_cmd=f"SOURce{channum}:BURSt:STATe?",
            val_mapping={"ON": 1, "OFF": 0},
            vals=vals.Enum("ON", "OFF"),
        )
        """Parameter burst_state"""

        self.burst_mode: Parameter = self.add_parameter(
            "burst_mode",
            label=f"Channel {channum} burst mode",
            set_cmd=f"SOURce{channum}:BURSt:MODE {{}}",
            get_cmd=f"SOURce{channum}:BURSt:MODE?",
            get_parser=str.rstrip,
            val_mapping={"N Cycle": "TRIG", "Gated": "GAT"},
            vals=vals.Enum("N Cycle", "Gated"),
        )
        """Parameter burst_mode"""

        self.burst_ncycles: Parameter = self.add_parameter(
            "burst_ncycles",
            label=f"Channel {channum} burst no. of cycles",
            set_cmd=f"SOURce{channum}:BURSt:NCYCles {{}}",
            get_cmd=f"SOURce{channum}:BURSt:NCYCLes?",
            get_parser=partial(val_parser, int),
            vals=vals.MultiType(vals.Ints(1), vals.Enum("MIN", "MAX", "INF")),
        )
        """Parameter burst_ncycles"""

        self.burst_phase: Parameter = self.add_parameter(
            "burst_phase",
            label=f"Channel {channum} burst start phase",
            set_cmd=f"SOURce{channum}:BURSt:PHASe {{}}",
            get_cmd=f"SOURce{channum}:BURSt:PHASe?",
            vals=vals.Numbers(-360, 360),
            unit="degrees",
            get_parser=float,
        )
        """Parameter burst_phase"""

        self.burst_polarity: Parameter = self.add_parameter(
            "burst_polarity",
            label=f"Channel {channum} burst gated polarity",
            set_cmd=f"SOURce{channum}:BURSt:GATE:POLarity {{}}",
            get_cmd=f"SOURce{channum}:BURSt:GATE:POLarity?",
            vals=vals.Enum("NORM", "INV"),
        )
        """Parameter burst_polarity"""

        self.burst_int_period: Parameter = self.add_parameter(
            "burst_int_period",
            label=(f"Channel {channum} burst internal period"),
            set_cmd=f"SOURce{channum}:BURSt:INTernal:PERiod {{}}",
            get_cmd=f"SOURce{channum}:BURSt:INTernal:PERiod?",
            unit="s",
            vals=vals.Numbers(1e-6, 8e3),
            get_parser=float,
            docstring=(
                "The burst period is the time "
                "between the starts of consecutive "
                "bursts when trigger is immediate."
            ),
        )
        """The burst period is the time between the starts of consecutive bursts when trigger is immediate."""


OutputChannel = Keysight33xxxOutputChannel


class Keysight33xxxSyncChannel(InstrumentChannel):
    """
    Class to hold the sync output of a Keysight 33xxxx waveform generator.
    Has very few parameters for single channel instruments.
    """

    def __init__(
        self,
        parent: Instrument,
        name: str,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ):
        super().__init__(parent, name, **kwargs)

        self.output: Parameter = self.add_parameter(
            "output",
            label="Sync output state",
            set_cmd="OUTPut:SYNC {}",
            get_cmd="OUTPut:SYNC?",
            val_mapping={"ON": 1, "OFF": 0},
            vals=vals.Enum("ON", "OFF"),
        )
        """Parameter output"""

        if parent.num_channels == 2:
            self.source: Parameter = self.add_parameter(
                "source",
                label="Source of sync function",
                set_cmd="OUTPut:SYNC:SOURce {}",
                get_cmd="OUTPut:SYNC:SOURce?",
                val_mapping={1: "CH1", 2: "CH2"},
                vals=vals.Enum(1, 2),
            )
            """Parameter source"""


SyncChannel = Keysight33xxxSyncChannel


class Keysight33xxx(KeysightErrorQueueMixin, VisaInstrument):
    """
    Base class for Keysight/Agilent 33XXX waveform generators.

    Not to be instantiated directly.
    """

    default_terminator = "\n"

    def __init__(
        self,
        name: str,
        address: str,
        silent: bool = False,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ):
        """
        Args:
            name: The name of the instrument used internally
                by QCoDeS. Must be unique.
            address: The VISA resource name.
            silent: If True, no connect message is printed.
            **kwargs: kwargs are forwarded to base class.
        """

        super().__init__(name, address, **kwargs)
        self.model = self.IDN()["model"]

        #######################################################################
        # Here go all model specific traits

        # TODO: Fill out this dict with all models
        no_of_channels = {
            "33210A": 1,
            "33250A": 1,
            "33511B": 1,
            "33512B": 2,
            "33522B": 2,
            "33622A": 2,
            "33510B": 2,
        }

        self._max_freqs = {
            "33210A": 10e6,
            "33511B": 20e6,
            "33512B": 20e6,
            "33250A": 80e6,
            "33522B": 30e6,
            "33622A": 120e6,
            "33510B": 20e6,
        }

        self.num_channels = no_of_channels[self.model]

        for i in range(1, self.num_channels + 1):
            channel = Keysight33xxxOutputChannel(self, f"ch{i}", i)
            self.add_submodule(f"ch{i}", channel)

        sync = Keysight33xxxSyncChannel(self, "sync")
        self.add_submodule("sync", sync)

        self.add_function("force_trigger", call_cmd="*TRG")

        self.add_function("sync_channel_phases", call_cmd="PHAS:SYNC")

        if not silent:
            self.connect_message()


@deprecated(
    "The base class for Keysight33xxx waveform generators has been renamed to Keysight33xxx",
    category=QCoDeSDeprecationWarning,
)
class WaveformGenerator_33XXX(Keysight33xxx):
    pass
