# All manual references are to R&S RTO Digital Oscilloscope User Manual
# for firmware 3.65, 2017

import logging
import time
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from packaging import version

import qcodes.validators as vals
from qcodes.instrument import (
    Instrument,
    InstrumentBaseKWArgs,
    InstrumentChannel,
    VisaInstrument,
    VisaInstrumentKWArgs,
)
from qcodes.parameters import ArrayParameter, Parameter, create_on_off_val_mapping

if TYPE_CHECKING:
    from typing_extensions import Unpack

log = logging.getLogger(__name__)


class ScopeTrace(ArrayParameter):
    def __init__(
        self, name: str, instrument: InstrumentChannel, channum: int, **kwargs: Any
    ) -> None:
        """
        The ScopeTrace parameter is attached to a channel of the oscilloscope.

        For now, we only support reading out the entire trace.
        """
        super().__init__(
            name=name,
            shape=(1,),
            label="Voltage",  # TODO: Is this sometimes dbm?
            unit="V",
            setpoint_names=("Time",),
            setpoint_labels=("Time",),
            setpoint_units=("s",),
            docstring="Holds scope trace",
            snapshot_value=False,
            instrument=instrument,
            **kwargs,
        )

        self.channel = instrument
        self.channum = channum
        self._trace_ready = False

    def prepare_trace(self) -> None:
        """
        Prepare the scope for returning data, calculate the setpoints
        """
        assert self.root_instrument is not None

        # We always use 16 bit integers for the data format
        self.root_instrument.dataformat("INT,16")
        # ensure little-endianess
        self.root_instrument.write("FORMat:BORder LSBFirst")
        # only export y-values
        self.root_instrument.write("EXPort:WAVeform:INCXvalues OFF")
        # only export one channel
        self.root_instrument.write("EXPort:WAVeform:MULTichannel OFF")

        # now get setpoints

        hdr = self.root_instrument.ask(f"CHANnel{self.channum}:DATA:HEADER?")
        hdr_vals = list(map(float, hdr.split(",")))
        t_start = hdr_vals[0]
        t_stop = hdr_vals[1]
        no_samples = int(hdr_vals[2])
        values_per_sample = hdr_vals[3]

        # NOTE (WilliamHPNielsen):
        # If samples are multi-valued, we need a `MultiParameter`
        # instead of an `ArrayParameter`.
        if values_per_sample > 1:
            raise NotImplementedError(
                "There are several values per sample "
                "in this trace (are you using envelope"
                " or peak detect?). We currently do "
                "not support saving such a trace."
            )

        self.shape = (no_samples,)
        self.setpoints = (tuple(np.linspace(t_start, t_stop, no_samples)),)

        self._trace_ready = True
        # we must ensure that all this took effect before proceeding
        self.root_instrument.ask("*OPC?")

    def get_raw(self) -> np.ndarray:
        """
        Returns a trace
        """

        instr = self.root_instrument
        assert instr is not None

        if not self._trace_ready:
            raise ValueError("Trace not ready! Please call prepare_trace().")

        if instr.run_mode() == "RUN Nx SINGLE":
            total_acquisitions = instr.num_acquisitions()
            completed_acquisitions = instr.completed_acquisitions()
            log.info(f"Acquiring {total_acquisitions} traces.")
            while completed_acquisitions < total_acquisitions:
                log.info(f"Acquired {completed_acquisitions}:{total_acquisitions}")
                time.sleep(0.25)
                completed_acquisitions = instr.completed_acquisitions()

        log.info("Acquisition completed. Polling trace from instrument.")
        vh = instr.visa_handle
        vh.write(f"CHANnel{self.channum}:DATA?")
        raw_vals = vh.read_raw()

        num_length = int(raw_vals[1:2])
        no_points = int(raw_vals[2 : 2 + num_length])

        # cut of the header and the trailing '\n'
        raw_vals = raw_vals[2 + num_length : -1]

        dataformat = instr.dataformat.get_latest()

        if dataformat == "INT,8":
            int_vals = np.frombuffer(raw_vals, dtype=np.int8, count=no_points)
        else:
            int_vals = np.frombuffer(raw_vals, dtype=np.int16, count=no_points // 2)

        # now the integer values must be converted to physical
        # values

        scale = self.channel.scale()
        no_divs = 10  # TODO: Is this ever NOT 10?

        # we always export as 16 bit integers
        quant_levels = 253 * 256
        conv_factor = scale * no_divs / quant_levels
        output = conv_factor * int_vals + self.channel.offset()

        return output


class RohdeSchwarzRTO1000ScopeMeasurement(InstrumentChannel):
    """
    Class to hold a measurement of the scope.
    """

    def __init__(
        self,
        parent: Instrument,
        name: str,
        meas_nr: int,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        """
        Args:
            parent: The instrument to which the channel is attached
            name: The name of the measurement
            meas_nr: The number of the measurement in question. Must match the
                actual number as used by the instrument (1..8)
            **kwargs: Forwarded to base class.
        """

        if meas_nr not in range(1, 9):
            raise ValueError("Invalid measurement number; Min: 1, max 8")

        self.meas_nr = meas_nr
        super().__init__(parent, name, **kwargs)

        self.sources = vals.Enum(
            "C1W1",
            "C1W2",
            "C1W3",
            "C2W1",
            "C2W2",
            "C2W3",
            "C3W1",
            "C3W2",
            "C3W3",
            "C4W1",
            "C4W2",
            "C4W3",
            "M1",
            "M2",
            "M3",
            "M4",
            "R1",
            "R2",
            "R3",
            "R4",
            "SBUS1",
            "SBUS2",
            "SBUS3",
            "SBUS4",
            "D0",
            "D1",
            "D2",
            "D3",
            "D4",
            "D5",
            "D6",
            "D7",
            "D8",
            "D9",
            "D10",
            "D11",
            "D12",
            "D13",
            "D14",
            "D15",
            "TRK1",
            "TRK2",
            "TRK3",
            "TRK4",
            "TRK5",
            "TRK6",
            "TRK7",
            "TRK8",
            "SG1TL1",
            "SG1TL2",
            "SG2TL1",
            "SG2TL2",
            "SG3TL1",
            "SG3TL2",
            "SG4TL1",
            "SG4TL2",
            "Z1V1",
            "Z1V2",
            "Z1V3",
            "Z1V4",
            "Z1I1",
            "Z1I2",
            "Z1I3",
            "Z1I4",
            "Z2V1",
            "Z2V2",
            "Z2V3",
            "Z2V4",
            "Z2I1",
            "Z2I2",
            "Z2I3",
            "Z2I4",
        )

        self.categories = vals.Enum(
            "AMPTime", "JITTer", "EYEJitter", "SPECtrum", "HISTogram", "PROTocol"
        )

        self.meas_type = vals.Enum(
            # Amplitude/time measurements
            "HIGH",
            "LOW",
            "AMPLitude",
            "MAXimum",
            "MINimum",
            "PDELta",
            "MEAN",
            "RMS",
            "STDDev",
            "POVershoot",
            "NOVershoot",
            "AREA",
            "RTIMe",
            "FTIMe",
            "PPULse",
            "NPULse",
            "PERiod",
            "FREQuency",
            "PDCYcle",
            "NDCYcle",
            "CYCarea",
            "CYCMean",
            "CYCRms",
            "CYCStddev",
            "PULCnt",
            "DELay",
            "PHASe",
            "BWIDth",
            "PSWitching",
            "NSWitching",
            "PULSetrain",
            "EDGecount",
            "SHT",
            "SHR",
            "DTOTrigger",
            "PROBemeter",
            "SLERising",
            "SLEFalling",
            # Jitter measurements
            "CCJitter",
            "NCJitter",
            "CCWidth",
            "CCDutycycle",
            "TIE",
            "UINTerval",
            "DRATe",
            "SKWDelay",
            "SKWPhase",
            # Eye diagram measurements
            "ERPercent",
            "ERDB",
            "EHEight",
            "EWIDth",
            "ETOP",
            "EBASe",
            "QFACtor",
            "RMSNoise",
            "SNRatio",
            "DCDistortion",
            "ERTime",
            "EFTime",
            "EBRate",
            "EAMPlitude",
            "PPJitter",
            "STDJitter",
            "RMSJitter",
            # Spectrum measurements
            "CPOWer",
            "OBWidth",
            "SBWidth",
            "THD",
            "THDPCT",
            "THDA",
            "THDU",
            "THDR",
            "HAR",
            "PLISt",
            # Histogram measurements
            "WCOunt",
            "WSAMples",
            "HSAMples",
            "HPEak",
            "PEAK",
            "UPEakvalue",
            "LPEakvalue",
            "HMAXimum",
            "HMINimum",
            "MEDian",
            "MAXMin",
            "HMEan",
            "HSTDdev",
            "M1STddev",
            "M2STddev",
            "M3STddev",
            "MKPositive",
            "MKNegative",
        )

        self.enable: Parameter = self.add_parameter(
            "enable",
            label=f"Measurement {meas_nr} enable",
            set_cmd=f"MEASurement{meas_nr}:ENABle {{}}",
            vals=vals.Enum("ON", "OFF"),
            docstring="Switches the measurement on or off.",
        )
        """Switches the measurement on or off."""

        self.source: Parameter = self.add_parameter(
            "source",
            label=f"Measurement {meas_nr} source",
            set_cmd=f"MEASurement{meas_nr}:SOURce {{}}",
            vals=self.sources,
            docstring="Set the source of a measurement if the "
            "measurement only needs one source.",
        )
        """Set the source of a measurement if the measurement only needs one source."""

        self.source_first: Parameter = self.add_parameter(
            "source_first",
            label=f"Measurement {meas_nr} first source",
            set_cmd=f"MEASurement{meas_nr}:FSRC {{}}",
            vals=self.sources,
            docstring="Set the first source of a measurement"
            " if the measurement only needs multiple"
            " sources.",
        )
        """Set the first source of a measurement if the measurement only needs multiple sources."""

        self.source_second: Parameter = self.add_parameter(
            "source_second",
            label=f"Measurement {meas_nr} second source",
            set_cmd=f"MEASurement{meas_nr}:SSRC {{}}",
            vals=self.sources,
            docstring="Set the second source of a measurement"
            " if the measurement only needs multiple"
            " sources.",
        )
        """Set the second source of a measurement if the measurement only needs multiple sources."""

        self.category: Parameter = self.add_parameter(
            "category",
            label=f"Measurement {meas_nr} category",
            set_cmd=f"MEASurement{meas_nr}:CATegory {{}}",
            vals=self.categories,
            docstring="Set the category of a measurement.",
        )
        """Set the category of a measurement."""

        self.main: Parameter = self.add_parameter(
            "main",
            label=f"Measurement {meas_nr} main",
            set_cmd=f"MEASurement{meas_nr}:MAIN {{}}",
            vals=self.meas_type,
            docstring="Set the main of a measurement.",
        )
        """Set the main of a measurement."""

        self.statistics_enable: Parameter = self.add_parameter(
            "statistics_enable",
            label=f"Measurement {meas_nr} enable statistics",
            set_cmd=f"MEASurement{meas_nr}:STATistics:ENABle {{}}",
            vals=vals.Enum("ON", "OFF"),
            docstring="Switches the measurement on or off.",
        )
        """Switches the measurement on or off."""

        self.clear: Parameter = self.add_parameter(
            "clear",
            label=f"Measurement {meas_nr} clear statistics",
            set_cmd=f"MEASurement{meas_nr}:CLEar",
            docstring="Clears/reset measurement.",
        )
        """Clears/reset measurement."""

        self.event_count: Parameter = self.add_parameter(
            "event_count",
            label=f"Measurement {meas_nr} number of events",
            get_cmd=f"MEASurement{meas_nr}:RESult:EVTCount?",
            get_parser=int,
            docstring="Number of measurement results in the long-term measurement.",
        )
        """Number of measurement results in the long-term measurement."""

        self.result_avg: Parameter = self.add_parameter(
            "result_avg",
            label=f"Measurement {meas_nr} averages",
            get_cmd=f"MEASurement{meas_nr}:RESult:AVG?",
            get_parser=float,
            docstring="Average of the long-term measurement results.",
        )
        """Average of the long-term measurement results."""


ScopeMeasurement = RohdeSchwarzRTO1000ScopeMeasurement


class RohdeSchwarzRTO1000ScopeChannel(InstrumentChannel):
    """
    Class to hold an input channel of the scope.

    Exposes: state, coupling, ground, scale, range, position, offset,
    invert, bandwidth, impedance, overload.
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
            parent: The instrument to which the channel is attached
            name: The name of the channel
            channum: The number of the channel in question. Must match the
                actual number as used by the instrument (1..4)
            **kwargs: Forwarded to base class.
        """

        if channum not in [1, 2, 3, 4]:
            raise ValueError("Invalid channel number! Must be 1, 2, 3, or 4.")

        self.channum = channum

        super().__init__(parent, name, **kwargs)

        self.state: Parameter = self.add_parameter(
            "state",
            label=f"Channel {channum} state",
            get_cmd=f"CHANnel{channum}:STATe?",
            set_cmd=f"CHANnel{channum}:STATE {{}}",
            vals=vals.Enum("ON", "OFF"),
            docstring="Switches the channel on or off",
        )
        """Switches the channel on or off"""

        self.coupling: Parameter = self.add_parameter(
            "coupling",
            label=f"Channel {channum} coupling",
            get_cmd=f"CHANnel{channum}:COUPling?",
            set_cmd=f"CHANnel{channum}:COUPling {{}}",
            vals=vals.Enum("DC", "DCLimit", "AC"),
            docstring=(
                "Selects the connection of the channel "
                "signal. DC: 50 Ohm, DCLimit 1 MOhm, "
                "AC: Con. through DC capacitor"
            ),
        )
        """
        Selects the connection of the channel signal.
        DC: 50 Ohm, DCLimit 1 MOhm, AC: Con. through DC capacitor
        """

        self.ground: Parameter = self.add_parameter(
            "ground",
            label=f"Channel {channum} ground",
            get_cmd=f"CHANnel{channum}:GND?",
            set_cmd=f"CHANnel{channum}:GND {{}}",
            vals=vals.Enum("ON", "OFF"),
            docstring=("Connects/disconnects the signal to/from the ground."),
        )
        """Connects/disconnects the signal to/from the ground."""

        # NB (WilliamHPNielsen): This parameter depends on other parameters and
        # should be dynamically updated accordingly. Cf. p 1178 of the manual
        self.scale: Parameter = self.add_parameter(
            "scale",
            label=f"Channel {channum} Y scale",
            unit="V/div",
            get_cmd=f"CHANnel{channum}:SCALe?",
            set_cmd=self._set_scale,
            get_parser=float,
        )
        """Parameter scale"""

        self.range: Parameter = self.add_parameter(
            "range",
            label=f"Channel {channum} Y range",
            unit="V",
            get_cmd=f"CHANnel{channum}:RANGe?",
            set_cmd=self._set_range,
            get_parser=float,
        )
        """Parameter range"""

        # TODO (WilliamHPNielsen): would it be better to recast this in terms
        # of Volts?
        self.position: Parameter = self.add_parameter(
            "position",
            label=f"Channel {channum} vert. pos.",
            unit="div",
            get_cmd=f"CHANnel{channum}:POSition?",
            set_cmd=f"CHANnel{channum}:POSition {{}}",
            get_parser=float,
            vals=vals.Numbers(-5, 5),
            docstring=(
                "Positive values move the waveform up, negative values move it down."
            ),
        )
        """Positive values move the waveform up, negative values move it down."""

        self.offset: Parameter = self.add_parameter(
            "offset",
            label=f"Channel {channum} offset",
            unit="V",
            get_cmd=f"CHANnel{channum}:OFFSet?",
            set_cmd=f"CHANnel{channum}:OFFSet {{}}",
            get_parser=float,
        )
        """Parameter offset"""

        self.invert: Parameter = self.add_parameter(
            "invert",
            label=f"Channel {channum} inverted",
            get_cmd=f"CHANnel{channum}:INVert?",
            set_cmd=f"CHANnel{channum}:INVert {{}}",
            vals=vals.Enum("ON", "OFF"),
        )
        """Parameter invert"""

        # TODO (WilliamHPNielsen): This parameter should be dynamically
        # validated since 800 MHz BW is only available for 50 Ohm coupling
        self.bandwidth: Parameter = self.add_parameter(
            "bandwidth",
            label=f"Channel {channum} bandwidth",
            get_cmd=f"CHANnel{channum}:BANDwidth?",
            set_cmd=f"CHANnel{channum}:BANDwidth {{}}",
            vals=vals.Enum("FULL", "B800", "B200", "B20"),
        )
        """Parameter bandwidth"""

        self.impedance: Parameter = self.add_parameter(
            "impedance",
            label=f"Channel {channum} impedance",
            unit="Ohm",
            get_cmd=f"CHANnel{channum}:IMPedance?",
            set_cmd=f"CHANnel{channum}:IMPedance {{}}",
            vals=vals.Ints(1, 100000),
            docstring=(
                "Sets the impedance of the channel "
                "for power calculations and "
                "measurements."
            ),
        )
        """Sets the impedance of the channel for power calculations and measurements."""

        self.overload: Parameter = self.add_parameter(
            "overload",
            label=f"Channel {channum} overload",
            get_cmd=f"CHANnel{channum}:OVERload?",
        )
        """Parameter overload"""

        self.arithmetics: Parameter = self.add_parameter(
            "arithmetics",
            label=f"Channel {channum} arithmetics",
            set_cmd=f"CHANnel{channum}:ARIThmetics {{}}",
            get_cmd=f"CHANnel{channum}:ARIThmetics?",
            val_mapping={"AVERAGE": "AVER", "OFF": "OFF", "ENVELOPE": "ENV"},
        )
        """Parameter arithmetics"""

        self.trace: ScopeTrace = self.add_parameter(
            "trace", channum=self.channum, parameter_class=ScopeTrace
        )
        """Parameter trace"""

        self._trace_ready = False

    # Specialised/interlinked set/getters
    def _set_range(self, value: float) -> None:
        self.scale.cache.set(value / 10)

        self._parent.write(f"CHANnel{self.channum}:RANGe {value}")

    def _set_scale(self, value: float) -> None:
        self.range.cache.set(value * 10)

        self._parent.write(f"CHANnel{self.channum}:SCALe {value}")


ScopeChannel = RohdeSchwarzRTO1000ScopeChannel


class RohdeSchwarzRTO1000(VisaInstrument):
    """
    QCoDeS Instrument driver for the
    Rohde-Schwarz RTO1000 series oscilloscopes.

    """

    default_timeout = 5.0
    default_terminator = "\n"

    def __init__(
        self,
        name: str,
        address: str,
        *,
        model: str | None = None,
        HD: bool = True,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ) -> None:
        """
        Args:
            name: name of the instrument
            address: VISA resource address
            model: The instrument model. For newer firmware versions,
                this can be auto-detected
            HD: Does the unit have the High Definition Option (allowing
                16 bit vertical resolution)
            **kwargs: kwargs are forwarded to base class.
        """
        super().__init__(name=name, address=address, **kwargs)

        # With firmware versions earlier than 3.65, it seems that the
        # model number can NOT be queried from the instrument
        # (at least fails with RTO1024, fw 2.52.1.1), so in that case
        # the user must provide the model manually.
        firmware_version_str = self.get_idn()["firmware"]
        if firmware_version_str is None:
            raise RuntimeError("Could not determine firmware version of RTO1000.")
        firmware_version = version.parse(firmware_version_str)

        if firmware_version < version.parse("3"):
            log.warning(
                "Old firmware version detected. This driver may "
                "not be compatible. Please upgrade your firmware."
            )

        if firmware_version >= version.parse("3.65"):
            # strip just in case there is a newline character at the end
            self.model = self.ask("DIAGnostic:SERVice:WFAModel?").strip()
            if model is not None and model != self.model:
                warnings.warn(
                    "The model number provided by the user "
                    "does not match the instrument's response."
                    " I am going to assume that this oscilloscope "
                    f"is a model {self.model}"
                )
        elif model is None:
            raise ValueError(
                "No model number provided. Please provide "
                'a model number (eg. "RTO1024").'
            )
        else:
            self.model = model

        self.HD = HD

        # Now assign model-specific values
        self.num_chans = int(self.model[-1])
        self.num_meas = 8

        self._horisontal_divs = int(self.ask("TIMebase:DIVisions?"))

        self.display: Parameter = self.add_parameter(
            "display",
            label="Display state",
            set_cmd="SYSTem:DISPlay:UPDate {}",
            val_mapping={"remote": 0, "view": 1},
        )
        """Parameter display"""

        # Triggering

        self.trigger_display: Parameter = self.add_parameter(
            "trigger_display",
            label="Trigger display state",
            set_cmd="DISPlay:TRIGger:LINes {}",
            get_cmd="DISPlay:TRIGger:LINes?",
            val_mapping={"ON": 1, "OFF": 0},
        )
        """Parameter trigger_display"""

        # TODO: (WilliamHPNielsen) There are more available trigger
        # settings than implemented here. See p. 1261 of the manual
        # here we just use trigger1, which is the A-trigger

        self.trigger_source: Parameter = self.add_parameter(
            "trigger_source",
            label="Trigger source",
            set_cmd="TRIGger1:SOURce {}",
            get_cmd="TRIGger1:SOURce?",
            val_mapping={
                "CH1": "CHAN1",
                "CH2": "CHAN2",
                "CH3": "CHAN3",
                "CH4": "CHAN4",
                "EXT": "EXT",
            },
        )
        """Parameter trigger_source"""

        self.trigger_mode: Parameter = self.add_parameter(
            "trigger_mode",
            label="Trigger mode",
            set_cmd="TRIGger:MODE {}",
            get_cmd="TRIGger1:SOURce?",
            vals=vals.Enum("AUTO", "NORMAL", "FREERUN"),
            docstring="Sets the trigger mode which determines"
            " the behaviour of the instrument if no"
            " trigger occurs.\n"
            "Options: AUTO, NORMAL, FREERUN.",
            unit="none",
        )
        """Sets the trigger mode which determines the behaviour of the instrument if no trigger occurs.
Options: AUTO, NORMAL, FREERUN."""

        self.trigger_type: Parameter = self.add_parameter(
            "trigger_type",
            label="Trigger type",
            set_cmd="TRIGger1:TYPE {}",
            get_cmd="TRIGger1:TYPE?",
            val_mapping={
                "EDGE": "EDGE",
                "GLITCH": "GLIT",
                "WIDTH": "WIDT",
                "RUNT": "RUNT",
                "WINDOW": "WIND",
                "TIMEOUT": "TIM",
                "INTERVAL": "INT",
                "SLEWRATE": "SLEW",
                "DATATOCLOCK": "DAT",
                "STATE": "STAT",
                "PATTERN": "PATT",
                "ANEDGE": "ANED",
                "SERPATTERN": "SERP",
                "NFC": "NFC",
                "TV": "TV",
                "CDR": "CDR",
            },
        )
        """Parameter trigger_type"""
        # See manual p. 1262 for an explanation of trigger types

        self.trigger_level: Parameter = self.add_parameter(
            "trigger_level",
            label="Trigger level",
            set_cmd=self._set_trigger_level,
            get_cmd=self._get_trigger_level,
        )
        """Parameter trigger_level"""

        self.trigger_edge_slope: Parameter = self.add_parameter(
            "trigger_edge_slope",
            label="Edge trigger slope",
            set_cmd="TRIGger1:EDGE:SLOPe {}",
            get_cmd="TRIGger1:EDGE:SLOPe?",
            vals=vals.Enum("POS", "NEG", "EITH"),
        )
        """Parameter trigger_edge_slope"""

        # Horizontal settings

        self.timebase_scale: Parameter = self.add_parameter(
            "timebase_scale",
            label="Timebase scale",
            set_cmd=self._set_timebase_scale,
            get_cmd="TIMebase:SCALe?",
            unit="s/div",
            get_parser=float,
            vals=vals.Numbers(25e-12, 10000),
        )
        """Parameter timebase_scale"""

        self.timebase_range: Parameter = self.add_parameter(
            "timebase_range",
            label="Timebase range",
            set_cmd=self._set_timebase_range,
            get_cmd="TIMebase:RANGe?",
            unit="s",
            get_parser=float,
            vals=vals.Numbers(250e-12, 100e3),
        )
        """Parameter timebase_range"""

        self.timebase_position: Parameter = self.add_parameter(
            "timebase_position",
            label="Horizontal position",
            set_cmd=self._set_timebase_position,
            get_cmd="TIMEbase:HORizontal:POSition?",
            get_parser=float,
            unit="s",
            vals=vals.Numbers(-100e24, 100e24),
        )
        """Parameter timebase_position"""

        # Acquisition

        # I couldn't find a way to query the run mode, so we manually keep
        # track of it. It is very important when getting the trace to make
        # sense of completed_acquisitions.
        self.run_mode: Parameter = self.add_parameter(
            "run_mode",
            label="Run/acquisition mode of the scope",
            get_cmd=None,
            set_cmd=None,
        )
        """Parameter run_mode"""

        self.run_mode("RUN CONT")

        self.num_acquisitions: Parameter = self.add_parameter(
            "num_acquisitions",
            label="Number of single acquisitions to perform",
            get_cmd="ACQuire:COUNt?",
            set_cmd="ACQuire:COUNt {}",
            vals=vals.Ints(1, 16777215),
            get_parser=int,
        )
        """Parameter num_acquisitions"""

        self.completed_acquisitions: Parameter = self.add_parameter(
            "completed_acquisitions",
            label="Number of completed acquisitions",
            get_cmd="ACQuire:CURRent?",
            get_parser=int,
        )
        """Parameter completed_acquisitions"""

        self.sampling_rate: Parameter = self.add_parameter(
            "sampling_rate",
            label="Sample rate",
            docstring="Number of averages for measuring trace.",
            unit="Sa/s",
            get_cmd="ACQuire:POINts:ARATe" + "?",
            get_parser=int,
        )
        """Number of averages for measuring trace."""

        self.acquisition_sample_rate: Parameter = self.add_parameter(
            "acquisition_sample_rate",
            label="Acquisition sample rate",
            unit="Sa/s",
            docstring="recorded waveform samples per second",
            get_cmd="ACQuire:SRATe" + "?",
            set_cmd="ACQuire:SRATe " + " {:.2f}",
            vals=vals.Numbers(2, 20e12),
            get_parser=float,
        )
        """recorded waveform samples per second"""

        # Data

        self.dataformat: Parameter = self.add_parameter(
            "dataformat",
            label="Export data format",
            set_cmd="FORMat:DATA {}",
            get_cmd="FORMat:DATA?",
            vals=vals.Enum("ASC,0", "REAL,32", "INT,8", "INT,16"),
        )
        """Parameter dataformat"""

        # High definition mode (might not be available on all instruments)

        if HD:
            self.high_definition_state: Parameter = self.add_parameter(
                "high_definition_state",
                label="High definition (16 bit) state",
                set_cmd=self._set_hd_mode,
                get_cmd="HDEFinition:STAte?",
                val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
                docstring="Sets the filter bandwidth for the"
                " high definition mode.\n"
                "ON: high definition mode, up to 16"
                " bit digital resolution\n"
                "Options: ON, OFF\n\n"
                "Warning/Bug: By opening the HD "
                "acquisition menu on the scope, "
                'this value will be set to "ON".',
            )
            """Sets the filter bandwidth for the high definition mode.
ON: high definition mode, up to 16 bit digital resolution
Options: ON, OFF

Warning/Bug: By opening the HD acquisition menu on the scope, this value will be set to "ON"."""

            self.high_definition_bandwidth: Parameter = self.add_parameter(
                "high_definition_bandwidth",
                label="High definition mode bandwidth",
                set_cmd="HDEFinition:BWIDth {}",
                get_cmd="HDEFinition:BWIDth?",
                unit="Hz",
                get_parser=float,
                vals=vals.Numbers(1e4, 1e9),
            )
            """Parameter high_definition_bandwidth"""

        self.error_count: Parameter = self.add_parameter(
            "error_count",
            label="Number of errors in the error stack",
            get_cmd="SYSTem:ERRor:COUNt?",
            unit="#",
            get_parser=int,
        )
        """Parameter error_count"""

        self.error_next: Parameter = self.add_parameter(
            "error_next",
            label="Next error from the error stack",
            get_cmd="SYSTem:ERRor:NEXT?",
            get_parser=str,
        )
        """Parameter error_next"""

        # Add the channels to the instrument
        for ch in range(1, self.num_chans + 1):
            chan = RohdeSchwarzRTO1000ScopeChannel(self, f"channel{ch}", ch)
            self.add_submodule(f"ch{ch}", chan)

        for measId in range(1, self.num_meas + 1):
            measCh = RohdeSchwarzRTO1000ScopeMeasurement(
                self, f"measurement{measId}", measId
            )
            self.add_submodule(f"meas{measId}", measCh)

        self.add_function("stop", call_cmd="STOP")
        self.add_function("reset", call_cmd="*RST")
        self.opc: Parameter = self.add_parameter("opc", get_cmd="*OPC?")
        """Parameter opc"""
        self.stop_opc: Parameter = self.add_parameter("stop_opc", get_cmd="STOP;*OPC?")
        """Parameter stop_opc"""
        self.status_operation: Parameter = self.add_parameter(
            "status_operation", get_cmd="STATus:OPERation:CONDition?", get_parser=int
        )
        """Parameter status_operation"""
        self.add_function("run_continues", call_cmd="RUNContinous")
        # starts the shutdown of the system
        self.add_function("system_shutdown", call_cmd="SYSTem:EXIT")

        self.connect_message()

    def run_cont(self) -> None:
        """
        Set the instrument in 'RUN CONT' mode
        """
        self.write("RUN")
        self.run_mode.set("RUN CONT")

    def run_single(self) -> None:
        """
        Set the instrument in 'RUN Nx SINGLE' mode
        """
        self.write("SINGLE")
        self.run_mode.set("RUN Nx SINGLE")

    def is_triggered(self) -> bool:
        wait_trigger_mask = 0b01000
        return bool(self.status_operation() & wait_trigger_mask) is False

    def is_running(self) -> bool:
        measuring_mask = 0b10000
        return bool(self.status_operation() & measuring_mask)

    def is_acquiring(self) -> bool:
        return self.is_triggered() & self.is_running()

    # Specialised set/get functions

    def _set_hd_mode(self, value: int) -> None:
        """
        Set/unset the high def mode
        """
        self._make_traces_not_ready()
        self.write(f"HDEFinition:STAte {value}")

    def _set_timebase_range(self, value: float) -> None:
        """
        Set the full range of the timebase
        """
        self._make_traces_not_ready()
        self.timebase_scale.cache.set(value / self._horisontal_divs)

        self.write(f"TIMebase:RANGe {value}")

    def _set_timebase_scale(self, value: float) -> None:
        """
        Set the length of one horizontal division.
        """
        self._make_traces_not_ready()
        self.timebase_range.cache.set(value * self._horisontal_divs)

        self.write(f"TIMebase:SCALe {value}")

    def _set_timebase_position(self, value: float) -> None:
        """
        Set the horizontal position.
        """
        self._make_traces_not_ready()
        self.write(f"TIMEbase:HORizontal:POSition {value}")

    def _make_traces_not_ready(self) -> None:
        """
        Make the scope traces be not ready.
        """
        self.ch1.trace._trace_ready = False
        self.ch2.trace._trace_ready = False
        self.ch3.trace._trace_ready = False
        self.ch4.trace._trace_ready = False

    def _set_trigger_level(self, value: float) -> None:
        """
        Set the trigger level on the currently used trigger source
        channel.
        """
        trans = {"CH1": 1, "CH2": 2, "CH3": 3, "CH4": 4, "EXT": 5}
        # We use get and not get_latest because we don't trust users to
        # not touch the front panel of an oscilloscope.
        source = trans[self.trigger_source.get()]
        if source != 5:
            submodule = self.submodules[f"ch{source}"]
            assert isinstance(submodule, InstrumentChannel)
            v_range = submodule.range()
            offset = submodule.offset()

            if (value < -v_range / 2 + offset) or (value > v_range / 2 + offset):
                raise ValueError("Trigger level outside channel range.")

        self.write(f"TRIGger1:LEVel{source} {value}")

    def _get_trigger_level(self) -> float:
        """
        Get the trigger level from the currently used trigger source
        """
        trans = {"CH1": 1, "CH2": 2, "CH3": 3, "CH4": 4, "EXT": 5}
        # we use get and not get_latest because we don't trust users to
        # not touch the front panel of an oscilloscope
        source = trans[self.trigger_source.get()]

        val = self.ask(f"TRIGger1:LEVel{source}?")

        return float(val.strip())


class RTO1000(RohdeSchwarzRTO1000):
    """
    Backwards compatibility alias for RohdeSchwarzRTO1000
    """

    pass
