import re
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional, Union

import numpy as np
from pyvisa import VisaIOError
from pyvisa.constants import StatusCode

import qcodes.validators as vals
from qcodes.instrument import (
    ChannelList,
    InstrumentBase,
    InstrumentBaseKWArgs,
    InstrumentChannel,
    InstrumentModule,
    VisaInstrument,
    VisaInstrumentKWArgs,
)
from qcodes.parameters import (
    Parameter,
    ParameterBase,
    ParameterWithSetpoints,
    create_on_off_val_mapping,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Unpack


class DSOTimeAxisParam(Parameter):
    """
    Time axis parameter for the Infiniium series DSO.
    """

    def __init__(self, xorigin: float, xincrement: float, points: int, **kwargs: Any):
        """
        Initialize time axis. If values are unknown, they can be initialized to zero and
        filled in later.
        """
        super().__init__(**kwargs)

        self.xorigin = xorigin
        self.xincrement = xincrement
        self.points = points

    def get_raw(self) -> np.ndarray:
        """
        Return the array corresponding to this time axis.
        """
        return np.linspace(
            self.xorigin,
            self.xorigin + self.points * self.xincrement,
            self.points,
            endpoint=False,
        )


class DSOFrequencyAxisParam(Parameter):
    """
    Frequency axis parameter for the Infiniium series DSO.
    """

    def __init__(self, xorigin: float, xincrement: float, points: int, **kwargs: Any):
        """
        Initialize frequency axis. If values are unknown, they can be initialized
        to zero and filled in later.
        """
        super().__init__(**kwargs)

        self.xorigin = xorigin
        self.xincrement = xincrement
        self.points = points

    def get_raw(self) -> np.ndarray:
        """
        Return the array corresponding to this time axis.
        """
        return np.linspace(
            self.xorigin,
            self.xorigin + self.points * self.xincrement,
            self.points,
            endpoint=False,
        )


class DSOTraceParam(ParameterWithSetpoints):
    """
    Trace parameter for the Infiniium series DSO
    """

    UNIT_MAP: ClassVar[dict[int, str]] = {
        0: "UNKNOWN",
        1: "V",
        2: "s",
        3: "''",
        4: "A",
        5: "dB",
    }

    def __init__(
        self,
        name: str,
        instrument: Union["KeysightInfiniiumChannel", "KeysightInfiniiumFunction"],
        channel: str,
        **kwargs: Any,
    ):
        """
        Initialize DSOTraceParam bound to a specific channel.
        """
        self._ch_valid = False
        super().__init__(name, instrument=instrument, **kwargs)
        self._channel = channel
        # This parameter will be updated prior to being retrieved if
        # self.root_instrument.auto_digitize is true.
        self._points = 0
        self._yoffset = 0.0
        self._yincrement = 0.0
        self._unit = 0

    @property
    def setpoints(self) -> "Sequence[ParameterBase]":
        """
        Overwrite setpoint parameter to update setpoints if auto_digitize is true
        """
        instrument = self.instrument
        if isinstance(instrument, KeysightInfiniiumChannel):
            root_instrument: KeysightInfiniium
            root_instrument = self.root_instrument  # type: ignore[assignment]
            cache_setpoints = root_instrument.cache_setpoints()
            if not cache_setpoints:
                self.update_setpoints()
            return (instrument.time_axis,)
        elif isinstance(instrument, KeysightInfiniiumFunction):
            if instrument.function().startswith("FFT"):
                self.update_fft_setpoints()
                return (instrument.frequency_axis,)
            else:
                self.update_setpoints()
                return (instrument.time_axis,)
        raise RuntimeError("Invalid type for parent instrument.")

    @setpoints.setter
    def setpoints(self, setpoints: Any) -> None:
        """
        Stub to allow initialization. Ignore any set attempts on setpoint as we
        figure it out on the fly.
        """
        return

    @property
    def unit(self) -> str:
        """
        Return the units for this measurement.
        """
        if self._ch_valid is False:
            return "''"
        elif self._unit != 0:
            return self.UNIT_MAP[self._unit]
        elif self.instrument is not None:
            self.instrument.write(f":WAV:SOUR {self._channel}")
            return self.instrument.ask(":WAV:YUN?")
        return "''"

    @unit.setter
    def unit(self, unit: Any) -> None:
        """
        Stub to allow initialization.
        """
        return

    def update_setpoints(self, preamble: Optional["Sequence[str]"] = None) -> None:
        """
        Update waveform parameters. Must be called before data
        acquisition if instr.cache_setpoints is False
        """
        instrument: Union[KeysightInfiniiumChannel, KeysightInfiniiumFunction]
        instrument = self.instrument  # type: ignore[assignment]
        if preamble is None:
            instrument.write(f":WAV:SOUR {self._channel}")
            preamble = instrument.ask(":WAV:PRE?").strip().split(",")
        self._points = int(preamble[2])
        self._yincrement = float(preamble[7])
        self._yoffset = float(preamble[8])
        self._unit = int(preamble[21])
        instrument.time_axis.points = int(preamble[2])
        instrument.time_axis.xorigin = float(preamble[5])
        instrument.time_axis.xincrement = float(preamble[4])
        self._ch_valid = True

    def update_fft_setpoints(self) -> None:
        """
        Update waveform parameters for an FFT.
        """
        instrument: KeysightInfiniiumFunction = self.instrument  # type: ignore[assignment]
        instrument.write(f":WAV:SOUR {self._channel}")
        preamble = instrument.ask(":WAV:PRE?").strip().split(",")
        self.update_setpoints(preamble)
        instrument.frequency_axis.points = int(preamble[2])
        instrument.frequency_axis.xorigin = float(preamble[5])
        instrument.frequency_axis.xincrement = float(preamble[4])

    def get_raw(self) -> np.ndarray:
        """
        Get waveform data from scope
        """
        if self.instrument is None:
            raise RuntimeError("Cannot get data without instrument")
        root_instr: KeysightInfiniium = self.root_instrument  # type: ignore[assignment]
        # Check if we can use cached trace parameters
        if not root_instr.cache_setpoints():
            self.update_setpoints()
        if not self._ch_valid:
            raise RuntimeError(
                "Trace parameters are unknown. If cache_setpoints is True, "
                "you must manually call instr.chX.update_setpoints at least"
                "once prior to measurement."
            )

        # Check if we should run a new sweep
        if root_instr.auto_digitize():
            root_instr.digitize()
        # Ask for waveform data
        root_instr.write(f":WAV:SOUR {self._channel}")
        root_instr.write(":WAV:DATA?")
        # Ignore first two bytes, which should be "#0"
        _ = root_instr.visa_handle.read_bytes(2)
        data: np.ndarray
        data = root_instr.visa_handle.read_binary_values(  # type: ignore[assignment]
            "h",
            container=np.ndarray,
            header_fmt="empty",
            expect_termination=True,
            data_points=self._points,
        )
        data = data.astype(np.float64)
        data = (data * self._yincrement) + self._yoffset
        return data


class AbstractMeasurementSubsystem(InstrumentModule):
    """
    Submodule containing the measurement subsystem commands and associated
    parameters.

    Note: these commands are executed on the waveform in the scope buffer.
    If you need to ensure a fresh value, run dso.digitize() prior to reading
    the measurement value.
    """

    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        """
        Add parameters to measurement subsystem. Note: This should not be initialized
        directly, rather initialize BoundMeasurementSubsystem
        or UnboundMeasurementSubsystem.
        """
        super().__init__(parent, name, **kwargs)

        ###################################
        # Voltage Parameters
        self.amplitude: Parameter = Parameter(
            name="amplitude",
            instrument=self,
            label="Voltage amplitude",
            get_cmd=self._create_query("VAMP"),
            get_parser=float,
            unit="V",
            snapshot_value=False,
        )
        self.average: Parameter = Parameter(
            name="average",
            instrument=self,
            label="Voltage average",
            get_cmd=self._create_query("VAV", "DISP"),
            get_parser=float,
            unit="V",
            snapshot_value=False,
        )
        self.base: Parameter = Parameter(
            name="base",
            instrument=self,
            label="Statistical base",
            get_cmd=self._create_query("VBAS"),
            get_parser=float,
            unit="V",
            snapshot_value=False,
        )
        # Threshold Voltage Measurements - this measurement ignores overshoot
        # in the data
        self.vlow: Parameter = Parameter(
            name="vlow",
            instrument=self,
            label="Lower threshold voltage",
            get_cmd=self._create_query("VLOW"),
            get_parser=float,
            unit="V",
            snapshot_value=False,
        )
        self.vmid: Parameter = Parameter(
            name="vmid",
            instrument=self,
            label="Middle threshold voltage",
            get_cmd=self._create_query("VMID"),
            get_parser=float,
            unit="V",
            snapshot_value=False,
        )
        self.vup: Parameter = Parameter(
            name="vup",
            instrument=self,
            label="Upper threshold voltage",
            get_cmd=self._create_query("VUPP"),
            get_parser=float,
            unit="V",
            snapshot_value=False,
        )
        # Limit values - the minimum/maximum shown on screen
        self.vmin: Parameter = Parameter(
            name="vmin",
            instrument=self,
            label="Voltage minimum",
            get_cmd=self._create_query("VMIN"),
            get_parser=float,
            unit="V",
            snapshot_value=False,
        )
        self.vmax: Parameter = Parameter(
            name="vmax",
            instrument=self,
            label="Voltage maximum",
            get_cmd=self._create_query("VMAX"),
            get_parser=float,
            unit="V",
            snapshot_value=False,
        )
        # Waveform Parameters
        self.overshoot: Parameter = Parameter(
            name="overshoot",
            instrument=self,
            label="Voltage overshoot",
            get_cmd=self._create_query("VOV"),
            get_parser=float,
            unit="V",
            snapshot_value=False,
        )
        self.vpp = Parameter(
            name="vpp",
            instrument=self,
            label="Voltage peak-to-peak",
            get_cmd=self._create_query("VPP"),
            get_parser=float,
            unit="V",
            snapshot_value=False,
        )
        self.vrms: Parameter = Parameter(
            name="vrms",
            instrument=self,
            label="Voltage RMS",
            get_cmd=self._create_query("VRMS", "CYCL,AC"),
            get_parser=float,
            unit="V_rms",
            snapshot_value=False,
        )
        self.vrms_dc: Parameter = Parameter(
            name="vrms_dc",
            instrument=self,
            label="Voltage RMS with DC Component",
            get_cmd=self._create_query("VRMS", "CYCL,DC"),
            get_parser=float,
            unit="V_rms",
            snapshot_value=False,
        )

        ###################################
        # Time Parameters
        self.rise_time: Parameter = Parameter(
            name="rise_time",
            instrument=self,
            label="Rise time",
            get_cmd=self._create_query("RIS"),
            get_parser=float,
            unit="s",
            snapshot_value=False,
        )
        self.fall_time: Parameter = Parameter(
            name="fall_time",
            instrument=self,
            label="Fall time",
            get_cmd=self._create_query("FALL"),
            get_parser=float,
            unit="s",
            snapshot_value=False,
        )
        self.duty_cycle: Parameter = Parameter(
            name="duty_cycle",
            instrument=self,
            label="Duty Cycle",
            get_cmd=self._create_query("DUTY"),
            get_parser=float,
            unit="%",
            snapshot_value=False,
        )
        self.period: Parameter = Parameter(
            name="period",
            instrument=self,
            label="Period",
            get_cmd=self._create_query("PER"),
            get_parser=float,
            unit="s",
            snapshot_value=False,
        )
        self.frequency: Parameter = Parameter(
            name="frequency",
            instrument=self,
            label="Signal frequency",
            get_cmd=self._create_query("FREQ"),
            get_parser=float,
            unit="Hz",
            docstring="""
                                     measure the frequency of the first
                                     complete cycle on the screen using
                                     the mid-threshold levels of the waveform
                                     """,
            snapshot_value=False,
        )
        self.slew_rate: Parameter = Parameter(
            name="slew_rate",
            instrument=self,
            label="Slew rate",
            get_cmd=self._create_query("SLEW"),
            get_parser=float,
            unit="S",
            snapshot_value=False,
        )

        ###################################
        # Deprecated parameter aliases
        self.rms = self.vrms_dc
        self.rms_no_dc = self.vrms
        self.min = self.vmin
        self.middle = self.vmid
        self.max = self.vmax
        self.lower = self.vlow

    def _create_query(self, cmd: str, pre_cmd: str = "", post_cmd: str = "") -> str:
        """
        Create a query string with the correct source included
        """
        chan_str = self._channel
        if chan_str:
            if pre_cmd:
                chan_str = f",{chan_str}"
            if post_cmd:
                chan_str = f"{chan_str},"
        elif pre_cmd and post_cmd:
            pre_cmd = f"{pre_cmd},"
        return f":MEAS:{cmd}? {pre_cmd}{chan_str}{post_cmd}".strip()


class KeysightInfiniiumBoundMeasurement(AbstractMeasurementSubsystem):
    def __init__(
        self,
        parent: Union["KeysightInfiniiumChannel", "KeysightInfiniiumFunction"],
        name: str,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ):
        """
        Initialize measurement subsystem bound to a specific channel
        """
        # Bind the channel
        self._channel = parent.channel_name

        # Initialize measurement parameters
        super().__init__(parent, name, **kwargs)


BoundMeasurement = KeysightInfiniiumBoundMeasurement
"""
Alias for backwards compatibility
"""


class KeysightInfiniiumUnboundMeasurement(AbstractMeasurementSubsystem):
    def __init__(
        self,
        parent: "KeysightInfiniium",
        name: str,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ):
        """
        Initialize measurement subsystem where target is set by the parameter `source`.
        """
        # Blank channel
        self._channel = ""

        # Initialize measurement parameters
        super().__init__(parent, name, **kwargs)

        self.source = Parameter(
            name="source",
            instrument=self,
            label="Primary measurement source",
            set_cmd=self._set_source,
            get_cmd=self._get_source,
            snapshot_value=False,
        )

    def _validate_source(self, source: str) -> str:
        """Validate and set the source."""
        valid_channels = f"CHAN[1-{self.root_instrument.no_channels}]"
        if re.fullmatch(valid_channels, source):
            if not int(self.ask(f"CHAN{source[-1]}:DISP?")):
                raise ValueError(f"Channel {source[-1]} not turned on.")
            return source
        if re.fullmatch("DIFF[1-2]", source):
            diff_chan = (int(source[-1]) - 1) * 2 + 1
            if int(self.ask(f"CHAN{diff_chan}:DIFF?")) != 1:
                raise ValueError(f"Differential channel {source[-1]} not turned on.")
            return source
        if re.fullmatch("COMM[1-2]", source):
            diff_chan = (int(source[-1]) - 1) * 2 + 1
            if int(self.ask(f"CHAN{diff_chan}:DIFF?")) != 1:
                raise ValueError(f"Differential channel {source[-1]} not turned on.")
            return source
        if re.fullmatch("WMEM[1-4]", source):
            return source
        match = re.fullmatch("FUNC([1-9]{1,2})", source)
        if match:
            func_chan = int(match.groups()[0])
            if not (1 <= func_chan <= 16):
                raise ValueError(
                    f"Function number should be in the range 1-16. Got {func_chan}."
                )
            if not int(self.ask(f"FUNC{func_chan}:DISP?")):
                raise ValueError(f"Function {func_chan} is not enabled.")
            return f"FUNC{func_chan}"

        raise ValueError(
            f"Invalid measurement source {source}. Valid values are: ("
            "CHAN[1-4], DIFF[1-2], COMM[1-2], WMEM[1-4], FUNC[1-16])."
        )

    def _set_source(self, source: str) -> None:
        source = self._validate_source(source)
        self._channel = source

        # Then set the measurement source
        self.write(f":MEAS:SOUR {self._channel}")

    def _get_source(self) -> str:
        if self._channel == "":
            source = self.ask(":MEAS:SOUR?")
            self._channel = source.strip().split(",")[0]
        return self._channel


UnboundMeasurement = KeysightInfiniiumUnboundMeasurement
"""
Alias for backwards compatibility
"""


class KeysightInfiniiumFunction(InstrumentChannel):
    def __init__(
        self,
        parent: "KeysightInfiniium",
        name: str,
        channel: int,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ):
        """
        Initialize an infiniium channel.
        """
        self._channel = channel
        super().__init__(parent, name, **kwargs)

        # display
        self.display = Parameter(
            name="display",
            instrument=self,
            label=f"Function {channel} display on/off",
            set_cmd=f"FUNC{channel}:DISP {{}}",
            get_cmd=f"FUNC{channel}:DISP?",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
        )

        # Retrieve basic settings of the function
        self.function: Parameter = Parameter(
            name="function",
            instrument=self,
            label=f"Function {channel} function",
            get_cmd=self._get_func,
            vals=vals.Strings(),
        )
        self.source: Parameter = Parameter(
            name="source",
            instrument=self,
            label=f"Function {channel} source",
            get_cmd=f"FUNC{channel}?",
        )

        # Trace settings
        self.points: Parameter = Parameter(
            name="points",
            instrument=self,
            label=f"Function {channel} points",
            get_cmd=self._get_points,
        )
        self.frequency_axis = DSOFrequencyAxisParam(
            name="frequency_axis",
            instrument=self,
            label="Frequency",
            unit="Hz",
            xorigin=0.0,
            xincrement=0.0,
            points=1,
            vals=vals.Arrays(shape=(self.points,)),
            snapshot_value=False,
        )
        self.time_axis = DSOTimeAxisParam(
            name="time_axis",
            instrument=self,
            label="Time",
            unit="s",
            xorigin=0.0,
            xincrement=0.0,
            points=1,
            vals=vals.Arrays(shape=(self.points,)),
            snapshot_value=False,
        )
        self.trace = DSOTraceParam(
            name="trace",
            instrument=self,
            label=f"Function {channel} trace",
            channel=self.channel_name,
            vals=vals.Arrays(shape=(self.points,)),
            snapshot_value=False,
        )

        # Measurement subsystem
        self.add_submodule(
            "measure", KeysightInfiniiumBoundMeasurement(self, "measure")
        )

    @property
    def channel(self) -> int:
        return self._channel

    @property
    def channel_name(self) -> str:
        return f"FUNC{self._channel}"

    def _get_points(self) -> int:
        """
        Return the number of points in the current function. This may be
        different to the number of points in the source as often functions
        modify the number of points.
        """
        self.write(f":WAV:SOUR {self.channel_name}")
        return int(self.ask(":WAV:POIN?"))

    def _get_func(self) -> str:
        """
        Return the function applied to the sources for this function
        """
        try:
            self.write(":SYST:HEAD ON")
            func, _ = self.ask(f":{self.channel_name}?").strip().split()
            match = re.fullmatch(f":{self.channel_name}:([\\w]+)", func)
            if match:
                return match.groups()[0]
            raise ValueError(
                f"Couldn't extract function for {self.channel_name}. Got {func}"
            )
        finally:
            self.write(":SYST:HEAD OFF")


InfiniiumFunction = KeysightInfiniiumFunction
"""
Alias for backwards compatibility
"""


class KeysightInfiniiumChannel(InstrumentChannel):
    def __init__(
        self,
        parent: "KeysightInfiniium",
        name: str,
        channel: int,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ):
        """
        Initialize an infiniium channel.
        """
        self._channel = channel

        super().__init__(parent, name, **kwargs)
        # display
        self.display: Parameter = Parameter(
            name="display",
            instrument=self,
            label=f"Channel {channel} display on/off",
            set_cmd=f"CHAN{channel}:DISP {{}}",
            get_cmd=f"CHAN{channel}:DISP?",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
        )

        # scaling
        self.offset: Parameter = Parameter(
            name="offset",
            instrument=self,
            label=f"Channel {channel} offset",
            set_cmd=f"CHAN{channel}:OFFS {{}}",
            unit="V",
            get_cmd=f"CHAN{channel}:OFFS?",
            get_parser=float,
        )
        self.range: Parameter = Parameter(
            name="range",
            instrument=self,
            label=f"Channel {channel} range",
            unit="V",
            set_cmd=f"CHAN{channel}:RANG {{}}",
            get_cmd=f"CHAN{channel}:RANG?",
            get_parser=float,
            vals=vals.Numbers(),
        )

        # Trigger level
        self.trigger_level: Parameter = Parameter(
            name="trigger_level",
            instrument=self,
            label=f"Channel {channel} trigger level",
            unit="V",
            set_cmd=f":TRIG:LEV CHAN{channel},{{}}",
            get_cmd=f":TRIG:LEV? CHAN{channel}",
            get_parser=float,
            vals=vals.Numbers(),
        )

        # Trace data
        self.time_axis = DSOTimeAxisParam(
            name="time_axis",
            instrument=self,
            label="Time",
            unit="s",
            xorigin=0.0,
            xincrement=0.0,
            points=1,
            vals=vals.Arrays(shape=(self.parent.acquire_points,)),
            snapshot_value=False,
        )
        self.trace = DSOTraceParam(
            name="trace",
            instrument=self,
            label=f"Channel {channel} trace",
            unit="V",
            channel=self.channel_name,
            vals=vals.Arrays(shape=(self.parent.acquire_points,)),
            snapshot_value=False,
        )

        # Measurement subsystem
        self.add_submodule(
            "measure", KeysightInfiniiumBoundMeasurement(self, "measure")
        )

    @property
    def channel(self) -> int:
        return self._channel

    @property
    def channel_name(self) -> str:
        return f"CHAN{self._channel}"

    def update_setpoints(self) -> None:
        """
        Update time axis and offsets for this channel.
        Calling this function is required when instr.cache_setpoints is True
        whenever the scope parameters are changed.
        """
        self.trace.update_setpoints()


InfiniiumChannel = KeysightInfiniiumChannel
"""
Alias for backwards compatibility
"""


class KeysightInfiniium(VisaInstrument):
    """
    This is the QCoDeS driver for the Keysight Infiniium oscilloscopes
    """

    default_timeout = 20
    default_terminator = "\n"

    def __init__(
        self,
        name: str,
        address: str,
        channels: int = 4,
        silence_pyvisapy_warning: bool = False,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ):
        """
        Initialises the oscilloscope.

        Args:
            name: Name of the instrument used by QCoDeS
            address: Instrument address as used by VISA
            timeout: Visa timeout, in secs.
            channels: The number of channels on the scope.
            silence_pyvisapy_warning: Don't warn about pyvisa-py at startup
            **kwargs: kwargs are forwarded to base class.
        """
        super().__init__(name, address, **kwargs)
        self.connect_message()

        # Check if we are using pyvisa-py as our visa lib and warn users that
        # this may cause long digitize operations to fail
        if (
            self.visa_handle.visalib.library_path == "py"
            and not silence_pyvisapy_warning
        ):
            self.log.warning(
                "Timeout not handled correctly in pyvisa_py. This may cause"
                " long acquisitions to fail. Either use ni/keysight visalib"
                " or set timeout to longer than longest expected acquisition"
                " time."
            )

        # switch the response header off else none of our parameters will work
        self.write(":SYSTem:HEADer OFF")

        # Then set up the data format used to retrieve waveforms
        self.write(":WAVEFORM:FORMAT WORD")
        self.write(":WAVEFORM:BYTEORDER LSBFirst")
        self.write(":WAVEFORM:STREAMING ON")

        # Query the oscilloscope parameters
        # Set sample rate, bandwidth and memory depth limits
        self._query_capabilities()
        # Number of channels can't be queried on most older scopes. Use a parameter
        # for now.
        self.no_channels = channels

        # Run state
        self.run_mode: Parameter = Parameter(
            name="run_mode",
            instrument=self,
            label="run mode",
            get_cmd=":RST?",
            vals=vals.Enum("RUN", "STOP", "SING"),
        )

        # Timing Parameters
        self.timebase_range: Parameter = Parameter(
            name="timebase_range",
            instrument=self,
            label="Range of the time axis",
            unit="s",
            get_cmd=":TIM:RANG?",
            set_cmd=":TIM:RANG {}",
            vals=vals.Numbers(5e-12, 20),
            get_parser=float,
        )
        self.timebase_position: Parameter = Parameter(
            name="timebase_position",
            instrument=self,
            label="Offset of the time axis",
            unit="s",
            get_cmd=":TIM:POS?",
            set_cmd=":TIM:POS {}",
            vals=vals.Numbers(),
            get_parser=float,
        )
        self.timebase_roll_enabled: Parameter = Parameter(
            name="timebase_roll_enabled",
            instrument=self,
            label="Is rolling mode enabled",
            get_cmd=":TIM:ROLL:ENABLE?",
            set_cmd=":TIM:ROLL:ENABLE {}",
            val_mapping={True: 1, False: 0},
        )

        # Trigger
        self.trigger_mode: Parameter = Parameter(
            name="trigger_mode",
            instrument=self,
            label="Trigger mode",
            get_cmd=":TRIG:MODE?",
        )
        self.trigger_sweep: Parameter = Parameter(
            name="trigger_sweep",
            instrument=self,
            label="Trigger sweep mode",
            get_cmd=":TRIG:SWE?",
            set_cmd=":TRIG:SWE {}",
            vals=vals.Enum("AUTO", "TRIG"),
        )
        self.trigger_state: Parameter = Parameter(
            name="trigger_state",
            instrument=self,
            label="Trigger state",
            get_cmd=":AST?",
            vals=vals.Enum("ARM", "TRIG", "ATRIG", "ADONE"),
            snapshot_value=False,
        )

        # Edge trigger parameters
        # Note that for now we only support parameterized edge triggers - this may
        # be something worth expanding.
        # To set trigger level, use the "trigger_level" parameter in each channel
        self.trigger_edge_source: Parameter = Parameter(
            name="trigger_edge_source",
            instrument=self,
            label="Source channel for the edge trigger",
            get_cmd=":TRIGger:EDGE:SOURce?",
            set_cmd=":TRIGger:EDGE:SOURce {}",
            vals=vals.Enum(
                *(
                    [f"CHAN{i}" for i in range(1, 4 + 1)]
                    + [f"DIG{i}" for i in range(16 + 1)]
                    + ["AUX", "LINE"]
                )
            ),
        )
        self.trigger_edge_slope: Parameter = Parameter(
            name="trigger_edge_slope",
            instrument=self,
            label="slope of the edge trigger",
            get_cmd=":TRIGger:EDGE:SLOPe?",
            set_cmd=":TRIGger:EDGE:SLOPe {}",
            vals=vals.Enum("POS", "POSITIVE", "NEG", "NEGATIVE", "EITH"),
        )
        self.trigger_level_aux: Parameter = Parameter(
            name="trigger_level_aux",
            instrument=self,
            label="Trigger level AUX",
            unit="V",
            get_cmd=":TRIGger:LEVel? AUX",
            set_cmd=":TRIGger:LEVel AUX,{}",
            get_parser=float,
            vals=vals.Numbers(),
        )

        # Aquisition
        # If sample points, rate and timebase_scale are set in an
        # incomensurate way, the scope only displays part of the waveform
        self.acquire_points: Parameter = Parameter(
            name="acquire_points",
            instrument=self,
            label="sample points",
            get_cmd=":ACQ:POIN?",
            set_cmd=":ACQ:POIN {}",
            get_parser=int,
            vals=vals.Numbers(min_value=self.min_pts, max_value=self.max_pts),
        )
        self.sample_rate: Parameter = Parameter(
            name="sample_rate",
            instrument=self,
            label="sample rate",
            get_cmd=":ACQ:SRAT?",
            set_cmd=":ACQ:SRAT {}",
            unit="Hz",
            get_parser=float,
            vals=vals.Numbers(min_value=self.min_srat, max_value=self.max_srat),
        )
        # Note: newer scopes allow a per-channel bandwidth. This is not implemented yet.
        self.bandwidth: Parameter = Parameter(
            name="bandwidth",
            instrument=self,
            label="bandwidth",
            get_cmd=":ACQ:BAND?",
            set_cmd=":ACQ:BAND {}",
            unit="Hz",
            get_parser=float,
            vals=vals.Numbers(min_value=self.min_bw, max_value=self.max_bw),
        )
        self.acquire_interpolate: Parameter = Parameter(
            name="acquire_interpolate",
            instrument=self,
            get_cmd=":ACQ:INTerpolate?",
            set_cmd=":ACQuire:INTerpolate {}",
            vals=vals.Enum(0, 1, "INT1", "INT2", "INT4", "INT8", "INT16", "INT32"),
        )
        self.acquire_mode: Parameter = Parameter(
            name="acquire_mode",
            instrument=self,
            label="Acquisition mode",
            get_cmd="ACQuire:MODE?",
            set_cmd="ACQuire:MODE {}",
            vals=vals.Enum(
                "ETIMe",
                "RTIMe",
                "PDETect",
                "HRESolution",
                "SEGMented",
                "SEGPdetect",
                "SEGHres",
            ),
        )
        self.average: Parameter = Parameter(
            name="average",
            instrument=self,
            label="Averages",
            get_cmd=self._get_avg,
            set_cmd=self._set_avg,
            vals=vals.Ints(min_value=1, max_value=10486575),
        )

        # Automatically digitize before acquiring a trace
        self.auto_digitize: Parameter = Parameter(
            name="auto_digitize",
            instrument=self,
            label="Auto digitize",
            set_cmd=None,
            get_cmd=None,
            val_mapping=create_on_off_val_mapping(),
            docstring=(
                "Digitize before each waveform download. "
                "If you need to acquire from multiple channels simultaneously "
                "or you wish to acquire with the scope running freely, "
                "set this value to False."
            ),
            initial_value=True,
        )
        self.cache_setpoints: Parameter = Parameter(
            name="cache_setpoints",
            instrument=self,
            label="Cache setpoints",
            set_cmd=None,
            get_cmd=None,
            val_mapping=create_on_off_val_mapping(),
            docstring=(
                "Cache setpoints. If false, the preamble is queried before each"
                " acquisition, which may add latency to measurements. If you"
                " are taking repeated measurements, set this to True and update"
                " setpoints manually by calling `instr.chX.update_setpoints()`."
            ),
            initial_value=False,
        )

        # Channels
        _channels = ChannelList(
            self, "channels", KeysightInfiniiumChannel, snapshotable=False
        )
        for i in range(1, self.no_channels + 1):
            channel = KeysightInfiniiumChannel(self, f"chan{i}", i)
            _channels.append(channel)
            self.add_submodule(f"ch{i}", channel)
        self.add_submodule("channels", _channels.to_channel_tuple())

        # Functions
        _functions = ChannelList(
            self, "functions", KeysightInfiniiumFunction, snapshotable=False
        )
        for i in range(1, 16 + 1):
            function = KeysightInfiniiumFunction(self, f"func{i}", i)
            _functions.append(function)
            self.add_submodule(f"func{i}", function)
        # Have to call channel list "funcs" here as functions is a
        # reserved name in Instrument.
        self.add_submodule("funcs", _functions.to_channel_tuple())

        # Submodules
        meassubsys = KeysightInfiniiumUnboundMeasurement(self, "measure")
        self.add_submodule("measure", meassubsys)

    def _query_capabilities(self) -> None:
        """
        Query scope capabilities (sample rate, bandwidth, memory depth)
        """
        try:
            # Bandwidth
            self.min_bw, self.max_bw = 0.0, 99.0e9  # Set default limits
            bw = self.ask(":ACQ:BAND:TESTLIMITS?")
            match = re.fullmatch(
                r"1,<numeric>([0-9.]+E\+[0-9]+):([0-9.]+E\+[0-9]+)", bw
            )
            if match:
                self.min_bw, self.max_bw = (float(f) for f in match.groups())
                self.log.info(f"Scope BW: {self.min_bw}-{self.max_bw}")
                self._meta_attrs.extend(("min_bw", "max_bw"))
            else:
                self.log.warning(
                    f"Unable to query bandwidth limits (inv. format ({bw})). "
                    f"Setting limits to default."
                )
        except VisaIOError as e:
            self.log.warning(
                f"Unable to query bandwidth limits ({e}). Setting limits to default."
            )

        # Memory depth
        try:
            self.min_pts, self.max_pts = 16, 1_000_000_000
            mem = self.ask(":ACQ:POIN:TESTLIMITS?")
            match = re.match("1,<numeric>([0-9]+):([0-9]+)", mem)
            if match:
                self.min_pts, self.max_pts = (int(p) for p in match.groups())
                self.log.info(f"Scope memory: {self.min_pts}-{self.max_pts}")
                self._meta_attrs.extend(("min_pts", "max_pts"))
            else:
                self.log.warning(
                    f"Unable to query memory depth (inv. format ({mem})). "
                    "Setting limits to default."
                )
        except VisaIOError as e:
            self.log.warning(
                f"Unable to query memory depth ({e}). Setting limits to default."
            )

        # Sample Rate
        try:
            # Set BW to auto in order to query this
            bw_set: Union[float, Literal["AUTO"]] = float(self.ask(":ACQ:BAND?"))
            if np.isclose(bw_set, self.max_bw):
                # Auto returns max bandwidth
                bw_set = "AUTO"
            self.write(":ACQ:BAND AUTO")
            self.min_srat, self.max_srat = 10.0, 99.0e9  # Set large limits
            srat = self.ask(":ACQ:SRAT:TESTLIMITS?")
            self.write(f":ACQ:BAND {bw_set}")
            match = re.fullmatch(
                r"1,<numeric>([0-9.]+E\+[0-9]+):([0-9.]+E\+[0-9]+)", srat
            )
            if match:
                self.min_srat, self.max_srat = (float(f) for f in match.groups())
                self.log.info(f"Scope sample rate: {self.min_srat}-{self.max_srat}")
                self._meta_attrs.extend(("min_srat", "max_srat"))
            else:
                self.log.warning(
                    f"Unable to query sample rate (inv. format ({srat})). "
                    "Setting limits to default."
                )
        except VisaIOError as e:
            self.log.warning(
                f"Unable to query sample rate ({e}). Setting limits to default."
            )

    def _get_avg(self) -> int:
        """
        Return the number of averages, or 1 if averaging is disabled.
        """
        enabled = int(self.ask(":ACQ:AVER?"))
        if not enabled:
            return 1
        else:
            return int(self.ask(":ACQ:AVER:COUN?"))

    def _set_avg(self, count: int) -> None:
        """
        Set the number of averages, or disable if 1.
        """
        if count == 1:
            self.write(":ACQ:AVER 0")
        else:
            self.write(f":ACQ:AVER:COUN {count}")
            self.write(":ACQ:AVER 1")

    # Simple oscilloscope commands
    def run(self) -> None:
        """
        Set the scope in run mode.
        """
        self.write(":RUN")
        self.run_mode()

    def stop(self) -> None:
        """
        Set the scope in stop mode.
        """
        self.write(":STOP")
        self.run_mode()

    def single(self) -> None:
        """
        Take a single acquisition
        """
        self.write(":SING")
        self.run_mode()

    def update_all_setpoints(self) -> None:
        """
        Update the setpoints for all enabled channels.
        This method may be run at the beginning of a measurement rather
        than looping through each channel manually.
        """
        for channel in self.channels:
            if channel.display():
                channel.update_setpoints()

    def digitize(self, timeout: Optional[int] = None) -> None:
        """
        Digitize a full waveform and block until the acquisition is complete.

        Warning: If using pyvisa_py as your visa library, this will not work with
        acquisitions longer than a single timeout period. If you require long
        acquisitions either use Keysight/NI Visa or set timeout to be longer than
        the expected acquisition time.
        """
        old_timeout = self.visa_handle.timeout
        if timeout is not None:
            self.visa_handle.timeout = timeout  # 1 second timeout
        try:
            self.visa_handle.write(":DIGITIZE;*OPC?")
            ret = None
            # Wait until we receive the "complete" reply
            while ret != "1":
                try:
                    ret = self.visa_handle.read()
                except VisaIOError as e:
                    # Ignore timeout errors - we could still be waiting for a trigger
                    # or taking a long acquisition
                    if e.error_code != StatusCode.error_timeout:
                        self.log.exception(
                            "Unexpected VisaError while waiting for acquisition."
                        )
                        raise  # Raise all other visa errors
        except KeyboardInterrupt:
            self.log.error(
                "Keyboard interrupt while waiting to digitize. Check your trigger?"
            )
            raise  # Pass error upwards
        finally:
            # Clear the device to unblock any failed digitize
            self.device_clear()
            if timeout is not None:
                self.visa_handle.timeout = old_timeout


Infiniium = KeysightInfiniium
"""
Alias for backwards compatibility
"""
