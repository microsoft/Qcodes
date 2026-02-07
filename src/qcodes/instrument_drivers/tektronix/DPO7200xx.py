"""
QCoDeS driver for the MSO/DPO5000/B, DPO7000/C,
DPO70000/B/C/D/DX/SX, DSA70000/B/C/D, and
MSO70000/C/DX Series Digital Oscilloscopes
"""

import textwrap
import time
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Generic

import numpy as np
import numpy.typing as npt

from qcodes.instrument import (
    ChannelList,
    Instrument,
    InstrumentBase,
    InstrumentBaseKWArgs,
    InstrumentChannel,
    VisaInstrument,
    VisaInstrumentKWArgs,
)
from qcodes.parameters import (
    Parameter,
    ParameterWithSetpoints,
    create_on_off_val_mapping,
)
from qcodes.parameters.parameter_base import ParameterDataTypeVar
from qcodes.validators import Arrays, Enum

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Unpack


def strip_quotes(string: str) -> str:
    """
    This function is used as a get_parser for various
    parameters in this driver
    """
    return string.strip('"')


class TektronixDPOModeError(Exception):
    """
    Raise this exception if we are in a wrong mode to
    perform an action
    """

    pass


ModeError = TektronixDPOModeError
"""
Alias for backwards compatibility
"""


class TektronixDPO7000xx(VisaInstrument):
    """
    QCoDeS driver for the MSO/DPO5000/B, DPO7000/C,
    DPO70000/B/C/D/DX/SX, DSA70000/B/C/D, and
    MSO70000/C/DX Series Digital Oscilloscopes
    """

    number_of_channels = 4
    number_of_measurements = 8  # The number of available
    # measurements does not change.
    default_terminator = "\n"

    def __init__(
        self, name: str, address: str, **kwargs: "Unpack[VisaInstrumentKWArgs]"
    ) -> None:
        super().__init__(name, address, **kwargs)

        self.horizontal: TektronixDPOHorizontal = self.add_submodule(
            "horizontal", TektronixDPOHorizontal(self, "horizontal")
        )
        """Instrument module horizontal"""
        self.data: TektronixDPOData = self.add_submodule(
            "data", TektronixDPOData(self, "data")
        )
        """Instrument module data"""
        self.waveform: TektronixDPOWaveformFormat = self.add_submodule(
            "waveform", TektronixDPOWaveformFormat(self, "waveform")
        )
        """Instrument module waveform"""
        self.trigger: TektronixDPOTrigger = self.add_submodule(
            "trigger", TektronixDPOTrigger(self, "trigger")
        )
        """Instrument module trigger"""
        self.delayed_trigger: TektronixDPOTrigger = self.add_submodule(
            "delayed_trigger",
            TektronixDPOTrigger(self, "delayed_trigger", delayed_trigger=True),
        )
        """Instrument module acquisition"""
        self.acquisition: TektronixDPOAcquisition = self.add_submodule(
            "acquisition", TektronixDPOAcquisition(self, "acquisition")
        )

        """Instrument module cursor"""
        self.cursor: TektronixDPOCursor = self.add_submodule(
            "cursor", TektronixDPOCursor(self, "cursor")
        )

        """Instrument module measure immediate"""
        self.measure_immediate: TektronixDPOMeasurementImmediate = self.add_submodule(
            "measure_immediate",
            TektronixDPOMeasurementImmediate(self, "measure_immediate"),
        )

        measurement_list = ChannelList(self, "measurement", TektronixDPOMeasurement)
        for measurement_number in range(1, self.number_of_measurements):
            measurement_name = f"measurement{measurement_number}"
            measurement_module = TektronixDPOMeasurement(
                self, measurement_name, measurement_number
            )

            self.add_submodule(measurement_name, measurement_module)
            measurement_list.append(measurement_module)

        self.measurement: ChannelList[TektronixDPOMeasurement] = self.add_submodule(
            "measurement", measurement_list
        )
        """Instrument module measurement"""
        self.statistics: TektronixDPOMeasurementStatistics = self.add_submodule(
            "statistics", TektronixDPOMeasurementStatistics(self, "statistics")
        )
        """Instrument module statistics"""

        channel_list = ChannelList(self, "channel", TektronixDPOChannel)
        for channel_number in range(1, self.number_of_channels + 1):
            channel_name = f"channel{channel_number}"
            channel_module = TektronixDPOChannel(
                self,
                channel_name,
                channel_number,
            )

            self.add_submodule(channel_name, channel_module)
            channel_list.append(channel_module)

        self.channel: ChannelList[TektronixDPOChannel] = self.add_submodule(
            "channel", channel_list
        )
        """Instrument module channel"""

        self.connect_message()

    def ask_raw(self, cmd: str) -> str:
        """
        Sometimes the instrument returns non-ascii characters in response
        strings manually adjust the encoding to latin-1
        """
        self.visa_log.debug(f"Querying: {cmd}")
        self.visa_handle.write(cmd)
        response = self.visa_handle.read(encoding="latin-1")
        self.visa_log.debug(f"Response: {response}")
        return response


class TektronixDPOData(InstrumentChannel):
    """
    This submodule sets and retrieves information regarding the
    data source for the "CURVE?" query, which is used when
    retrieving waveform data.
    """

    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        super().__init__(parent, name, **kwargs)
        # We can choose to retrieve data from arbitrary
        # start and stop indices of the buffer.
        self.start_index: Parameter = self.add_parameter(
            "start_index",
            get_cmd="DATa:STARt?",
            set_cmd="DATa:STARt {}",
            get_parser=int,
        )
        """Parameter start_index"""

        self.stop_index: Parameter = self.add_parameter(
            "stop_index", get_cmd="DATa:STOP?", set_cmd="DATa:STOP {}", get_parser=int
        )
        """Parameter stop_index"""

        self.source: Parameter = self.add_parameter(
            "source",
            get_cmd="DATa:SOU?",
            set_cmd="DATa:SOU {}",
            vals=Enum(*TektronixDPOWaveform.valid_identifiers),
        )
        """Parameter source"""

        self.encoding: Parameter = self.add_parameter(
            "encoding",
            get_cmd="DATa:ENCdg?",
            set_cmd="DATa:ENCdg {}",
            get_parser=strip_quotes,
            vals=Enum(
                "ASCIi",
                "FAStest",
                "RIBinary",
                "RPBinary",
                "FPBinary",
                "SRIbinary",
                "SRPbinary",
                "SFPbinary",
            ),
            docstring=textwrap.dedent(
                """
            For a detailed explanation of the
            set arguments, please consult the
            programmers manual at page 263/264.

            http://download.tek.com/manual/077001022.pdf
            """
            ),
        )
        """
        Parameter encoding

        For a detailed explanation of the
        set arguments, please consult the
        programmers manual at page 263/264.

        http://download.tek.com/manual/077001022.pdf
        """


class TektronixDPOWaveform(InstrumentChannel):
    """
    This submodule retrieves data from waveform sources, e.g.
    channels.
    """

    valid_identifiers: ClassVar[list[str]] = [
        f"{source_type}{i}"
        for source_type in ["CH", "MATH", "REF"]
        for i in range(1, TektronixDPO7000xx.number_of_channels + 1)
    ]

    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        identifier: str,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        super().__init__(parent, name, **kwargs)

        if identifier not in self.valid_identifiers:
            raise ValueError(
                f"Identifier {identifier} must be one of {self.valid_identifiers}"
            )

        self._identifier = identifier

        self.raw_data_offset: Parameter = self.add_parameter(
            "raw_data_offset",
            get_cmd=self._get_cmd("WFMOutPRE:YOFF?"),
            get_parser=float,
            docstring=textwrap.dedent(
                """
                Raw acquisition values range from min to max.
                For instance, for unsigned binary values of one
                byte, min=0 and max=255. The data offset specifies
                the center of this range
                """
            ),
        )
        """
        Raw acquisition values range from min to max.
        For instance, for unsigned binary values of one
        byte, min=0 and max=255. The data offset specifies
        the center of this range
        """

        self.x_unit: Parameter = self.add_parameter(
            "x_unit", get_cmd=self._get_cmd("WFMOutpre:XUNit?"), get_parser=strip_quotes
        )
        """Parameter x_unit"""

        self.x_increment: Parameter = self.add_parameter(
            "x_increment",
            get_cmd=self._get_cmd("WFMOutPRE:XINCR?"),
            unit=self.x_unit(),
            get_parser=float,
        )
        """Parameter x_increment"""

        self.y_unit: Parameter = self.add_parameter(
            "y_unit", get_cmd=self._get_cmd("WFMOutpre:YUNit?"), get_parser=strip_quotes
        )
        """Parameter y_unit"""

        self.offset: Parameter = self.add_parameter(
            "offset",
            get_cmd=self._get_cmd("WFMOutPRE:YZERO?"),
            get_parser=float,
            unit=self.y_unit(),
        )
        """Parameter offset"""

        self.scale: Parameter = self.add_parameter(
            "scale",
            get_cmd=self._get_cmd("WFMOutPRE:YMULT?"),
            get_parser=float,
            unit=self.y_unit(),
        )
        """Parameter scale"""

        self.length: Parameter = self.add_parameter(
            "length", get_cmd=self._get_cmd("WFMOutpre:NR_Pt?"), get_parser=int
        )
        """Parameter length"""

        hor_unit = self.x_unit()
        hor_label = "Time" if hor_unit == "s" else "Frequency"

        self.trace_axis: Parameter = self.add_parameter(
            "trace_axis",
            label=hor_label,
            get_cmd=self._get_trace_setpoints,
            vals=Arrays(shape=(self.length,)),
            unit=hor_unit,
        )
        """Parameter trace_axis"""

        ver_unit = self.y_unit()
        ver_label = "Voltage" if ver_unit == "s" else "Amplitude"

        self.trace: ParameterWithSetpoints = self.add_parameter(
            "trace",
            label=ver_label,
            get_cmd=self._get_trace_data,
            vals=Arrays(shape=(self.length,)),
            unit=ver_unit,
            setpoints=(self.trace_axis,),
            parameter_class=ParameterWithSetpoints,
        )
        """Parameter trace"""

    def _get_cmd(self, cmd_string: str) -> "Callable[[], str]":
        """
        Parameters defined in this submodule require the correct
        data source being selected first.
        """

        def inner() -> str:
            self.root_instrument.data.source(self._identifier)
            return self.ask(cmd_string)

        return inner

    def _get_trace_data(self) -> npt.NDArray:
        self.root_instrument.data.source(self._identifier)
        waveform = self.root_instrument.waveform

        if not waveform.is_binary():
            raw_data = self.root_instrument.visa_handle.query_ascii_values(
                "CURVE?", container=np.array
            )
        else:
            bytes_per_sample = waveform.bytes_per_sample()
            data_type = {1: "b", 2: "h", 4: "f", 8: "d"}[bytes_per_sample]

            if waveform.data_format() == "unsigned_integer":
                data_type = data_type.upper()

            is_big_endian = waveform.is_big_endian()

            raw_data = self.root_instrument.visa_handle.query_binary_values(
                "CURVE?",
                datatype=data_type,
                is_big_endian=is_big_endian,
                container=np.array,
            )

        return (raw_data - self.raw_data_offset()) * self.scale() + self.offset()

    def _get_trace_setpoints(self) -> npt.NDArray:
        """
        Infer the set points of the waveform
        """
        sample_count = self.length()
        x_increment = self.x_increment()
        return np.linspace(0, x_increment * sample_count, sample_count)


class TektronixDPOWaveformFormat(InstrumentChannel):
    """
    With this sub module we can query waveform
    formatting data. Please note that parameters
    defined in this submodule effects all
    waveform sources, whereas parameters defined in the
    submodule 'TekronixDPOWaveform' apply to
    specific waveform sources (e.g. channel1 or math2)
    """

    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        super().__init__(parent, name, **kwargs)

        self.data_format: Parameter = self.add_parameter(
            "data_format",
            get_cmd="WFMOutpre:BN_Fmt?",
            set_cmd="WFMOutpre:BN_Fmt {}",
            val_mapping={
                "signed_integer": "RI",
                "unsigned_integer": "RP",
                "floating_point": "FP",
            },
        )
        """Parameter data_format"""

        self.is_big_endian: Parameter = self.add_parameter(
            "is_big_endian",
            get_cmd="WFMOutpre:BYT_Or?",
            set_cmd="WFMOutpre:BYT_Or {}",
            val_mapping={False: "LSB", True: "MSB"},
        )
        """Parameter is_big_endian"""

        self.bytes_per_sample: Parameter = self.add_parameter(
            "bytes_per_sample",
            get_cmd="WFMOutpre:BYT_Nr?",
            set_cmd="WFMOutpre:BYT_Nr {}",
            get_parser=int,
            vals=Enum(1, 2, 4, 8),
        )
        """Parameter bytes_per_sample"""

        self.is_binary: Parameter = self.add_parameter(
            "is_binary",
            get_cmd="WFMOutpre:ENCdg?",
            set_cmd="WFMOutpre:ENCdg {}",
            val_mapping={True: "BINARY", False: "ASCII"},
        )
        """Parameter is_binary"""


class TektronixDPOChannel(InstrumentChannel):
    """
    The main channel module for the oscilloscope. The parameters
    defined here reflect the waveforms as they are displayed on
    the instrument display.
    """

    def __init__(
        self,
        parent: Instrument | InstrumentChannel,
        name: str,
        channel_number: int,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        super().__init__(parent, name, **kwargs)
        self._identifier = f"CH{channel_number}"

        self.waveform: TektronixDPOWaveform = self.add_submodule(
            "waveform", TektronixDPOWaveform(self, "waveform", self._identifier)
        )
        """Instrument module waveform"""

        self.coupling: Parameter = self.add_parameter(
            "coupling",
            get_cmd=f"{self._identifier}:COUPling?",
            set_cmd=f"{self._identifier}:COUPling {{}}",
            vals=Enum("AC", "DC", "DCREJECT" "GND"),
            get_parser=str,
        )
        """Parameter coupling: 'AC', 'DC', 'DCREJECT', 'GND'"""

        self.scale: Parameter = self.add_parameter(
            "scale",
            get_cmd=f"{self._identifier}:SCA?",
            set_cmd=f"{self._identifier}:SCA {{}}",
            get_parser=float,
            unit="V/div",
        )
        """Parameter scale V/div"""

        self.offset: Parameter = self.add_parameter(
            "offset",
            get_cmd=f"{self._identifier}:OFFS?",
            set_cmd=f"{self._identifier}:OFFS {{}}",
            get_parser=float,
            unit="V",
        )
        """Parameter offset voltage"""

        self.position: Parameter = self.add_parameter(
            "position",
            get_cmd=f"{self._identifier}:POS?",
            set_cmd=f"{self._identifier}:POS {{}}",
            get_parser=float,
            unit="div",
        )
        """Parameter position [-8, 8] divisions"""

        self.termination: Parameter = self.add_parameter(
            "termination",
            get_cmd=f"{self._identifier}:TER?",
            set_cmd=f"{self._identifier}:TER {{}}",
            vals=Enum(50, 1e6),
            get_parser=float,
            unit="Ohm",
        )
        """Parameter termination"""

        self.analog_to_digital_threshold: Parameter = self.add_parameter(
            "analog_to_digital_threshold",
            get_cmd=f"{self._identifier}:THRESH?",
            set_cmd=f"{self._identifier}:THRESH {{}}",
            get_parser=float,
            unit="V",
        )
        """Parameter analog_to_digital_threshold"""

        self.termination_voltage: Parameter = self.add_parameter(
            "termination_voltage",
            get_cmd=f"{self._identifier}:VTERm:BIAS?",
            set_cmd=f"{self._identifier}:VTERm:BIAS {{}}",
            get_parser=float,
            unit="V",
        )
        """Parameter termination_voltage"""

    def set_trace_length(self, value: int) -> None:
        """
        Set the trace length when retrieving data
        through the 'waveform' interface

        Args:
            value: The requested number of samples in the trace

        """
        if self.root_instrument.horizontal.record_length() < value:
            raise ValueError(
                "Cannot set a trace length which is larger than "
                "the record length. Please switch to manual mode "
                "and adjust the record length first"
            )

        self.root_instrument.data.start_index(1)
        self.root_instrument.data.stop_index(value)

    def set_trace_time(self, value: float) -> None:
        """
        Args:
            value: The time over which a trace is desired.

        """
        sample_rate = self.root_instrument.horizontal.sample_rate()
        required_sample_count = int(sample_rate * value)
        self.set_trace_length(required_sample_count)


class TektronixDPOHorizontal(InstrumentChannel):
    """
    This module controls the horizontal axis of the scope
    """

    def __init__(
        self,
        parent: Instrument | InstrumentChannel,
        name: str,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        super().__init__(parent, name, **kwargs)

        self.mode: Parameter = self.add_parameter(
            "mode",
            get_cmd="HORizontal:MODE?",
            set_cmd="HORizontal:MODE {}",
            vals=Enum("auto", "constant", "manual"),
            get_parser=str.lower,
            docstring="""
            Auto mode attempts to keep record length
            constant as you change the time per division
            setting. Record length is read only.

            Constant mode attempts to keep sample rate
            constant as you change the time per division
            setting. Record length is read only.

            Manual mode lets you change sample mode and
            record length. Time per division or Horizontal
            scale is read only.
            """,
        )
        """
            Auto mode attempts to keep record length
            constant as you change the time per division
            setting. Record length is read only.

            Constant mode attempts to keep sample rate
            constant as you change the time per division
            setting. Record length is read only.

            Manual mode lets you change sample mode and
            record length. Time per division or Horizontal
            scale is read only.
            """

        self.unit: Parameter = self.add_parameter(
            "unit", get_cmd="HORizontal:MAIn:UNIts?", get_parser=strip_quotes
        )
        """Parameter unit"""

        self.record_length: Parameter = self.add_parameter(
            "record_length",
            get_cmd="HORizontal:MODE:RECOrdlength?",
            set_cmd=self._set_record_length,
            get_parser=float,
        )
        """Parameter record_length"""

        self.sample_rate: Parameter = self.add_parameter(
            "sample_rate",
            get_cmd="HORizontal:MODE:SAMPLERate?",
            set_cmd="HORizontal:MODE:SAMPLERate {}",
            get_parser=float,
            unit=f"sample/{self.unit()}",
        )
        """Parameter sample_rate"""

        self.scale: Parameter = self.add_parameter(
            "scale",
            get_cmd="HORizontal:MODE:SCAle?",
            set_cmd=self._set_scale,
            get_parser=float,
            unit=f"{self.unit()}/div",
        )
        """Parameter scale"""

        self.position: Parameter = self.add_parameter(
            "position",
            get_cmd="HORizontal:POSition?",
            set_cmd="HORizontal:POSition {}",
            get_parser=float,
            unit="%",
            docstring=textwrap.dedent(
                """
            The horizontal position relative to a
            received trigger. E.g. a value of '10'
            sets the trigger position of the waveform
            such that 10% of the display is to the
            left of the trigger position.
            """
            ),
        )
        """
        The horizontal position relative to a
        received trigger. E.g. a value of '10'
        sets the trigger position of the waveform
        such that 10% of the display is to the
        left of the trigger position.
        """

        self.roll: Parameter = self.add_parameter(
            "roll",
            get_cmd="HORizontal:ROLL?",
            set_cmd="HORizontal:ROLL {}",
            vals=Enum("Auto", "On", "Off"),
            docstring=textwrap.dedent(
                """
            Use Roll Mode when you want to view data at
            very slow sweep speeds.
            """
            ),
        )
        """
        Use Roll Mode when you want to view data at
        very slow sweep speeds.
        """

    def _set_record_length(self, value: int) -> None:
        if self.mode() != "manual":
            raise TektronixDPOModeError(
                "The record length can only be changed in manual mode"
            )

        self.write(f"HORizontal:MODE:RECOrdlength {value}")

    def _set_scale(self, value: float) -> None:
        if self.mode() == "manual":
            raise TektronixDPOModeError("The scale cannot be changed in manual mode")

        self.write(f"HORizontal:MODE:SCAle {value}")


class TektronixDPOAcquisition(InstrumentChannel):
    """
    This submodule controls the acquisition mode of the
    oscilloscope. It is used to set the acquisition mode
    and the number of acquisitions.
    """

    def __init__(
        self,
        parent: Instrument,
        name: str,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        super().__init__(parent, name, **kwargs)

        self.mode: Parameter = self.add_parameter(
            "mode",
            get_cmd="ACQuire:MODe?",
            set_cmd="ACQuire:MODe {}",
            vals=Enum(
                "sample",
                "peakdetect",
                "average",
                "high_res",
                "average",
                "wfmdb",
                "envelope",
            ),
            get_parser=str.lower,
        )
        """Parameter mode"""

        self.state: Parameter = self.add_parameter(
            "state",
            get_cmd="ACQuire:STATE?",
            set_cmd=f"ACQuire:STATE {{}}",
            vals=Enum(
                "ON",
                "OFF",
                "RUN",
                "STOP",
            ),
            get_parser=str.lower,
        )
        """This command starts or stops acquisitions. When state is set to ON or RUN, a
        new acquisition will be started. If the last acquisition was a single acquisition
        sequence, a new single sequence acquisition will be started. If the last acquisition
        was continuous, a new continuous acquisition will be started.
        
        Args:
            state: 'ON', 'OFF', 'RUN', or 'STOP'
        """

        self.stop_after: Parameter = self.add_parameter(
            "stop_after",
            get_cmd="ACQuire:STOPAfter?",
            set_cmd=f"ACQuire:STOPAfter {{}}",
            vals=Enum("SEQUENCE", "RUNSTOP"),
            get_parser=str.lower,
        )
        """This command sets or queries whether the instrument continually acquires
        acquisitions or acquires a single sequence. Pressing SINGLE on the front
        panel button is equivalent to sending these commands: ACQUIRE:STOPAFTER
        SEQUENCE and ACQUIRE:STATE 1."""


class TektronixDPOTrigger(InstrumentChannel):
    """
    Submodule for trigger setup.

    You can trigger with the A (Main) trigger system alone
    or combine the A (Main) trigger with the B (Delayed) trigger
    to trigger on sequential events. When using sequential
    triggering, the A trigger event arms the trigger system, and
    the B trigger event triggers the instrument when the B
    trigger conditions are met.

    A and B triggers can (and typically do) have separate sources.
    The B trigger condition is based on a time delay or a specified
    number of events.

    See page75, Using A (Main) and B (Delayed) triggers.
    https://download.tek.com/manual/MSO70000C-DX-DPO70000C-DX-MSO-DPO7000C-MSO-DPO5000B-Oscilloscope-Quick-Start-User-Manual-071298006.pdf
    """

    def __init__(
        self,
        parent: Instrument,
        name: str,
        delayed_trigger: bool = False,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ):
        super().__init__(parent, name, **kwargs)
        self._identifier = "B" if delayed_trigger else "A"

        trigger_types = ["EDGE", "edge", "logic", "pulse"]
        if self._identifier == "A":
            trigger_types.extend(
                ["video", "i2c", "can", "spi", "communication", "serial", "rs232"]
            )

        self.ready: Parameter = self.add_parameter(
            "ready",
            get_cmd=f"TRIGger:{self._identifier}:READY?",
            get_parser=str.lower,
        )
        """Indicates whether the trigger system is ready to accept a trigger.
        A value of 1 indicates that the trigger system is ready to accept a trigger.
        A value of 0 indicates that the trigger system is not ready to accept a trigger.
        """

        self.state: Parameter = self.add_parameter(
            "state",
            get_cmd="TRIGger:STATe?",
            get_parser=str.lower,
        )
        """Gets the current Trigger state:

            ARMED indicates that the instrument is acquiring pretrigger information.

            AUTO indicates that the instrument is in the automatic mode and acquires data
            even in the absence of a trigger.

            DPO indicates that the instrument is in DPO mode.

            PARTIAL indicates that the A trigger has occurred and the instrument is waiting
            for the B trigger to occur.

            READY indicates that all pretrigger information is acquired and that the instrument
            is ready to accept a trigger.

            SAVE indicates that the instrument is in save mode and is not acquiring data.

            TRIGGER indicates that the instrument triggered and is acquiring the post trigger
            information.
        """

        self.type: Parameter = self.add_parameter(
            "type",
            get_cmd=f"TRIGger:{self._identifier}:TYPE?",
            set_cmd=self._trigger_type,
            vals=Enum(*trigger_types),
            get_parser=str.lower,
        )
        """Trigger type"""

        edge_couplings = ["ac", "dc", "hfrej", "lfrej", "noiserej"]
        if self._identifier == "B":
            edge_couplings.append("atrigger")

        self.edge_coupling: Parameter = self.add_parameter(
            "edge_coupling",
            get_cmd=f"TRIGger:{self._identifier}:EDGE:COUPling?",
            set_cmd=f"TRIGger:{self._identifier}:EDGE:COUPling {{}}",
            vals=Enum(*edge_couplings),
            get_parser=str.lower,
        )
        """Trigger edge coupling: 'ac', 'dc', 'hfrej', 'lfrej', 'noiserej', 'atrigger'"""

        self.edge_slope: Parameter = self.add_parameter(
            "edge_slope",
            get_cmd=f"TRIGger:{self._identifier}:EDGE:SLOpe?",
            set_cmd=f"TRIGger:{self._identifier}:EDGE:SLOpe {{}}",
            vals=Enum("RISE", "rise", "FALL", "fall", "EITHER", "either"),
            get_parser=str.lower,
        )
        """Trigger edge slope: 'rise', 'fall', or 'either'"""

        trigger_sources = [
            f"CH{i}" for i in range(1, TektronixDPO7000xx.number_of_channels)
        ]

        trigger_sources.extend([f"D{i}" for i in range(0, 16)])

        if self._identifier == "A":
            trigger_sources.append("line")

        trigger_sources.append("AUX")

        self.source: Parameter = self.add_parameter(
            "source",
            get_cmd=f"TRIGger:{self._identifier}:EDGE:SOUrce?",
            set_cmd=f"TRIGger:{self._identifier}:EDGE:SOUrce {{}}",
            vals=Enum(*trigger_sources),
        )
        """Trigger source: 'CH1', 'CH2', ..., 'CH4', 'D0', 'D1', ..., 'D15', 'AUX', 'LINE'"""

        self.level: Parameter = self.add_parameter(
            "level",
            get_cmd=f"TRIGger:{self._identifier}:LEVel?",
            set_cmd=f"TRIGger:{self._identifier}:LEVel {{}}",
            get_parser=float,
            unit="V",
        )
        """Trigger level: The voltage level at which the trigger condition is met."""

    def _trigger_type(self, value: str) -> None:
        if value.lower() != "edge":
            raise NotImplementedError(
                "We currently only support the 'edge' trigger type"
            )
        self.write(f"TRIGger:{self._identifier}:TYPE {value}")


class TektronixDPOMeasurementParameter(
    Parameter[ParameterDataTypeVar, "TektronixDPOMeasurement"],
    Generic[ParameterDataTypeVar],
):
    """
    A measurement parameter does not only return the instantaneous value
    of a measurement, but can also return some statistics. The accumulation
    time over which these statistics are gathered can be controlled through
    the 'time_constant' parameter on the submodule
    'TektronixDPOMeasurementStatistics'. Here we also find the method 'reset'
    to reset the values over which the statistics are gathered.
    """

    def _get(self, metric: str) -> float:
        measurement_channel = self.instrument
        if measurement_channel.type.get_latest() != self.name:
            measurement_channel.type(self.name)

        measurement_channel.state(1)
        measurement_channel.wait_adjustment_time()
        measurement_number = measurement_channel.measurement_number

        str_value = measurement_channel.ask(
            f"MEASUrement:MEAS{measurement_number}:{metric}?"
        )

        return float(str_value)

    def mean(self) -> float:
        return self._get("MEAN")

    def max(self) -> float:
        return self._get("MAX")

    def min(self) -> float:
        return self._get("MINI")

    def stdev(self) -> float:
        return self._get("STDdev")

    def get_raw(self) -> float:
        return self._get("VALue")

    def set_raw(self, value: Any) -> None:
        raise ValueError("A measurement cannot be set")


class TektronixDPOMeasurement(InstrumentChannel):
    """
    The measurement submodule
    """

    # It was found by trial and error that adjusting
    # the measurement type and source takes some time
    # to reflect properly on the value of the
    # measurement. Wait a minimum of ...
    _minimum_adjustment_time = 0.1
    # seconds after setting measurement type/source before
    # calling the measurement value SCPI command.

    measurements: ClassVar[list[tuple[str, str]]] = [
        ("amplitude", "V"),
        ("area", "Vs"),
        ("burst", "s"),
        ("carea", "Vs"),
        ("cmean", "V"),
        ("crms", "V"),
        ("delay", "s"),
        ("distduty", "%"),
        ("extinctdb", "dB"),
        ("extinctpct", "%"),
        ("extinctratio", ""),
        ("eyeheight", "V"),
        ("eyewidth", "s"),
        ("fall", "s"),
        ("frequency", "Hz"),
        ("high", "V"),
        ("hits", "hits"),
        ("low", "V"),
        ("maximum", "V"),
        ("mean", "V"),
        ("median", "V"),
        ("minimum", "V"),
        ("ncross", "s"),
        ("nduty", "%"),
        ("novershoot", "%"),
        ("nwidth", "s"),
        ("pbase", "V"),
        ("pcross", "s"),
        ("pctcross", "%"),
        ("pduty", "%"),
        ("peakhits", "hits"),
        ("period", "s"),
        ("phase", "°"),
        ("pk2pk", "V"),
        ("pkpkjitter", "s"),
        ("pkpknoise", "V"),
        ("povershoot", "%"),
        ("ptop", "V"),
        ("pwidth", "s"),
        ("qfactor", ""),
        ("rise", "s"),
        ("rms", "V"),
        ("rmsjitter", "s"),
        ("rmsnoise", "V"),
        ("sigma1", "%"),
        ("sigma2", "%"),
        ("sigma3", "%"),
        ("sixsigmajit", "s"),
        ("snratio", ""),
        ("stddev", "V"),
        ("undefined", ""),
        ("waveforms", "wfms"),
    ]

    def __init__(
        self,
        parent: Instrument,
        name: str,
        measurement_number: int,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        super().__init__(parent, name, **kwargs)
        self._measurement_number = measurement_number
        self._adjustment_time = time.perf_counter()

        self.state: Parameter = self.add_parameter(
            "state",
            get_cmd=f"MEASUrement:MEAS{self._measurement_number}:STATe?",
            set_cmd=f"MEASUrement:MEAS{self._measurement_number}:STATe {{}}",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0"),
        )
        """Parameter state"""

        self.type: Parameter = self.add_parameter(
            "type",
            get_cmd=f"MEASUrement:MEAS{self._measurement_number}:TYPe?",
            set_cmd=self._set_measurement_type,
            get_parser=str.lower,
            vals=Enum(*(m[0] for m in self.measurements)),
            docstring=textwrap.dedent(
                "Please see page 566-569 of the programmers manual "
                "for a detailed description of these arguments. "
                "http://download.tek.com/manual/077001022.pdf"
            ),
        )
        """Parameter type"""

        for measurement, unit in self.measurements:
            self.add_parameter(
                name=measurement,
                unit=unit,
                parameter_class=TektronixDPOMeasurementParameter,
            )

        for src in [1, 2]:
            self.add_parameter(
                f"source{src}",
                get_cmd=f"MEASUrement:MEAS{self._measurement_number}:SOUrce{src}?",
                set_cmd=partial(self._set_source, src),
                vals=Enum(*([*TektronixDPOWaveform.valid_identifiers, "HISTogram"])),
            )

    @property
    def measurement_number(self) -> int:
        return self._measurement_number

    def _set_measurement_type(self, value: str) -> None:
        self._adjustment_time = time.perf_counter()
        self.write(f"MEASUrement:MEAS{self._measurement_number}:TYPe {value}")

    def _set_source(self, source_number: int, value: str) -> None:
        self._adjustment_time = time.perf_counter()
        self.write(
            f"MEASUrement:MEAS{self._measurement_number}:SOUrce{source_number} {value}"
        )

    def wait_adjustment_time(self) -> None:
        """
        Wait until the minimum time after adjusting the measurement source or
        type has elapsed
        """
        time_since_adjust = time.perf_counter() - self._adjustment_time
        if time_since_adjust < self._minimum_adjustment_time:
            time_remaining = self._minimum_adjustment_time - time_since_adjust
            time.sleep(time_remaining)


class TektronixDPOMeasurementStatistics(InstrumentChannel):
    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ):
        super().__init__(parent=parent, name=name, **kwargs)

        self.mode: Parameter = self.add_parameter(
            "mode",
            get_cmd="MEASUrement:STATIstics:MODe?",
            set_cmd="MEASUrement:STATIstics:MODe {}",
            vals=Enum("OFF", "ALL", "VALUEMean", "MINMax", "MEANSTDdev"),
            docstring=textwrap.dedent(
                "This command controls the operation and display of measurement "
                "statistics. "
                "1. OFF turns off all measurements. This is the default value "
                "2. ALL turns on statistics and displays all statistics for "
                "each measurement. "
                "3. VALUEMean turns on statistics and displays the value and the "
                "mean (μ) of each measurement. "
                "4. MINMax turns on statistics and displays the min and max of "
                "each measurement. "
                "5. MEANSTDdev turns on statistics and displays the mean and "
                "standard deviation of each measurement."
            ),
        )
        """Parameter mode"""

        self.time_constant: Parameter = self.add_parameter(
            "time_constant",
            get_cmd="MEASUrement:STATIstics:WEIghting?",
            set_cmd="MEASUrement:STATIstics:WEIghting {}",
            get_parser=int,
            docstring=textwrap.dedent(
                "This command sets or queries the time constant for mean and "
                "standard deviation statistical accumulations, which is equivalent "
                "to selecting Measurement Setup from the Measure menu, clicking "
                "the Statistics button and entering the desired Weight n= value."
            ),
        )
        """Parameter time_constant"""

    def reset(self) -> None:
        self.write("MEASUrement:STATIstics:COUNt RESEt")


class TektronixDPOMeasurementImmediate(InstrumentChannel):
    """
    The cursor submodule allows you to set and retrieve
    information regarding the cursor type, state, and
    positions. The cursor can be used to measure
    voltage and time differences between two points on
    the waveform display.

    Methods:
        - function: Set or get the cursor type (e.g., horizontal bars, vertical bars, etc.)
        - state: Set or get the cursor state (ON or OFF)
        - x1: Set or get the x1 position of the cursor (in seconds)
        - x2: Set or get the x2 position of the cursor (in seconds)
        - y1: Set or get the y1 position of the cursor (in Volts)
        - y2: Set or get the y2 position of the cursor (in Volts)
    """

    def __init__(
        self,
        parent: Instrument,
        name: str,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        super().__init__(parent, name, **kwargs)

        self.gating: Parameter = self.add_parameter(
            "gating",
            get_cmd="MEASUrement:GATing?",
            set_cmd="MEASUrement:GATing {}",
            vals=Enum("ON", "OFF", "ZOOM1", "ZOOM2", "ZOOM3", "ZOOM4", "CURSOR"),
        )
        self.source1: Parameter = self.add_parameter(
            "source1",
            get_cmd="MEASUrement:IMMed:SOUrce1?",
            set_cmd="MEASUrement:IMMed:SOUrce1 {}",
            vals=Enum(*TektronixDPOWaveform.valid_identifiers),
        )

        self.source2: Parameter = self.add_parameter(
            "source2",
            get_cmd="MEASUrement:IMMed:SOUrce2?",
            set_cmd="MEASUrement:IMMed:SOUrce2 {}",
            vals=Enum(*TektronixDPOWaveform.valid_identifiers),
        )

        self.type: Parameter = self.add_parameter(
            "type",
            get_cmd="MEASUrement:IMMed:TYPE?",
            set_cmd="MEASUrement:IMMed:TYPE {}",
            vals=Enum(
                "MEAN",
            ),
            get_parser=str.lower,
        )
        """Cursor Type [OFF, HBARS, VBARS, SCREEN, WAVEFORM]"""

        self.units: Parameter = self.add_parameter(
            "units",
            get_cmd="MEASUrement:IMMed:UNITS?",
            get_parser=strip_quotes,
        )

        self.value: Parameter = self.add_parameter(
            "value",
            get_cmd="MEASUrement:IMMed:VALue?",
            get_parser=float,
        )


class TektronixDPOCursor(InstrumentChannel):
    """
    The cursor submodule allows you to set and retrieve
    information regarding the cursor type, state, and
    positions. The cursor can be used to measure
    voltage and time differences between two points on
    the waveform display.

    Methods:
        - function: Set or get the cursor type (e.g., horizontal bars, vertical bars, etc.)
        - state: Set or get the cursor state (ON or OFF)
        - x1: Set or get the x1 position of the cursor (in seconds)
        - x2: Set or get the x2 position of the cursor (in seconds)
        - y1: Set or get the y1 position of the cursor (in Volts)
        - y2: Set or get the y2 position of the cursor (in Volts)
    """

    def __init__(
        self,
        parent: Instrument,
        name: str,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        super().__init__(parent, name, **kwargs)

        self.function: Parameter = self.add_parameter(
            "function",
            get_cmd="CURSOR:FUNCtion?",
            set_cmd="CURSOR:FUNCtion {}",
            vals=Enum(
                "OFF",
                "HBARS",
                "VBARS",
                "SCREEN",
                "WAVEFORM",
            ),
            get_parser=str.lower,
        )
        """Cursor Type [OFF, HBARS, VBARS, SCREEN, WAVEFORM]"""

        self.state: Parameter = self.add_parameter(
            "state",
            get_cmd="CURSOR:STATE?",
            set_cmd="CURSOR:STATE {}",
            vals=Enum("ON", "OFF"),
            get_parser=str.lower,
        )
        """Cursor state [ON, OFF]"""

        self.x1: Parameter = self.add_parameter(
            "x1",
            get_cmd="CURSOR:VBARS:POSITION1?",
            set_cmd="CURSOR:VBARS:POSITION1 {}",
            get_parser=float,
            unit="s",
        )
        """Cursor x1 position in seconds"""

        self.x2: Parameter = self.add_parameter(
            "x2",
            get_cmd="CURSOR:VBARS:POSITION2?",
            set_cmd="CURSOR:VBARS:POSITION2 {}",
            get_parser=float,
            unit="s",
        )
        """Cursor x2 position in seconds"""

        self.y1: Parameter = self.add_parameter(
            "y1",
            get_cmd="CURSOR:HBARS:POSITION1?",
            set_cmd="CURSOR:HBARS:POSITION1 {}",
            get_parser=float,
            unit="V",
        )
        """Cursor y1 position in Volts"""

        self.y2: Parameter = self.add_parameter(
            "y2",
            get_cmd="CURSOR:HBARS:POSITION2?",
            set_cmd="CURSOR:HBARS:POSITION2 {}",
            get_parser=float,
            unit="V",
        )
        """Cursor y2 position in Volts"""
