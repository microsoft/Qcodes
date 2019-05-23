"""
QCoDeS driver for the MSO/DPO5000/B, DPO7000/C,
DPO70000/B/C/D/DX/SX, DSA70000/B/C/D, and
MSO70000/C/DX Series Digital Oscilloscopes
"""
import numpy as np
from typing import cast, Any
from functools import partial
import time

from qcodes import (
    Instrument, VisaInstrument, InstrumentChannel, ParameterWithSetpoints,
    ChannelList, Parameter
)

from qcodes.utils.validators import Enum, Arrays


def strip_quotes(string: str) -> str:
    """
    This function is used as a get_parser for various
    parameters in this driver
    """
    return string.strip('"')


class ModeError(Exception):
    """
    Raise this exception if we are in a wrong mode to
    perform an action
    """
    pass


class _TektronixDPOData(InstrumentChannel):
    """
    This is meant to be a private class, only to be used
    by the TektronixDPO7000xx driver class. The end user
    is not intended to call parameters on this submodule
    directly.
    """

    def __init__(self, parent: Instrument, name: str) -> None:
        super().__init__(parent, name)

        self.add_parameter(
            "start",
            get_cmd="DATa:STARt?",
            set_cmd="DATa:STARt {}",
            get_parser=int
        )

        self.add_parameter(
            "stop",
            get_cmd="DATa:STOP?",
            set_cmd="DATa:STOP {}",
            get_parser=int
        )

        self.add_parameter(
            "source",
            get_cmd="DATa:SOU?",
            set_cmd="DATa:SOU {}",
            vals=Enum(*[
                f"{name}{i}"
                for name in TektronixDPOChannel.channel_types
                for i in range(1, TektronixDPO7000xx.channel_count + 1)
            ])
        )

        self.add_parameter(
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
            docstring="""
            For a detailed explanation of the 
            set arguments, please consult the 
            programmers manual at page 263/264. 

            http://download.tek.com/manual/077001022.pdf
            """
        )


class _TektronixDPOWaveformFormat(InstrumentChannel):
    """
    This class abstracts the waveform formatting data
    and is meant to be strictly private. Instances of
    this class are coupled to specific channels, because
    parameter values of this class depend on which
    channel has been selected. The end user
    is not intended to call parameters on this submodule
    directly.
    """

    def __init__(self, parent: Instrument, name: str) -> None:
        super().__init__(parent, name)

        self.add_parameter(
            "raw_data_offset",
            get_cmd="WFMOutPRE:YOFF?",
            get_parser=float,
            docstring="""
            Raw acquisition values range from min to max. 
            For instance, for unsigned binary values of one 
            byte, min=0 and max=255. The data offset specifies 
            the center of this range
            """
        )

        self.add_parameter(
            "x_unit",
            get_cmd="WFMOutpre:XUNit?",
            get_parser=strip_quotes
        )

        self.add_parameter(
            "x_increment",
            get_cmd="WFMOutPRE:XINCR?",
            unit=self.x_unit(),
            get_parser=float
        )

        self.add_parameter(
            "y_unit",
            get_cmd="WFMOutpre:YUNit?",
            get_parser=strip_quotes
        )

        self.add_parameter(
            "offset",
            get_cmd="WFMOutPRE:YZERO?",
            get_parser=float,
            unit=self.y_unit()
        )

        self.add_parameter(
            "scale",
            get_cmd="WFMOutPRE:YMULT?",
            get_parser=float,
            unit=self.y_unit()
        )


class TektronixDPOChannel(InstrumentChannel):
    channel_types = {
        "CH": "channel",
        "MATH": "math",
        "REF": "reference"
    }

    def __init__(
            self,
            parent: Instrument,
            name: str,
            channel_type: str,
            channel_number: int,
    ) -> None:

        if channel_type not in self.channel_types:
            acceptable_types = "".join(self.channel_types.keys())
            raise ValueError(
                f"Channel type needs to be one of {acceptable_types}"
            )

        super().__init__(parent, name)

        self._channel_type = channel_type
        self._channel_number = channel_number
        self._identifier = f"{channel_type}{channel_number}"

        waveform_format = _TektronixDPOWaveformFormat(
            cast(VisaInstrument, self.parent),
            "_waveform_format"
        )

        self.add_submodule(
            "_waveform_format",
            waveform_format
        )

        self.add_submodule(
            "_data",
            _TektronixDPOData(
                cast(Instrument, self),
                "_data"
            )
        )

        self.add_parameter(
            "scale",
            get_cmd=f"{self._identifier}:SCA?",
            set_cmd=f"{self._identifier}:SCA {{}}",
            get_parser=float,
            unit="V/div"
        )

        self.add_parameter(
            "offset",
            get_cmd=f"{self._identifier}:OFFS?",
            set_cmd=f"{self._identifier}:OFFS {{}}",
            get_parser=float,
            unit="V"
        )

        self.add_parameter(
            "position",
            get_cmd=f"{self._identifier}:POS?",
            set_cmd=f"{self._identifier}:POS {{}}",
            get_parser=float,
            unit="V"
        )

        self.add_parameter(
            "termination",
            get_cmd=f"{self._identifier}:TER?",
            set_cmd=f"{self._identifier}:TER {{}}",
            vals=Enum(50, 1E6),
            get_parser=float,
            unit="Ohm"
        )

        self.add_parameter(
            "analog_to_digital_threshold",
            get_cmd=f"{self._identifier}:THRESH?",
            set_cmd=f"{self._identifier}:THRESH {{}}",
            get_parser=float,
            unit="V",
        )

        self.add_parameter(
            "termination_voltage",
            get_cmd=f"{self._identifier}:VTERm:BIAS?",
            set_cmd=f"{self._identifier}:VTERm:BIAS {{}}",
            get_parser=float,
            unit="V"
        )

        self.add_parameter(
            "trace_axis",
            label="Time",
            get_cmd=self._get_trace_setpoints,
            vals=Arrays(shape=(self.get_trace_length,)),
            unit=waveform_format.x_unit()
        )

        self.add_parameter(
            "trace",
            label="Voltage",
            get_cmd=self._get_trace_data,
            vals=Arrays(shape=(self.get_trace_length,)),
            unit=waveform_format.y_unit(),
            setpoints=(self.trace_axis,),
            parameter_class=ParameterWithSetpoints
        )

    def _get_trace_data(self) -> np.ndarray:
        """
        Query the instrument for the waveform
        """
        self._data.source(self._identifier)
        self._data.encoding("RPBinary")

        scale = self._waveform_format.scale()
        offset = self._waveform_format.offset()
        raw_data_offset = self._waveform_format.raw_data_offset()

        raw_data = self.parent.visa_handle.query_binary_values(
            'CURVE?',
            datatype='B'
        )

        return (np.array(raw_data) - raw_data_offset) * scale + offset

    def _get_trace_setpoints(self) -> np.ndarray:
        """
        Infer the set points of the waveform
        """
        sample_count = self.get_trace_length()
        sample_rate = self.parent.horizontal.sample_rate()

        return np.linspace(0, sample_count / sample_rate, sample_count)

    def get_trace_length(self) -> int:
        """
        Return:
            The number of samples in the trace
        """
        return self._data.stop() - self._data.start() + 1

    def set_trace_length(self, value: int) -> None:
        """
        Args:
            The requested number of samples in the trace
        """
        if self.parent.horizontal.record_length() < value:
            raise ValueError(
                "Cannot set a trace length which is larger than "
                "the record length. Please switch to manual mode "
                "and adjust the record length first"
            )

        self._data.start(1)
        self._data.stop(value)

    def set_trace_time(self, value: float) -> None:
        """
        Args:
            The time over which a trace is desired.
        """
        sample_rate = self.parent.horizontal.sample_rate()
        required_sample_count = sample_rate * value
        self.set_trace_length(required_sample_count)


class TektronixDPOHorizontal(InstrumentChannel):
    """
    This module controls the horizontal axis of the scope
    """

    def __init__(self, parent, name):
        super().__init__(parent, name)

        self.add_parameter(
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
            """
        )

        self.add_parameter(
            "unit",
            get_cmd="HORizontal:MAIn:UNIts?",
            get_parser=strip_quotes
        )

        self.add_parameter(
            "record_length",
            get_cmd="HORizontal:MODE:RECOrdlength?",
            set_cmd=self._set_record_length,
            get_parser=float
        )

        self.add_parameter(
            "sample_rate",
            get_cmd="HORizontal:MODE:SAMPLERate?",
            set_cmd="HORizontal:MODE:SAMPLERate {}",
            get_parser=float,
            unit=f"sample/{self.unit()}"
        )

        self.add_parameter(
            "scale",
            get_cmd="HORizontal:MODE:SCAle?",
            set_cmd=self._set_scale,
            get_parser=float,
            unit=f"{self.unit()}/div"
        )

        self.add_parameter(
            "position",
            get_cmd="HORizontal:POSition?",
            set_cmd="HORizontal:POSition {}",
            get_parser=float,
            unit="%",
            docstring="""
            The horizontal position relative to a 
            received trigger. E.g. a value of '10'
            sets the trigger position of the waveform 
            such that 10% of the display is to the 
            left of the trigger position.
            """
        )

        self.add_parameter(
            "roll",
            get_cmd="HORizontal:ROLL?",
            set_cmd="HORizontal:ROLL {}",
            vals=Enum("Auto", "On", "Off"),
            docstring="""
            Use Roll Mode when you want to view data at 
            very slow sweep speeds.
            """
        )

    def _set_record_length(self, value: int) -> None:
        if self.mode() != "manual":
            raise ModeError(
                "The record length can only be changed in manual mode"
            )

        self.write(f"HORizontal:MODE:RECOrdlength {value}")

    def _set_scale(self, value):
        if self.mode() == "manual":
            raise ModeError(
                "The scale cannot be changed in manual mode"
            )

        self.write(f"HORizontal:MODE:SCAle {value}")


class TektronixDPOMeasurement(InstrumentChannel):
    """
    The measurement submodule
    """
    # It was found by trial and error that adjusting
    # the measurement type and source takes some time
    # to reflect properly on the value of the
    # measurement. Wait a minimum of ...
    minimum_adjustment_time = 0.1
    # seconds after setting measurement type/source before
    # calling the measurement value SCPI command.

    def __init__(
            self,
            parent: Instrument,
            name: str,
            measurement_number: int
    ) -> None:

        super().__init__(parent, name)
        self._measurement_number = measurement_number
        self._adjustment_time = time.time()

        self.add_parameter(
            "type",
            get_cmd=f"MEASUrement:MEAS{self._measurement_number}:TYPe?",
            set_cmd=self._set_measurement_type,
            get_parser=str.lower,
            vals=Enum(
                "amplitude", "area", "burst", "carea", "cmean", "crms",
                "delay", "distduty", "extinctdb", "extinctpct",
                "extinctratio", "eyeheight", "eyewidth", "fall",
                "frequency", "high", "hits", "low", "maximum", "mean",
                "median", "minimum", "ncross", "nduty", "novershoot",
                "nwidth", "pbase", "pcross", "pctcross", "pduty",
                "peakhits", "period", "phase", "pk2pk", "pkpkjitter",
                "pkpknoise", "povershoot", "ptop", "pwidth", "qfactor",
                "rise", "rms", "rmsjitter", "rmsnoise", "sigma1",
                "sigma2", "sigma3", "sixsigmajit", "snratio", "stddev",
                "undefined", "waveforms"
            ),
            docstring="Please see page 566-569 of the programmers manual "
                      "for a detailed description of these arguments. "
                      "http://download.tek.com/manual/077001022.pdf"
        )

        self.add_parameter(
            "unit",
            get_cmd=f"MEASUrement:MEAS{self._measurement_number}:UNIts?",
            get_parser=strip_quotes
        )

        for src in [1, 2]:
            self.add_parameter(
                f"source{src}",
                get_cmd=f"MEASUrement:MEAS{self._measurement_number}:SOUrce{src}?",
                set_cmd=partial(self._set_source, src),
                vals=Enum(
                    *[f"CH{i}" for i in range(1, TektronixDPO7000xx.channel_count + 1)]
                ),
            )

    def _set_measurement_type(self, value):
        self._adjustment_time = time.time()
        self.write(
            f"MEASUrement:MEAS{self._measurement_number}:TYPe {value}"
        )

    def _set_source(self, source_number, value):
        self._adjustment_time = time.time()
        self.write(
            f"MEASUrement:MEAS{self._measurement_number}:SOUrce{source_number} {value}"
        )

    @property
    def value(self) -> Parameter:
        """
        Return the appropriate parameter for the selected measurement
        type
        """
        measurement_type = self.type()
        name = f"_{measurement_type}_measurement"

        if name not in self.parameters:
            self.add_parameter(
                name,
                get_cmd=self._measure,
                get_parser=float,
                unit=self.unit()
            )

        return self.parameters[name]

    def _measure(self) -> Any:
        """
        We need to wait a minimum amount of time after performing
        some set actions to get a measurement value. Note that we
        cannot use the post_delay or inter_delay parameter options
        here, because these are minimum delays between consecutive
        set operations, not delays between set and get of two
        different parameters.
        """
        time_since_adjust = time.time() - self._adjustment_time
        if time_since_adjust < self.minimum_adjustment_time:
            time_remaining = self.minimum_adjustment_time - time_since_adjust
            time.sleep(time_remaining)

        return self.ask(f"MEASUrement:MEAS{self._measurement_number}:VALue?")


class TektronixDPO7000xx(VisaInstrument):
    """
    QCoDeS driver for the MSO/DPO5000/B, DPO7000/C,
    DPO70000/B/C/D/DX/SX, DSA70000/B/C/D, and
    MSO70000/C/DX Series Digital Oscilloscopes
    """
    channel_count = 4
    measurement_count = 8

    def __init__(
            self,
            name: str,
            address: str,
            **kwargs
    ) -> None:

        super().__init__(name, address, terminator="\n", **kwargs)

        self.add_submodule(
            "horizontal",
            TektronixDPOHorizontal(self, "horizontal")
        )

        measurement_list = ChannelList(self, "measurements", TektronixDPOMeasurement)
        for measurement_number in range(1, self.measurement_count):
            measurement_name = f"measurement{measurement_number}"
            measurement_module = TektronixDPOMeasurement(
                self,
                measurement_name,
                measurement_number
            )
            self.add_submodule(measurement_name, measurement_module)
            measurement_list.append(measurement_module)

        self.add_submodule("measurement", measurement_list)

        channel_types = TektronixDPOChannel.channel_types.items()
        for channel_type, friendly_name in channel_types:
            channel_list = ChannelList(self, friendly_name, TektronixDPOChannel)

            for channel_number in range(1, self.channel_count + 1):

                channel_name = f"{channel_type}{channel_number}"
                channel_module = TektronixDPOChannel(
                    self,
                    channel_name,
                    channel_type,
                    channel_number,
                )

                self.add_submodule(channel_name, channel_module)
                channel_list.append(channel_module)

            self.add_submodule(friendly_name, channel_list)

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
