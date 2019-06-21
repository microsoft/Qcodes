"""
QCoDeS driver for the MSO/DPO5000/B, DPO7000/C,
DPO70000/B/C/D/DX/SX, DSA70000/B/C/D, and
MSO70000/C/DX Series Digital Oscilloscopes
"""
import numpy as np
from typing import Any, Union, Callable
from functools import partial
import time
import textwrap

from qcodes import (
    Instrument, VisaInstrument, InstrumentChannel, ParameterWithSetpoints,
    ChannelList
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


class TektronixDPO7000xx(VisaInstrument):
    """
    QCoDeS driver for the MSO/DPO5000/B, DPO7000/C,
    DPO70000/B/C/D/DX/SX, DSA70000/B/C/D, and
    MSO70000/C/DX Series Digital Oscilloscopes
    """
    number_of_channels = 4
    number_of_measurements = 8  # The number of available
    # measurements does not change.

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

        self.add_submodule(
            "data",
            TektronixDPOData(self, "data")
        )

        self.add_submodule(
            "waveform",
            TektronixDPOWaveformFormat(
                self, "waveform"
            )
        )

        measurement_list = ChannelList(
            self, "measurement", TektronixDPOMeasurement
        )
        for measurement_number in range(1, self.number_of_measurements):

            measurement_name = f"measurement{measurement_number}"
            measurement_module = TektronixDPOMeasurement(
                self,
                measurement_name,
                measurement_number
            )

            self.add_submodule(measurement_name, measurement_module)
            measurement_list.append(measurement_module)

        self.add_submodule("measurement", measurement_list)

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

        self.add_submodule("channel", channel_list)

        self.add_submodule(
            "trigger",
            TekronixDPOTrigger(self, "trigger")
        )

        self.add_submodule(
            "delayed_trigger",
            TekronixDPOTrigger(self, "delayed_trigger", delayed_trigger=True)
        )

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
            parent: Union[Instrument, InstrumentChannel],
            name: str
    ) -> None:

        super().__init__(parent, name)
        # We can choose to retrieve data from arbitrary
        # start and stop indices of the buffer.
        self.add_parameter(
            "start_index",
            get_cmd="DATa:STARt?",
            set_cmd="DATa:STARt {}",
            get_parser=int
        )

        self.add_parameter(
            "stop_index",
            get_cmd="DATa:STOP?",
            set_cmd="DATa:STOP {}",
            get_parser=int
        )

        self.add_parameter(
            "source",
            get_cmd="DATa:SOU?",
            set_cmd="DATa:SOU {}",
            vals=Enum(*TekronixDPOWaveform.valid_identifiers)
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
            docstring=textwrap.dedent("""
            For a detailed explanation of the 
            set arguments, please consult the 
            programmers manual at page 263/264. 

            http://download.tek.com/manual/077001022.pdf
            """)
        )


class TekronixDPOWaveform(InstrumentChannel):
    """
    This submodule retrieves data from waveform sources, e.g.
    channels.
    """
    valid_identifiers = [
        f"{source_type}{i}"
        for source_type in ["CH", "MATH", "REF"]
        for i in range(1, TektronixDPO7000xx.number_of_channels + 1)
    ]

    def __init__(
            self,
            parent: Union[Instrument, InstrumentChannel],
            name: str,
            identifier: str,
    ) -> None:

        super().__init__(parent, name)

        if identifier not in self.valid_identifiers:
            raise ValueError(
                f"Identifier {identifier} must be one of "
                f"{self.valid_identifiers}"
            )

        self._identifier = identifier

        self.add_parameter(
            "raw_data_offset",
            get_cmd=self._get_cmd("WFMOutPRE:YOFF?"),
            get_parser=float,
            docstring=textwrap.dedent("""
                Raw acquisition values range from min to max. 
                For instance, for unsigned binary values of one 
                byte, min=0 and max=255. The data offset specifies 
                the center of this range
                """)
        )

        self.add_parameter(
            "x_unit",
            get_cmd=self._get_cmd("WFMOutpre:XUNit?"),
            get_parser=strip_quotes
        )

        self.add_parameter(
            "x_increment",
            get_cmd=self._get_cmd("WFMOutPRE:XINCR?"),
            unit=self.x_unit(),
            get_parser=float
        )

        self.add_parameter(
            "y_unit",
            get_cmd=self._get_cmd("WFMOutpre:YUNit?"),
            get_parser=strip_quotes
        )

        self.add_parameter(
            "offset",
            get_cmd=self._get_cmd("WFMOutPRE:YZERO?"),
            get_parser=float,
            unit=self.y_unit()
        )

        self.add_parameter(
            "scale",
            get_cmd=self._get_cmd("WFMOutPRE:YMULT?"),
            get_parser=float,
            unit=self.y_unit()
        )

        self.add_parameter(
            "length",
            get_cmd=self._get_cmd("WFMOutpre:NR_Pt?"),
            get_parser=int
        )

        hor_unit = self.x_unit()
        hor_label = "Time" if hor_unit == "s" else "Frequency"

        self.add_parameter(
            "trace_axis",
            label=hor_label,
            get_cmd=self._get_trace_setpoints,
            vals=Arrays(shape=(self.length,)),
            unit=hor_unit
        )

        ver_unit = self.y_unit()
        ver_label = "Voltage" if ver_unit == "s" else "Amplitude"

        self.add_parameter(
            "trace",
            label=ver_label,
            get_cmd=self._get_trace_data,
            vals=Arrays(shape=(self.length,)),
            unit=ver_unit,
            setpoints=(self.trace_axis,),
            parameter_class=ParameterWithSetpoints
        )

    def _get_cmd(self, cmd_string: str) -> Callable:
        """
        Parameters defined in this submodule require the correct
        data source being selected first.
        """
        def inner():
            self.root_instrument.data.source(self._identifier)
            return self.ask(cmd_string)

        return inner

    def _get_trace_data(self):

        self.root_instrument.data.source(self._identifier)
        waveform = self.root_instrument.waveform

        if not waveform.is_binary():
            raw_data = self.root_instrument.visa_handle.query_ascii_values(
                "CURVE?",
                container=np.array
            )
        else:
            bytes_per_sample = waveform.bytes_per_sample()
            data_type = {1: "b", 2: "h", 4: "f", 8: "d"}[
                bytes_per_sample
            ]

            if waveform.data_format() == "unsigned_integer":
                data_type = data_type.upper()

            is_big_endian = waveform.is_big_endian()

            raw_data = self.root_instrument.visa_handle.query_binary_values(
                "CURVE?",
                datatype=data_type,
                is_big_endian=is_big_endian,
                container=np.array
            )

        return (raw_data - self.raw_data_offset()) * self.scale() \
            + self.offset()

    def _get_trace_setpoints(self) -> np.ndarray:
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
            parent: Union[Instrument, InstrumentChannel],
            name: str
    ) -> None:

        super().__init__(parent, name)

        self.add_parameter(
            "data_format",
            get_cmd="WFMOutpre:BN_Fmt?",
            set_cmd="WFMOutpre:BN_Fmt {}",
            val_mapping={
                "signed_integer": "RI",
                "unsigned_integer": "RP",
                "floating_point": "FP"
            }
        )

        self.add_parameter(
            "is_big_endian",
            get_cmd="WFMOutpre:BYT_Or?",
            set_cmd="WFMOutpre:BYT_Or {}",
            val_mapping={
                False: "LSB",
                True: "MSB"
            }
        )

        self.add_parameter(
            "bytes_per_sample",
            get_cmd="WFMOutpre:BYT_Nr?",
            set_cmd="WFMOutpre:BYT_Nr {}",
            get_parser=int,
            vals=Enum(1, 2, 4, 8)
        )

        self.add_parameter(
            "is_binary",
            get_cmd="WFMOutpre:ENCdg?",
            set_cmd="WFMOutpre:ENCdg {}",
            val_mapping={
                True: "BINARY",
                False: "ASCII"
            }
        )


class TektronixDPOChannel(InstrumentChannel):
    """
    The main channel module for the oscilloscope. The parameters
    defined here reflect the waveforms as they are displayed on
    the instrument display.
    """
    def __init__(
            self,
            parent: Union[Instrument, InstrumentChannel],
            name: str,
            channel_number: int,
    ) -> None:

        super().__init__(parent, name)
        self._identifier = f"CH{channel_number}"

        self.add_submodule(
            "waveform",
            TekronixDPOWaveform(
                self, "waveform", self._identifier
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
            parent: Union[Instrument, InstrumentChannel],
            name: str
    ) -> None:

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
            docstring=textwrap.dedent("""
            The horizontal position relative to a 
            received trigger. E.g. a value of '10'
            sets the trigger position of the waveform 
            such that 10% of the display is to the 
            left of the trigger position.
            """)
        )

        self.add_parameter(
            "roll",
            get_cmd="HORizontal:ROLL?",
            set_cmd="HORizontal:ROLL {}",
            vals=Enum("Auto", "On", "Off"),
            docstring=textwrap.dedent("""
            Use Roll Mode when you want to view data at 
            very slow sweep speeds.
            """)
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


class TekronixDPOTrigger(InstrumentChannel):
    """
    Submodule for trigger setup.

    You can trigger with the A (Main) trigger system alone
    or combine the A (Main) trigger with the B (Delayed) trigger
    to trigger on sequential events. When using sequential
    triggering, the A trigger event arms the trigger system, and
    the B trigger event triggers the instrument when the B
    trigger conditions are met.

    A and B triggers can (and typically do) have separate sources.
    The B trigger condition is based on a time delay or a speciﬁed
    number of events.

    See page75, Using A (Main) and B (Delayed) triggers.
    https://download.tek.com/manual/MSO70000C-DX-DPO70000C-DX-MSO-DPO7000C-MSO-DPO5000B-Oscilloscope-Quick-Start-User-Manual-071298006.pdf
    """
    def __init__(
            self,
            parent: Instrument,
            name: str,
            delayed_trigger: bool = False
    ):
        super().__init__(parent, name)
        self._identifier = "B" if delayed_trigger else "A"

        trigger_types = ["edge", "logic", "pulse"]
        if self._identifier == "A":
            trigger_types.extend([
                "video", "i2c", "can", "spi", "communication", "serial", "rs232"
            ])

        self.add_parameter(
            "type",
            get_cmd=f"TRIGger:{self._identifier}:TYPE?",
            set_cmd=self._trigger_type,
            vals=Enum(*trigger_types),
            get_parser=str.lower
        )

        edge_couplings = ["ac", "dc", "hfrej", "lfrej", "noiserej"]
        if self._identifier == "B":
            edge_couplings.append("atrigger")

        self.add_parameter(
            "edge_coupling",
            get_cmd=f"TRIGger:{self._identifier}:EDGE:COUPling?",
            set_cmd=f"TRIGger:{self._identifier}:EDGE:COUPling {{}}",
            vals=Enum(*edge_couplings),
            get_parser=str.lower
        )

        self.add_parameter(
            "edge_slope",
            get_cmd=f"TRIGger:{self._identifier}:EDGE:SLOpe?",
            set_cmd=f"TRIGger:{self._identifier}:EDGE:SLOpe {{}}",
            vals=Enum("rise", "fall", "either"),
            get_parser=str.lower
        )

        trigger_sources = [
            f"CH{i}" for i in range(1, TektronixDPO7000xx.number_of_channels)
        ]

        trigger_sources.extend([
            f"D{i}" for i in range(0, 16)
        ])

        if self._identifier == "A":
            trigger_sources.append("line")

        self.add_parameter(
            "source",
            get_cmd=f"TRIGger:{self._identifier}:EDGE:SOUrce?",
            set_cmd=f"TRIGger:{self._identifier}:EDGE:SOUrce {{}}",
            vals=Enum(*trigger_sources)
        )

    def _trigger_type(self, value: str):
        if value != "edge":
            raise NotImplementedError(
                "We currently only support the 'edge' trigger type"
            )
        self.write(f"TRIGger:{self._identifier}:TYPE {value}")


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

    measurements = [
        ('amplitude', 'V'), ('area', 'Vs'), ('burst', 's'), ('carea', 'Vs'),
        ('cmean', 'V'), ('crms', 'V'), ('delay', 's'), ('distduty', '%'),
        ('extinctdb', 'dB'), ('extinctpct', '%'), ('extinctratio', ''),
        ('eyeheight', 'V'), ('eyewidth', 's'), ('fall', 's'),
        ('frequency', 'Hz'), ('high', 'V'), ('hits', 'hits'), ('low', 'V'),
        ('maximum', 'V'), ('mean', 'V'), ('median', 'V'), ('minimum', 'V'),
        ('ncross', 's'), ('nduty', '%'), ('novershoot', '%'), ('nwidth', 's'),
        ('pbase', 'V'), ('pcross', 's'), ('pctcross', '%'), ('pduty', '%'),
        ('peakhits', 'hits'), ('period', 's'), ('phase', '°'), ('pk2pk', 'V'),
        ('pkpkjitter', 's'), ('pkpknoise', 'V'), ('povershoot', '%'),
        ('ptop', 'V'), ('pwidth', 's'), ('qfactor', ''), ('rise', 's'),
        ('rms', 'V'), ('rmsjitter', 's'), ('rmsnoise', 'V'), ('sigma1', '%'),
        ('sigma2', '%'), ('sigma3', '%'), ('sixsigmajit', 's'), ('snratio', ''),
        ('stddev', 'V'), ('undefined', ''), ('waveforms', 'wfms')
    ]

    def __init__(
            self,
            parent: Instrument,
            name: str,
            measurement_number: int
    ) -> None:

        super().__init__(parent, name)
        self._measurement_number = measurement_number
        self._adjustment_time = time.perf_counter()

        self.add_parameter(
            "type",
            get_cmd=f"MEASUrement:MEAS{self._measurement_number}:TYPe?",
            set_cmd=self._set_measurement_type,
            get_parser=str.lower,
            vals=Enum(*[m[0] for m in self.measurements]),
            docstring=textwrap.dedent(
                "Please see page 566-569 of the programmers manual "
                "for a detailed description of these arguments. "
                "http://download.tek.com/manual/077001022.pdf"
            )
        )

        for measurement, unit in self.measurements:
            self.add_parameter(
                measurement,
                get_cmd=partial(self._measure, measurement),
                get_parser=float,
                unit=unit
            )

        for src in [1, 2]:
            self.add_parameter(
                f"source{src}",
                get_cmd=f"MEASUrement:MEAS{self._measurement_number}:SOUrce"
                        f"{src}?",
                set_cmd=partial(self._set_source, src),
                vals=Enum(
                    *(TekronixDPOWaveform.valid_identifiers + ["HISTogram"])
                )
            )

    def _set_measurement_type(self, value: str) -> None:
        self._adjustment_time = time.perf_counter()
        self.write(
            f"MEASUrement:MEAS{self._measurement_number}:TYPe {value}"
        )

    def _set_source(self, source_number: int, value: str) -> None:
        self._adjustment_time = time.perf_counter()
        self.write(
            f"MEASUrement:MEAS{self._measurement_number}:SOUrce{source_number} "
            f"{value}"
        )

    def _measure(self, measurement: str) -> Any:
        """
        Args:
            measurement: The type of measurement we wish to perform
        """
        if self.type.get_latest() != measurement:
            self.type(measurement)

        time_since_adjust = time.perf_counter() - self._adjustment_time
        if time_since_adjust < self._minimum_adjustment_time:
            time_remaining = self._minimum_adjustment_time - time_since_adjust
            time.sleep(time_remaining)

        return self.ask(f"MEASUrement:MEAS{self._measurement_number}:VALue?")
