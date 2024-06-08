from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from qcodes.instrument import (
    InstrumentBaseKWArgs,
    InstrumentChannel,
    VisaInstrument,
    VisaInstrumentKWArgs,
)
from qcodes.parameters import (
    Parameter,
    ParameterWithSetpoints,
    ParamRawDataType,
    create_on_off_val_mapping,
)
from qcodes.validators import Arrays, Bool, Enum, Ints, Numbers

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Unpack


class FrequencyAxis(Parameter):
    def __init__(
        self,
        start: Parameter,
        stop: Parameter,
        npts: Parameter,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._start: Parameter = start
        self._stop: Parameter = stop
        self._npts: Parameter = npts

    def get_raw(self) -> ParamRawDataType:
        start_val = self._start()
        stop_val = self._stop()
        npts_val = self._npts()
        assert start_val is not None
        assert stop_val is not None
        assert npts_val is not None
        return np.linspace(start_val, stop_val, npts_val)


class Trace(ParameterWithSetpoints):
    def __init__(
        self,
        number: int,
        *args: Any,
        get_data: Callable[[int], ParamRawDataType],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # the parameter classes should ideally be generic in instrument
        # and root instrument classes so we can specialize here.
        # for now we have to ignore a type error from pyright
        self.instrument: (
            KeysightN9030BSpectrumAnalyzerMode | KeysightN9030BPhaseNoiseMode
        )
        self.root_instrument: KeysightN9030B

        self.number = number
        self.get_data = get_data

    def get_raw(self) -> ParamRawDataType:
        return self.get_data(self.number)


class KeysightN9030BSpectrumAnalyzerMode(InstrumentChannel):
    """
    Spectrum Analyzer Mode for Keysight N9030B instrument.
    """

    def __init__(
        self,
        parent: KeysightN9030B,
        name: str,
        *arg: Any,
        additional_wait: int = 1,
        **kwargs: Unpack[InstrumentBaseKWArgs],
    ):
        super().__init__(parent, name, *arg, **kwargs)

        self._additional_wait = additional_wait
        self._min_freq = -8e7
        self._valid_max_freq: dict[str, float] = {
            "503": 3.7e9,
            "508": 8.5e9,
            "513": 13.8e9,
            "526": 27e9,
            "544": 44.5e9,
        }
        opt: str | None = None
        for hw_opt_for_max_freq in self._valid_max_freq:
            if hw_opt_for_max_freq in self.root_instrument.options():
                opt = hw_opt_for_max_freq
        assert opt is not None
        self._max_freq = self._valid_max_freq[opt]

        # Frequency Parameters
        self.start: Parameter = self.add_parameter(
            name="start",
            unit="Hz",
            get_cmd=":SENSe:FREQuency:STARt?",
            set_cmd=self._set_start,
            get_parser=float,
            vals=Numbers(self._min_freq, self._max_freq - 10),
            docstring="Start Frequency",
        )
        """Start Frequency"""
        self.stop: Parameter = self.add_parameter(
            name="stop",
            unit="Hz",
            get_cmd=":SENSe:FREQuency:STOP?",
            set_cmd=self._set_stop,
            get_parser=float,
            vals=Numbers(self._min_freq + 10, self._max_freq),
            docstring="Stop Frequency",
        )
        """Stop Frequency"""
        self.center: Parameter = self.add_parameter(
            name="center",
            unit="Hz",
            get_cmd=":SENSe:FREQuency:CENTer?",
            set_cmd=self._set_center,
            get_parser=float,
            vals=Numbers(self._min_freq + 5, self._max_freq - 5),
            docstring="Sets and gets center frequency",
        )
        """Sets and gets center frequency"""
        self.span: Parameter = self.add_parameter(
            name="span",
            unit="Hz",
            get_cmd=":SENSe:FREQuency:SPAN?",
            set_cmd=self._set_span,
            get_parser=float,
            vals=Numbers(10, self._max_freq - self._min_freq),
            docstring="Changes span of frequency",
        )
        """Changes span of frequency"""
        self.npts: Parameter = self.add_parameter(
            name="npts",
            get_cmd=":SENSe:SWEep:POINts?",
            set_cmd=":SENSe:SWEep:POINts {}",
            get_parser=int,
            vals=Ints(1, 20001),
            docstring="Number of points for the sweep",
        )
        """Number of points for the sweep"""

        # Amplitude/Input Parameters
        self.mech_attenuation: Parameter = self.add_parameter(
            name="mech_attenuation",
            unit="dB",
            get_cmd=":SENS:POW:ATT?",
            set_cmd=":SENS:POW:ATT {}",
            get_parser=int,
            vals=Ints(0, 70),
            docstring="Internal mechanical attenuation",
        )
        """Internal mechanical attenuation"""
        self.preamp: Parameter = self.add_parameter(
            name="preamp",
            get_cmd=":SENS:POW:GAIN:BAND?",
            set_cmd=":SENS:POW:GAIN:BAND {}",
            vals=Enum("LOW", "FULL"),
            docstring="Preamplifier selection",
        )
        """Preamplifier selection"""
        self.preamp_enabled: Parameter = self.add_parameter(
            name="preamp_enabled",
            get_cmd=":SENS:POW:GAIN:STAT?",
            set_cmd=":SENS:POW:GAIN:STAT {}",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
            docstring="Preamplifier state",
        )
        """Preamplifier state"""

        # Resolution parameters
        self.res_bw: Parameter = self.add_parameter(
            name="res_bw",
            unit="Hz",
            get_cmd=":SENS:BAND:RES?",
            set_cmd=":SENS:BAND:RES {}",
            get_parser=float,
            vals=Numbers(1, 8e6),
            docstring="Resolution Bandwidth",
        )
        """Resolution Bandwidth"""
        self.video_bw: Parameter = self.add_parameter(
            name="video_bw",
            unit="Hz",
            get_cmd=":SENS:BAND:VID?",
            set_cmd=":SENS:BAND:VID {}",
            get_parser=float,
            vals=Numbers(1, 50e6),
            docstring="Video Filter Bandwidth",
        )
        """Video Filter Bandwidth"""
        self.res_bw_type: Parameter = self.add_parameter(
            name="res_bw_type",
            get_cmd=":SENS:BAND:TYPE?",
            set_cmd=":SENS:BAND:TYPE {}",
            vals=Enum("DB3", "DB6", "IMP", "NOISE"),
            docstring=(
                "The instrument provides four ways of specifying the "
                "bandwidth of a Gaussian filter:\n"
                " 1. The -3 dB bandwidth of the filter (DB3)\n"
                " 2. The -6 dB bandwidth of the filter (DB6)\n"
                " 3. The equivalent Noise bandwidth of the filter, "
                "which is defined as the bandwidth of a rectangular "
                "filter with the same peak gain which would pass the "
                "same power for noise signals\n"
                " 4. The equivalent Impulse bandwidth of the filter, "
                "which is defined as the bandwidth of a rectangular "
                "filter with the same peak gain which would pass the "
                "same power for impulsive (narrow pulsed) signals."
            ),
        )
        """
        The instrument provides four ways of specifying the bandwidth
        of a Gaussian filter:

            1. The -3 dB bandwidth of the filter (DB3)
            2. The -6 dB bandwidth of the filter (DB6)
            3. The equivalent Noise bandwidth of the filter,
               which is defined as the bandwidth of a rectangular
               filter with the same peak gain which would pass the
               same power for noise signals
            4. The equivalent Impulse bandwidth of the filter,
               which is defined as the bandwidth of a rectangular filter
               with the same peak gain which would pass the same power
               for impulsive (narrow pulsed) signals.
            """

        # Input parameters
        self.detector: Parameter = self.add_parameter(
            name="detector",
            get_cmd=":SENS:DET:TRAC?",
            set_cmd=":SENS:DET:TRAC {}",
            vals=Enum("NORM", "AVER", "POS", "SAMP", "NEG"),
            docstring="Detector type",
        )
        """Detector type"""
        self.average_type: Parameter = self.add_parameter(
            name="average_type",
            get_cmd=":SENS:AVER:TYPE?",
            set_cmd=":SENS:AVER:TYPE {}",
            vals=Enum("LOG", "RMS", "SCAL"),
            docstring=(
                "Lets you control the way averaging is done. The averaging processes "
                "affected are:\n"
                " 1. Trace averaging\n"
                " 2. Average detector averages signals within the resolution BW\n"
                " 3. Noise marker is corrected for average type\n"
                " 4. VBW filtering (not affected if Average detector is used).\n"
                "The averaging types are:"
                " 1. LOG: Selects the logarithmic (decibel) scale for all filtering and "
                "averaging processes. This scale is sometimes called 'Video' because it "
                "is the most common display and analysis scale for the video signal "
                "within a spectrum instrument. This scale is excellent for finding CW "
                "signals near noise, but its response to noise-like signals is 2.506 dB "
                "lower than the average power of those noise signals. This is compensated "
                "for in the Marker Noise function.\n"
                " 2. RMS: All filtering and averaging processes work on the power (the square "
                "of the magnitude) of the signal, instead of its log or envelope voltage. This "
                "scale is best for measuring the true time average power of complex signals. "
                "This scale is sometimes called RMS because the resulting voltage is proportional "
                "to the square root of the mean of the square of the voltage.\n"
                " 3. SCAL: (Voltage) All filtering and averaging processes work on the voltage "
                "of the envelope of the signal. This scale is good for observing rise and fall "
                "behavior of AM or pulse-modulated signals such as radar and TDMA transmitters, "
                "but its response to noise-like signals is 1.049 dB lower than the average power "
                "of those noise signals. This is compensated for in the Marker Noise function."
            ),
        )
        """
        Lets you control the way averaging is done. The averaging
        processes affected are:

        1. Trace averaging
        2. Average detector averages signals within the resolution BW
        3. Noise marker is corrected for average type
        4. VBW filtering (not affected if Average detector is used).

        The averaging types are:

        1. LOG: Selects the logarithmic (decibel) scale for all filtering
           and averaging processes. This scale is sometimes called
           'Video' because it is the most common display and analysis
           scale for the video signal within a spectrum instrument.
           This scale is excellent for finding CW signals near noise,
           but its response to noise-like signals is 2.506 dB lower
           than the average power of those noise signals.
           This is compensated for in the Marker Noise function.
        2. RMS: All filtering and averaging processes work on the power
           (the square of the magnitude) of the signal, instead of its
           log or envelope voltage. This scale is best for measuring
           the true time average power of complex signals. This scale
           is sometimes called RMS because the resulting voltage is
           proportional to the square root of the mean of the square
           of the voltage.
        3. SCAL: (Voltage) All filtering and averaging processes
           work on the voltage of the envelope of the signal.
           This scale is good for observing rise and fall behavior
           of AM or pulse-modulated signals such as radar and TDMA transmitters,
           but its response to noise-like signals is 1.049 dB lower than
           the average power of those noise signals. This is compensated
           for in the Marker Noise function."""

        # Sweep Parameters
        self.sweep_time: Parameter = self.add_parameter(
            name="sweep_time",
            label="Sweep time",
            get_cmd=":SENSe:SWEep:TIME?",
            set_cmd=":SENSe:SWEep:TIME {}",
            get_parser=float,
            unit="s",
            docstring="gets sweep time",
        )
        """gets sweep time"""
        self.auto_sweep_time_enabled: Parameter = self.add_parameter(
            name="auto_sweep_time_enabled",
            get_cmd=":SENSe:SWEep:TIME:AUTO?",
            set_cmd=":SENSe:SWEep:TIME:AUTO {}",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
            docstring="enables auto sweep time",
        )
        """enables auto sweep time"""
        self.auto_sweep_type_enabled: Parameter = self.add_parameter(
            name="auto_sweep_type_enabled",
            get_cmd=":SENSe:SWEep:TYPE:AUTO?",
            set_cmd=":SENSe:SWEep:TYPE:AUTO {}",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
            docstring="enables auto sweep type",
        )
        """enables auto sweep type"""
        self.sweep_type: Parameter = self.add_parameter(
            name="sweep_type",
            get_cmd=":SENSe:SWEep:TYPE?",
            set_cmd=":SENSe:SWEep:TYPE {}",
            val_mapping={
                "fft": "FFT",
                "sweep": "SWE",
            },
            docstring="Sets up sweep type. Possible options are 'fft' and 'sweep'.",
        )
        """Sets up sweep type. Possible options are 'fft' and 'sweep'."""

        # Array (Data) Parameters
        self.freq_axis: FrequencyAxis = self.add_parameter(
            name="freq_axis",
            label="Frequency",
            unit="Hz",
            start=self.start,
            stop=self.stop,
            npts=self.npts,
            vals=Arrays(shape=(self.npts.get_latest,)),
            parameter_class=FrequencyAxis,
            docstring="Creates frequency axis for the sweep from start, "
            "stop and npts values.",
        )
        """Creates frequency axis for the sweep from start, stop and npts values."""
        self.trace: Trace = self.add_parameter(
            name="trace",
            label="Trace",
            unit="dB",
            number=1,
            vals=Arrays(shape=(self.npts.get_latest,)),
            setpoints=(self.freq_axis,),
            get_data=self._get_data,
            parameter_class=Trace,
            docstring="Gets trace data.",
        )
        """Gets trace data."""

    def _set_start(self, val: float) -> None:
        """
        Sets start frequency
        """
        self.write(f":SENSe:FREQuency:STARt {val}")
        self.update_trace()

        start = self.start.cache.get()
        if abs(val - start) >= 1:
            self.log.warning(f"Start frequency rounded to {start}")

    def _set_stop(self, val: float) -> None:
        """
        Sets stop frequency
        """
        self.write(f":SENSe:FREQuency:STOP {val}")
        self.update_trace()

        stop = self.stop.cache.get()
        if abs(val - stop) >= 1:
            self.log.warning(f"Stop frequency rounded to {stop}")

    def _set_center(self, val: float) -> None:
        """
        Sets center frequency and updates start and stop frequencies if they
        change.
        """
        self.write(f":SENSe:FREQuency:CENTer {val}")
        self.update_trace()

    def _set_span(self, val: float) -> None:
        """
        Sets frequency span and updates start and stop frequencies if they
        change.
        """
        self.write(f":SENSe:FREQuency:SPAN {val}")
        self.update_trace()

    def _get_data(self, trace_num: int) -> ParamRawDataType:
        """
        Gets data from the measurement.
        """
        root_instr = self.root_instrument
        # Check if we should run a new sweep
        auto_sweep = root_instr.auto_sweep()

        if auto_sweep:
            # If we need to run a sweep, we need to set the timeout to take into account
            # the sweep time
            timeout = self.sweep_time() + self._additional_wait
            with root_instr.timeout.set_to(timeout):
                data = root_instr.visa_handle.query_binary_values(
                    f":READ:{root_instr.measurement()}{trace_num}?",
                    datatype="d",
                    is_big_endian=False,
                )
        else:
            data = root_instr.visa_handle.query_binary_values(
                f":FETC:{root_instr.measurement()}{trace_num}?",
                datatype="d",
                is_big_endian=False,
            )

        data = np.array(data).reshape((-1, 2))
        return data[:, 1]

    def update_trace(self) -> None:
        """
        Updates all frequency parameters together when one is changed
        """
        self.start()
        self.stop()
        self.span()
        self.center()

    def setup_swept_sa_sweep(self, start: float, stop: float, npts: int) -> None:
        """
        Sets up the Swept SA measurement sweep for Spectrum Analyzer Mode.
        """
        self.root_instrument.mode("SA")
        if "SAN" in self.root_instrument.available_meas():
            self.root_instrument.measurement("SAN")
        else:
            raise RuntimeError(
                "Swept SA measurement is not available on your "
                "Keysight N9030B instrument with Spectrum "
                "Analyzer mode."
            )
        self.start(start)
        self.stop(stop)
        self.npts(npts)

    def autotune(self) -> None:
        """
        Autotune quickly get to the most likely signal of interest, and
        position it optimally on the display.
        """
        self.write(":SENS:FREQuency:TUNE:IMMediate")
        self.center()


class KeysightN9030BPhaseNoiseMode(InstrumentChannel):
    """
    Phase Noise Mode for Keysight N9030B instrument.
    """

    def __init__(
        self,
        parent: KeysightN9030B,
        name: str,
        *arg: Any,
        **kwargs: Unpack[InstrumentBaseKWArgs],
    ):
        super().__init__(parent, name, *arg, **kwargs)

        self._min_freq = 1
        self._valid_max_freq: dict[str, float] = {
            "503": 3699999995,
            "508": 8499999995,
            "513": 13799999995,
            "526": 26999999995,
            "544": 44499999995,
        }
        opt: str | None = None
        for hw_opt_for_max_freq in self._valid_max_freq:
            if hw_opt_for_max_freq in self.root_instrument.options():
                opt = hw_opt_for_max_freq
        assert opt is not None
        self._max_freq = self._valid_max_freq[opt]

        self.npts: Parameter = self.add_parameter(
            name="npts",
            get_cmd=":SENSe:LPLot:SWEep:POINts?",
            set_cmd=":SENSe:LPLot:SWEep:POINts {}",
            get_parser=int,
            vals=Ints(601, 20001),
            docstring="Number of points for the sweep",
        )
        """Number of points for the sweep"""

        self.start_offset: Parameter = self.add_parameter(
            name="start_offset",
            unit="Hz",
            get_cmd=":SENSe:LPLot:FREQuency:OFFSet:STARt?",
            set_cmd=self._set_start_offset,
            get_parser=float,
            vals=Numbers(self._min_freq, self._max_freq - 10),
            docstring="start frequency offset for the plot",
        )
        """start frequency offset for the plot"""

        self.stop_offset: Parameter = self.add_parameter(
            name="stop_offset",
            unit="Hz",
            get_cmd=":SENSe:LPLot:FREQuency:OFFSet:STOP?",
            set_cmd=self._set_stop_offset,
            get_parser=float,
            vals=Numbers(self._min_freq + 99, self._max_freq),
            docstring="stop frequency offset for the plot",
        )
        """stop frequency offset for the plot"""

        self.signal_tracking_enabled: Parameter = self.add_parameter(
            name="signal_tracking_enabled",
            get_cmd=":SENSe:FREQuency:CARRier:TRACk?",
            set_cmd=":SENSe:FREQuency:CARRier:TRACk {}",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
            docstring="Gets/Sets signal tracking. When signal tracking is "
            "enabled carrier signal is repeatedly realigned. Signal "
            "Tracking assumes the new acquisition occurs repeatedly "
            "without pause.",
        )
        """
        Gets/Sets signal tracking.
        When signal tracking is enabled carrier signal is repeatedly
        realigned. Signal Tracking assumes the new acquisition occurs
        repeatedly without pause.
        """

        self.freq_axis: FrequencyAxis = self.add_parameter(
            name="freq_axis",
            label="Frequency",
            unit="Hz",
            start=self.start_offset,
            stop=self.stop_offset,
            npts=self.npts,
            vals=Arrays(shape=(self.npts.get_latest,)),
            parameter_class=FrequencyAxis,
            docstring="Creates frequency axis for the sweep from "
            "start_offset, stop_offset and npts values.",
        )
        """
        Creates frequency axis for the sweep from start_offset,
        stop_offset and npts values.
        """

        self.trace: Trace = self.add_parameter(
            name="trace",
            label="Trace",
            unit="dB",
            number=3,
            vals=Arrays(shape=(self.npts.get_latest,)),
            setpoints=(self.freq_axis,),
            get_data=self._get_data,
            parameter_class=Trace,
            docstring="Gets trace data",
        )
        """Gets trace data"""

    def _set_start_offset(self, val: float) -> None:
        """
        Sets start offset for frequency in the plot
        """
        stop_offset = self.stop_offset()
        self.write(f":SENSe:LPLot:FREQuency:OFFSet:STARt {val}")
        start_offset = self.start_offset()

        if abs(val - start_offset) >= 1:
            self.log.warning(
                f"Could not set start offset to {val} setting it to {start_offset}"
            )
        if val >= stop_offset or abs(val - stop_offset) < 10:
            self.log.warning(
                f"Provided start frequency offset {val} Hz was "
                f"greater than preset stop frequency offset "
                f"{stop_offset} Hz. Provided start frequency "
                f"offset {val} Hz is set and new stop freq offset"
                f" is: {self.stop_offset()} Hz."
            )

    def _set_stop_offset(self, val: float) -> None:
        """
        Sets stop offset for frequency in the plot
        """
        start_offset = self.start_offset()
        self.write(f":SENSe:LPLot:FREQuency:OFFSet:STOP {val}")
        stop_offset = self.stop_offset()

        if abs(val - stop_offset) >= 1:
            self.log.warning(
                f"Could not set stop offset to {val} setting it to {stop_offset}"
            )

        if val <= start_offset or abs(val - start_offset) < 10:
            self.log.warning(
                f"Provided stop frequency offset {val} Hz was "
                f"less than preset start frequency offset "
                f"{start_offset} Hz. Provided stop frequency "
                f"offset {val} Hz is set and new start freq offset"
                f" is: {self.start_offset()} Hz."
            )

    def _get_data(self, trace_num: int) -> ParamRawDataType:
        """
        Gets data from the measurement.
        """
        root_instr = self.root_instrument
        measurement = root_instr.measurement()
        raw_data = root_instr.visa_handle.query_binary_values(
            f":READ:{measurement}1?",
            datatype="d",
            is_big_endian=False,
        )
        trace_res_details = np.array(raw_data)

        if len(trace_res_details) != 7 or (
            len(trace_res_details) >= 1 and trace_res_details[0] < -50
        ):
            self.log.warning("Carrier(s) Incorrect or Missing!")
            return -1 * np.ones(self.npts())

        try:
            data = root_instr.visa_handle.query_binary_values(
                f":READ:{measurement}{trace_num}?",
                datatype="d",
                is_big_endian=False,
            )
            data = np.array(data).reshape((-1, 2))
        except TimeoutError as e:
            raise TimeoutError("Couldn't receive any data. Command timed out.") from e

        return data[:, 1]

    def setup_log_plot_sweep(
        self, start_offset: float, stop_offset: float, npts: int
    ) -> None:
        """
        Sets up the Log Plot measurement sweep for Phase Noise Mode.
        """
        self.root_instrument.mode("PNOISE")
        if "LPL" in self.root_instrument.available_meas():
            self.root_instrument.measurement("LPL")
        else:
            raise RuntimeError(
                "Log Plot measurement is not available on your "
                "Keysight N9030B instrument with Phase Noise "
                "mode."
            )

        self.start_offset(start_offset)
        self.stop_offset(stop_offset)
        self.npts(npts)

    def autotune(self) -> None:
        """
        On autotune, the measurement automatically searches for and tunes to
        the strongest signal in the full span of the analyzer.
        """
        self.write(":SENSe:FREQuency:CARRier:SEARch")
        self.start_offset()
        self.stop_offset()


class KeysightN9030B(VisaInstrument):
    """
    Driver for Keysight N9030B PXA signal analyzer. Keysight N9030B PXA
    signal analyzer is part of Keysight X-Series Multi-touch Signal
    Analyzers.
    This driver allows Swept SA measurements in Spectrum Analyzer mode and
    Log Plot measurements in Phase Noise mode of the instrument.

    Args:
        name
        address
    """

    default_terminator = "\n"

    def __init__(
        self, name: str, address: str, **kwargs: Unpack[VisaInstrumentKWArgs]
    ) -> None:
        super().__init__(name, address, **kwargs)

        self._min_freq: float
        self._max_freq: float
        self._additional_wait: float = 1

        self.mode: Parameter = self.add_parameter(
            name="mode",
            get_cmd=":INSTrument:SELect?",
            set_cmd=":INSTrument:SELect {}",
            vals=Enum(*self.available_modes()),
            docstring="Allows setting of different modes present and licensed "
            "for the instrument.",
        )
        """
        Allows setting of different modes present and licensed
        for the instrument.
        """

        self.measurement: Parameter = self.add_parameter(
            name="measurement",
            get_cmd=":CONFigure?",
            set_cmd=":CONFigure:{}",
            vals=Enum("SAN", "LPL"),
            docstring="Sets measurement type from among the available "
            "measurement types.",
        )
        """Sets measurement type from among the available measurement types."""

        self.cont_meas: Parameter = self.add_parameter(
            name="cont_meas",
            initial_value=False,
            get_cmd=":INITiate:CONTinuous?",
            set_cmd=":INITiate:CONTinuous {}",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
            docstring="Enables or disables continuous measurement.",
        )
        """Enables or disables continuous measurement."""

        # Set auto_sweep parameter
        # If we want to return multiple traces per setpoint without sweeping
        # multiple times, or return data on screen, then we can set this value false
        self.auto_sweep: Parameter = self.add_parameter(
            "auto_sweep",
            label="Auto Sweep",
            set_cmd=None,
            get_cmd=None,
            vals=Bool(),
            initial_value=True,
        )
        """Parameter auto_sweep"""

        # Set binary format and don't allow change. There isn't much point to
        # allow this value to be varied. Retained for backwards compatibility.
        self.format: Parameter = self.add_parameter(
            name="format",
            get_cmd=lambda: "real64",
            set_cmd=False,
            docstring="Sets up format of data received",
        )
        """Sets up format of data received"""
        # Set default format on initialisation
        self.write("FORM REAL,64")
        self.write("FORM:BORD SWAP")

        if "SA" in self.available_modes():
            sa_mode = KeysightN9030BSpectrumAnalyzerMode(
                self, name="sa", additional_wait=self._additional_wait
            )
            self.add_submodule("sa", sa_mode)
        else:
            self.log.info("Spectrum Analyzer mode is not available on this instrument.")

        if "PNOISE" in self.available_modes():
            pnoise_mode = KeysightN9030BPhaseNoiseMode(self, name="pn")
            self.add_submodule("pn", pnoise_mode)
        else:
            self.log.info("Phase Noise mode is not available on this instrument.")
        self.connect_message()

    def available_modes(self) -> tuple[str, ...]:
        """
        Returns present and licensed modes for the instrument.
        """
        available_modes = self.ask(":INSTrument:CATalog?")
        av_modes = available_modes[1:-1].split(",")
        modes: tuple[str, ...] = ()
        for i, mode in enumerate(av_modes):
            if i == 0:
                modes = modes + (mode.split(" ")[0],)
            else:
                modes = modes + (mode.split(" ")[1],)
        return modes

    def available_meas(self) -> tuple[str, ...]:
        """
        Gives available measurement with a given mode for the instrument
        """
        available_meas = self.ask(":CONFigure:CATalog?")
        av_meas = available_meas[1:-1].split(",")
        measurements: tuple[str, ...] = ()
        for i, meas in enumerate(av_meas):
            if i == 0:
                measurements = measurements + (meas,)
            else:
                measurements = measurements + (meas[1:],)
        return measurements

    def options(self) -> tuple[str, ...]:
        """
        Returns installed options numbers.
        """
        options_raw = self.ask("*OPT?")
        return tuple(options_raw[1:-1].split(","))

    def reset(self) -> None:
        """
        Reset the instrument by sending the RST command
        """
        self.write("*RST")

    def abort(self) -> None:
        """
        Aborts the measurement
        """
        self.write(":ABORt")
