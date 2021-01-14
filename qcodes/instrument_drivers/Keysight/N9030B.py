import numpy as np
from typing import Any, Tuple

from qcodes import VisaInstrument, Parameter, ParameterWithSetpoints
from qcodes.instrument.parameter import ParamRawDataType
from qcodes.utils.validators import Enum, Numbers, Arrays, Ints
from qcodes.utils.helpers import create_on_off_val_mapping


class FrequencyAxis(Parameter):

    def __init__(self, start: float, stop: float, npts: int, *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._start: float = start
        self._stop: float = stop
        self._npts: int = npts

    def get_raw(self) -> ParamRawDataType:
        return np.linspace(self._start, self._stop, self._npts)


class Trace(ParameterWithSetpoints):

    def __init__(self, number: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.instrument: "SpectrumAnalyzer"
        self.root_instrument: "N9030B"

        self.n = number

    def get_raw(self) -> ParamRawDataType:
        return self.instrument._get_data(trace_num=self.n)


class N9030B(VisaInstrument):
    """
    Driver for Keysight N9030B PXA signal analyzer.
    """

    def __init__(self, name: str, address: str, **kwargs: Any) -> None:
        super().__init__(name, address, terminator='\n', **kwargs)

        self._min_freq: float
        self._max_freq: float
        self._additional_wait: float = 1

        self.add_parameter(
            name="mode",
            get_cmd=":INSTrument:SELect?",
            set_cmd=":INSTrument:SELect {}",
            vals=Enum(*self._available_modes())
        )

        self.add_parameter(
            name="cont_meas",
            initial_value=False,
            get_cmd=":INITiate:CONTinuous?",
            set_cmd=self._enable_cont_meas,
            val_mapping=create_on_off_val_mapping(on_val="ON", off_val="OFF")
        )

        self.connect_message()

    def _available_modes(self) -> Tuple[str, ...]:
        available_modes = self.ask(":INSTrument:CATalog?")
        return tuple(available_modes.split(','))

    def _available_meas(self) -> Tuple[str, ...]:
        """
        Gives available measurement with a given mode for the instrument
        """
        available_meas = self.ask(":CONFigure:CATalog?")
        return tuple(available_meas.split(','))

    def _enable_cont_meas(self, val: str) -> None:
        self.write(f":INITiate:CONTinuous {val}")

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


class SpectrumAnalyzer(N9030B):
    """
    Spectrum Analyzer Mode for Kyesight N9030B instrument.
    """
    def __init__(self, name: str, *arg, **kwargs):
        super().__init__(name, *arg, **kwargs)

        if "SA" in self._available_modes():
            self.mode("SA")
        else:
            raise RuntimeError("Spectrum Analyzer Mode is not available on "
                               "your Keysight N9030B instrument.")

        self._min_freq = 2
        self._max_freq = 50e9

        self.add_parameter(
            name="measurement",
            get_cmd=":CONFigure?",
            set_cmd=":CONFigure:{}",
            vals=Enum(*self._available_meas()),
            docstring="Sets measurement type from among the available "
                      "measurement types."
        )

        self.add_parameter(
            name="start",
            get_cmd=":SENSe:FREQuency:STARt?",
            set_cmd=self._set_start,
            get_parser=float,
            vals=Numbers(self._min_freq, self._max_freq - 10),
            docstring="start frequency for the sweep"
        )

        self.add_parameter(
            name="stop",
            get_cmd=":SENSe:FREQuency:STOP?",
            set_cmd=self._set_stop,
            get_parser=float,
            vals=Numbers(self._min_freq + 10, self._max_freq),
            docstring="stop frequency for the sweep"
        )

        self.add_parameter(
            name="center",
            get_cmd=":SENSe:FREQuency:CENTer?",
            set_cmd=self._set_center,
            get_parser=float,
            vals=Numbers(self._min_freq + 5, self._max_freq - 5),
            docstring="Sets and gets center frequency"
        )

        self.add_parameter(
            name="span",
            get_cmd=":SENSe:FREQuency:SPAN?",
            set_cmd=self._set_span,
            get_parser=float,
            vals=Numbers(10, self._max_freq - self._min_freq),
            docstring="Changes span of frequency"
        )

        self.add_parameter(
            name="npts",
            get_cmd=":SENSe:SWEep:POINts?",
            set_cmd=self._set_npts,
            get_parser=int,
            vals=Ints(1, 20001),
            docstring="Number of points for the sweep"
        )

        self.add_parameter(
            name="sweep_time",
            label="Sweep time",
            get_cmd=":SENSe:SWEep:TIME?",
            get_parser=float,
            unit="s",
            docstring="gets sweep time"
        )

        self.add_parameter(
            name="auto_sweep_time_enabled",
            initial_value=False,
            get_cmd=":SENSe:SWEep:TIME:AUTO?",
            set_cmd=self._enable_auto_sweep_time,
            val_mapping=create_on_off_val_mapping(on_val="ON", off_val="OFF"),
            docstring="enables auto sweep time"
        )

        self.add_parameter(
            name="auto_sweep_type_enabled",
            initial_value=False,
            get_cmd=":SENSe:SWEep:TYPE:AUTO?",
            set_cmd=self._enable_auto_sweep_type,
            val_mapping=create_on_off_val_mapping(on_val="ON", off_val="OFF"),
            docstring="enables auto sweep type"
        )

        self.add_parameter(
            name="sweep_type",
            get_cmd=":SENSe:SWEep:TYPE?",
            set_cmd=self._set_sweep_type,
            val_mapping={
                "fft": "FFT",
                "sweep": "SWE",
            },
            docstring="Sets up sweep type. Possible options are 'fft' and "
                      "'sweep'."
        )

        self.add_parameter(
            name="format",
            get_cmd=":FORMat:TRACe:DATA?",
            set_cmd=":FORMat:TRACe:DATA {}",
            val_mapping={
                "ascii": "ASCii",
                "int32": "INTeger,32",
                "real32": "REAL,32",
                "real64": "REAL,64"
            },
            docstring="Sets up format of data received"
        )

        self.add_parameter(
            name="freq_axis",
            label="Frequency",
            unit="Hz",
            start=self.start,
            stop=self.stop,
            npts=self.npts,
            vals=Arrays(shape=(self.npts.get_latest,)),
            parameter_class=FrequencyAxis,
            docstring="Sets frequency axis for the sweep."
        )

        self.add_parameter(
            name="trace",
            label="Trace",
            unit="dB",
            number=1,
            vals=Arrays(shape=(self.npts.get_latest,)),
            setpoints=(self.freq_axis,),
            parameter_class=Trace,
            docstring="Gets trace data."
        )

    def _set_start(self, val: float) -> None:
        """
        Sets start frequency
        """
        stop = self.stop()
        if val >= stop:
            raise ValueError(f"Start frequency must be smaller than stop "
                             f"frequency. Provided start freq is: {val} Hz and "
                             f"set stop freq is: {stop} Hz")

        self.write(f":SENSe:FREQuency:STARt {val}")

        start = self.start()
        if abs(val - start) >= 1:
            self.log.warning(
                f"Could not set start to {val} setting it to {start}"
            )

    def _set_stop(self, val: float) -> None:
        """
        Sets stop frequency
        """
        start = self.start()
        if val <= start:
            raise ValueError(f"Stop frequency must be larger than start "
                             f"frequency. Provided stop freq is: {val} Hz and "
                             f"set start freq is: {start} Hz")

        self.write(f":SENSe:FREQuency:STOP {val}")

        stop = self.stop()
        if abs(val - stop) >= 1:
            self.log.warning(
                f"Could not set stop to {val} setting it to {stop}"
            )

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

    def _set_npts(self, val: int) -> None:
        """
        Sets number of points for sweep
        """
        self.write(f":SENSe:SWEep:POINts {val}")

    def _enable_auto_sweep_time(self, val: str) -> None:
        """
        Enables auto sweep time
        """
        self.write(f":SENSe:SWEep:TIME:AUTO {val}")

    def _enable_auto_sweep_type(self, val: str) -> None:
        """
        Enables auto sweep type
        """
        self.write(f":SENSe:SWEep:TYPE:AUTO {val}")

    def _set_sweep_type(self, val: str) -> None:
        """
        Sets sweep type
        """
        self.write(f":SENSe:SWEep:TYPE {val}")

    def _get_data(self, trace_num: int) -> ParamRawDataType:
        """
        Gets data from the measurement.
        """
        self.cont_meas("OFF")
        try:
            timeout = self.sweep_time() + self._additional_wait
            with self.timeout.set_to(timeout):
                data_str = self.ask(f":READ:{self.measurement}{trace_num}?")
                data = np.array(data_str.rstrip()).astype("float64")
        finally:
            self.cont_meas("ON")

        return data

    def update_trace(self) -> None:
        """
        Updates start and stop frequencies whenever span of/or center frequency
        is updated.
        """
        self.start()
        self.stop()

    def setup_swept_sa_sweep(self,
                             start: float,
                             stop: float,
                             npts: int) -> None:
        """
        Sets up the Swept SA measurement sweep for Spectrum Analyzer Mode.
        """
        if "SANalyzer" in self._available_meas():
            self.measurement("SANalyzer")
        else:
            raise RuntimeError("Swept SA measurement is not available on your "
                               "Keysight N9030B instrument with Spectrum "
                               "Analyzer mode.")
        self.start(start)
        self.stop(stop)
        self.npts(npts)

    def autotune(self) -> None:
        """
        Autotunes frequency
        """
        self.write(":SENS:FREQuency:TUNE:IMMediate")
