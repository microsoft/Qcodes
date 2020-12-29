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

    def get_raw(self) -> Arrays:
        return np.linspace(self._start, self._stop, self._npts)


class Trace(ParameterWithSetpoints):
    pass


class N9030B(VisaInstrument):
    """
    Driver for Keysight N9030B PXA signal analyzer.
    """

    def __init__(self, name: str, address: str, **kwargs: Any) -> None:
        super().__init__(name, address, terminator='\n', **kwargs)

        self._min_freq: float = 2
        self._max_freq: float = 50e9

        self.add_parameter(
            name="mode",
            get_cmd=":INSTrument:SELect?",
            set_cmd=":INSTrument:SELect {}",
            vals=Enum(*self._available_modes())
        )

        self.add_parameter(
            name="measurement",
            get_cmd=":CONFigure?",
            set_cmd=":CONFigure:{}",
            val_mapping={"Swept SA": "SANalyzer"}
        )

        self.add_parameter(
            name="start",
            get_cmd=":SENSe:FREQuency:STARt?",
            set_cmd=self._set_start,
            get_parser=float,
            vals=Numbers(self._min_freq, self._max_freq - 10)
        )

        self.add_parameter(
            name="stop",
            get_cmd=":SENSe:FREQuency:STOP?",
            set_cmd=self._set_stop,
            get_parser=float,
            vals=Numbers(self._min_freq + 10, self._max_freq)
        )

        self.add_parameter(
            name="center",
            get_cmd=":SENSe:FREQuency:CENTer?",
            set_cmd=self._set_center,
            get_parser=float,
            vals=Numbers(self._min_freq + 5, self._max_freq - 5)
        )

        self.add_parameter(
            name="span",
            get_cmd=":SENSe:FREQuency:SPAN?",
            set_cmd=self._set_span,
            get_parser=float,
            vals=Numbers(10, self._max_freq - self._min_freq),
        )

        self.add_parameter(
            name="npts",
            get_cmd=":SENSe:SWEep:POINts?",
            set_cmd=self._set_npts,
            get_parser=int,
            vals=Ints(1, 20001)
        )

        self.add_parameter(
            name="sweep_time",
            label="Sweep time",
            get_cmd=":SENSe:SWEep:TIME?",
            get_parser=float,
            unit="s",
        )

        self.add_parameter(
            name="auto_sweep_time_enabled",
            initial_value=False,
            get_cmd=":SENSe:SWEep:TIME:AUTO?",
            set_cmd=self._enable_auto_sweep_time,
            val_mapping=create_on_off_val_mapping(on_val="ON", off_val="OFF")
        )

        self.add_parameter(
            name="auto_sweep_type_enabled",
            initial_value=False,
            get_cmd=":SENSe:SWEep:TYPE:AUTO?",
            set_cmd=self._enable_auto_sweep_type,
            val_mapping=create_on_off_val_mapping(on_val="ON", off_val="OFF")
        )

        self.add_parameter(
            name="sweep_type",
            get_cmd=":SENSe:SWEep:TYPE?",
            set_cmd=self._set_sweep_type,
            val_mapping={
                "fft": "FFT",
                "sweep": "SWE",
            }
        )

        self.add_parameter(
            name='freq_axis',
            label='Frequency',
            unit='Hz',
            start=self.start,
            stop=self.stop,
            npts=self.npts,
            vals=Arrays(shape=(self.npts.get_latest,)),
            parameter_class=FrequencyAxis
        )

        self.add_parameter(
            name='trace',
            vals=Arrays(shape=(self.npts,)),
            setpoints=(self.freq_axis,),
            parameter_class=Trace
        )

        self.add_function("reset", call_cmd="*RST")
        self.add_function("abort", call_cmd=":ABORt")
        self.add_function("autotune", call_cmd=":SENS:FREQuency:TUNE:IMMediate")
        self.add_function("cont_meas_on", call_cmd=":INITiate:CONTinuous ON")
        self.add_function("cont_meas_off", call_cmd=":INITiate:CONTinuous OFF")

        self.connect_message()

    def _available_modes(self) -> Tuple[str, ...]:
        available_modes = self.ask(":INSTrument:CATalog?")
        return tuple(available_modes.split(','))

    def _set_start(self, val: float) -> None:
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
        self.write(f":SENSe:FREQuency:CENTer {val}")
        self.update_trace()

    def _set_span(self, val: float) -> None:
        self.write(f":SENSe:FREQuency:SPAN {val}")
        self.update_trace()

    def _set_npts(self, val: int) -> None:
        self.write(f":SENSe:SWEep:POINts {val}")
        self.update_trace()

    def _enable_auto_sweep_time(self, val: str) -> None:
        self.write(f":SENSe:SWEep:TIME:AUTO {val}")

    def _enable_auto_sweep_type(self, val: str) -> None:
        self.write(f":SENSe:SWEep:TYPE:AUTO {val}")

    def _set_sweep_type(self, val: str) -> None:
        self.write(f":SENSe:SWEep:TYPE {val}")

    def _get_data(self) -> ParamRawDataType:
        pass

    def update_trace(self) -> None:
        pass
