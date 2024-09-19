import logging
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import deprecated

import qcodes.validators as vals
from qcodes.instrument import (
    ChannelList,
    Instrument,
    InstrumentBaseKWArgs,
    InstrumentChannel,
    VisaInstrument,
    VisaInstrumentKWArgs,
)
from qcodes.parameters import (
    ArrayParameter,
    ManualParameter,
    MultiParameter,
    Parameter,
    ParamRawDataType,
    create_on_off_val_mapping,
)
from qcodes.utils import QCoDeSDeprecationWarning

if TYPE_CHECKING:
    from typing_extensions import Unpack

log = logging.getLogger(__name__)


class FixedFrequencyTraceIQ(MultiParameter):
    """
    Parameter for sweep that returns the real (I) and imaginary (Q) parts of
    the VNA response.
    Requires the use of the sweep type to be set to continuous wave mode.
    See (https://www.rohde-schwarz.com/webhelp/ZNB_ZNBT_HTML_UserManual_en
    /ZNB_ZNBT_HTML_UserManual_en.htm) under GUI reference -> sweep softtool
    -> sweep type tab -> CW mode
    """

    def __init__(
        self,
        name: str,
        instrument: "RohdeSchwarzZNBChannel",
        npts: int,
        bandwidth: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name,
            instrument=instrument,
            names=("I", "Q"),
            labels=(f"{instrument.short_name} I", f"{instrument.short_name} Q"),
            units=("", ""),
            setpoint_names=(
                (f"{instrument.short_name}_frequency",),
                (f"{instrument.short_name}_frequency",),
            ),
            setpoint_units=(("s",), ("s",)),
            setpoint_labels=(("time",), ("time",)),
            shapes=(
                (npts,),
                (npts,),
            ),
            **kwargs,
        )
        self.set_cw_sweep(npts, bandwidth)

    def set_cw_sweep(self, npts: int, bandwidth: int) -> None:
        """
        Updates config of the software parameter on sweep change. This is
        needed in order to sync the setpoint shape with the returned data
        shape after a change of sweep settings.

        Sets setpoints to the tuple which are hashable for look up.

        Note: This is similar to the set_sweep functions of the frequency
        sweep parameters. The time setpoints here neglect a small VNA
        overhead. The total time including overhead can be queried with the
        sweep_time function of the vna, but since it is not clear where this
        overhead is spend, we keep the x-axis set to 1/bandwidth. The error
        is only apparent in really fast measurements at 1us and 10us but
        depends on the amount of points you take. More points give less
        overhead.
        """
        t = tuple(np.linspace(0, npts / bandwidth, num=npts))
        self.setpoints = ((t,), (t,))
        self.shapes = ((npts,), (npts,))

    def get_raw(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Gets the raw real and imaginary part of the data. If parameter
        `cw_check_sweep_first` is set to `True` then at the cost of a few ms
        overhead checks if the vna is setup correctly.
        """
        assert isinstance(self.instrument, RohdeSchwarzZNBChannel)
        i, q = self.instrument._get_cw_data()
        return i, q


class FixedFrequencyPointIQ(MultiParameter):
    """
    Parameter for sweep that returns the mean of the real (I) and imaginary (Q)
    parts of the VNA response.
    Requires the use of the sweep type to be set to continuous wave mode.
    See (https://www.rohde-schwarz.com/webhelp/ZNB_ZNBT_HTML_UserManual_en
    /ZNB_ZNBT_HTML_UserManual_en.htm) under GUI reference -> sweep softtool
    -> sweep type tab -> CW mode
    Useful for two-tone and other bigger sweeps where you do not want to
    store all individual I-Q values.

    Args:
        name: parameter name
        instrument: instrument the parameter belongs to
    """

    def __init__(
        self, name: str, instrument: "RohdeSchwarzZNBChannel", **kwargs: Any
    ) -> None:
        super().__init__(
            name,
            instrument=instrument,
            names=("I", "Q"),
            labels=(f"{instrument.short_name} I", f"{instrument.short_name} Q"),
            units=("", ""),
            setpoints=(
                (),
                (),
            ),
            shapes=(
                (),
                (),
            ),
            **kwargs,
        )

    def get_raw(self) -> tuple[float, float]:
        """
        Gets the mean of the raw real and imaginary part of the data. If
        parameter `cw_check_sweep_first` is set to `True` then at the cost of a
        few ms overhead checks if the vna is setup correctly.
        """
        assert isinstance(self.instrument, RohdeSchwarzZNBChannel)
        i, q = self.instrument._get_cw_data()
        return float(np.mean(i)), float(np.mean(q))


class FixedFrequencyPointMagPhase(MultiParameter):
    """
    Parameter for sweep that returns the magnitude of mean of the real (I) and
    imaginary (Q) parts of the VNA response and it's phase.
    Requires the use of the sweep type to be set to continuous wave mode.
    See (https://www.rohde-schwarz.com/webhelp/ZNB_ZNBT_HTML_UserManual_en
    /ZNB_ZNBT_HTML_UserManual_en.htm) under GUI reference -> sweep softtool
    -> sweep type tab -> CW mode

    Args:
        name: parameter name
        instrument: instrument the parameter belongs to
    """

    def __init__(
        self, name: str, instrument: "RohdeSchwarzZNBChannel", **kwargs: Any
    ) -> None:
        super().__init__(
            name,
            instrument=instrument,
            names=("magnitude", "phase"),
            labels=(
                f"{instrument.short_name} magnitude",
                f"{instrument.short_name} phase",
            ),
            units=("", "rad"),
            setpoints=(
                (),
                (),
            ),
            shapes=(
                (),
                (),
            ),
            **kwargs,
        )

    def get_raw(self) -> tuple[float, ...]:
        """
        Gets the magnitude and phase of the mean of the raw real and imaginary
        part of the data. If the parameter `cw_check_sweep_first` is set to
        `True` for the instrument then at the cost of a few ms overhead
        checks if the vna is setup correctly.
        """
        assert isinstance(self.instrument, RohdeSchwarzZNBChannel)
        i, q = self.instrument._get_cw_data()
        s = np.mean(i) + 1j * np.mean(q)
        return float(np.abs(s)), float(np.angle(s))


class FrequencySweepMagPhase(MultiParameter):
    """
    Sweep that return magnitude and phase.
    """

    def __init__(
        self,
        name: str,
        instrument: "RohdeSchwarzZNBChannel",
        start: float,
        stop: float,
        npts: int,
        channel: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name,
            instrument=instrument,
            names=("magnitude", "phase"),
            labels=(
                f"{instrument.short_name} magnitude",
                f"{instrument.short_name} phase",
            ),
            units=("", "rad"),
            setpoint_units=(("Hz",), ("Hz",)),
            setpoint_labels=(
                (f"{instrument.short_name} frequency",),
                (f"{instrument.short_name} frequency",),
            ),
            setpoint_names=(
                (f"{instrument.short_name}_frequency",),
                (f"{instrument.short_name}_frequency",),
            ),
            shapes=(
                (npts,),
                (npts,),
            ),
            **kwargs,
        )
        self.set_sweep(start, stop, npts)
        self._channel = channel

    def set_sweep(self, start: float, stop: float, npts: int) -> None:
        # Needed to update config of the software parameter on sweep change
        # frequency setpoints tuple as needs to be hashable for look up.
        f = tuple(np.linspace(int(start), int(stop), num=npts))
        self.setpoints = ((f,), (f,))
        self.shapes = ((npts,), (npts,))

    def get_raw(self) -> tuple[ParamRawDataType, ...]:
        assert isinstance(self.instrument, RohdeSchwarzZNBChannel)
        with self.instrument.format.set_to("Complex"):
            data = self.instrument._get_sweep_data(force_polar=True)
        return abs(data), np.angle(data)


class FrequencySweepDBPhase(MultiParameter):
    """
    Sweep that return magnitude in decibel (dB) and phase in radians.
    """

    def __init__(
        self,
        name: str,
        instrument: "RohdeSchwarzZNBChannel",
        start: float,
        stop: float,
        npts: int,
        channel: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name,
            instrument=instrument,
            names=("magnitude", "phase"),
            labels=(
                f"{instrument.short_name} magnitude",
                f"{instrument.short_name} phase",
            ),
            units=("dB", "rad"),
            setpoint_units=(("Hz",), ("Hz",)),
            setpoint_labels=(
                (f"{instrument.short_name} frequency",),
                (f"{instrument.short_name} frequency",),
            ),
            setpoint_names=(
                (f"{instrument.short_name}_frequency",),
                (f"{instrument.short_name}_frequency",),
            ),
            shapes=(
                (npts,),
                (npts,),
            ),
            **kwargs,
        )
        self.set_sweep(start, stop, npts)
        self._channel = channel

    def set_sweep(self, start: float, stop: float, npts: int) -> None:
        # Needed to update config of the software parameter on sweep change
        # frequency setpoints tuple as needs to be hashable for look up.
        f = tuple(np.linspace(int(start), int(stop), num=npts))
        self.setpoints = ((f,), (f,))
        self.shapes = ((npts,), (npts,))

    def get_raw(self) -> tuple[ParamRawDataType, ...]:
        assert isinstance(self.instrument, RohdeSchwarzZNBChannel)
        with self.instrument.format.set_to("Complex"):
            data = self.instrument._get_sweep_data(force_polar=True)
        return 20 * np.log10(np.abs(data)), np.angle(data)


class FrequencySweep(ArrayParameter):
    """
    Hardware controlled parameter class for Rohde Schwarz ZNB trace.

    Instrument returns an array of transmission or reflection data depending
    on the active measurement.

    Args:
        name: parameter name
        instrument: instrument the parameter belongs to
        start: starting frequency of sweep
        stop: ending frequency of sweep
        npts: number of points in frequency sweep

    Methods:
          get(): executes a sweep and returns magnitude and phase arrays

    """

    def __init__(
        self,
        name: str,
        instrument: Instrument,
        start: float,
        stop: float,
        npts: int,
        channel: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name,
            shape=(npts,),
            instrument=instrument,
            unit="dB",
            label=f"{instrument.short_name} magnitude",
            setpoint_units=("Hz",),
            setpoint_labels=(f"{instrument.short_name} frequency",),
            setpoint_names=(f"{instrument.short_name}_frequency",),
            **kwargs,
        )
        self.set_sweep(start, stop, npts)
        self._channel = channel

    def set_sweep(self, start: float, stop: float, npts: int) -> None:
        """
        sets the shapes and setpoint arrays of the parameter to
        correspond with the sweep

        Args:
            start: Starting frequency of the sweep
            stop: Stopping frequency of the sweep
            npts: Number of points in the sweep

        """
        # Needed to update config of the software parameter on sweep change
        # freq setpoints tuple as needs to be hashable for look up.
        f = tuple(np.linspace(int(start), int(stop), num=npts))
        self.setpoints = (f,)
        self.shape = (npts,)

    def get_raw(self) -> ParamRawDataType:
        assert isinstance(self.instrument, RohdeSchwarzZNBChannel)
        return self.instrument._get_sweep_data()


class RohdeSchwarzZNBChannel(InstrumentChannel):
    def __init__(
        self,
        parent: "RohdeSchwarzZNBBase",
        name: str,
        channel: int,
        vna_parameter: str | None = None,
        existing_trace_to_bind_to: str | None = None,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        """
        Args:
            parent: Instrument that this channel is bound to.
            name: Name to use for this channel.
            channel: channel on the VNA to use
            vna_parameter: Name of parameter on the vna that this should
                measure such as S12. If left empty this will fall back to
                `name`.
            existing_trace_to_bind_to: Name of an existing trace on the VNA.
                If supplied try to bind to an existing trace with this name
                rather than creating a new trace.
            **kwargs: Forwarded to base class.
        """
        n = channel
        self._instrument_channel = channel

        if vna_parameter is None:
            vna_parameter = name
        self._vna_parameter = vna_parameter
        super().__init__(parent, name, **kwargs)

        if existing_trace_to_bind_to is None:
            self._tracename = f"Trc{channel}"
        else:
            traces = self._parent.ask("CONFigure:TRACe:CATalog?")
            if existing_trace_to_bind_to not in traces:
                raise RuntimeError(
                    f"Trying to bind to"
                    f" {existing_trace_to_bind_to} "
                    f"which is not in {traces}"
                )
            self._tracename = existing_trace_to_bind_to

        # map hardware channel to measurement
        # hardware channels are mapped one to one to QCoDeS channels
        # we are not using sub traces within channels.
        if existing_trace_to_bind_to is None:
            self.write(
                f"CALC{self._instrument_channel}:PAR:SDEF"
                f" '{self._tracename}', '{self._vna_parameter}'"
            )

        # Source power is dependent on model, but not well documented.
        # Here we assume -60 dBm for ZNB20, the others are set,
        # due to lack of knowledge, to -80 dBm as of before the edit.
        full_modelname = self._parent.get_idn()["model"]
        if full_modelname is not None:
            model = full_modelname.split("-")[0]
        else:
            raise RuntimeError("Could not determine ZNB model")
        self._model_min_source_power = {
            "ZNB4": -80,
            "ZNB8": -80,
            "ZNB20": -60,
            "ZNB40": -60,
        }
        if model not in self._model_min_source_power.keys():
            raise RuntimeError(f"Unsupported ZNB model: {model}")
        self._min_source_power: float
        self._min_source_power = self._model_min_source_power[model]

        self.vna_parameter: Parameter = self.add_parameter(
            name="vna_parameter",
            label="VNA parameter",
            get_cmd=f"CALC{self._instrument_channel}:PAR:MEAS? '{self._tracename}'",
            get_parser=self._strip,
        )
        """Parameter vna_parameter"""
        self.power: Parameter = self.add_parameter(
            name="power",
            label="Power",
            unit="dBm",
            get_cmd=f"SOUR{n}:POW?",
            set_cmd=f"SOUR{n}:POW {{:.4f}}",
            get_parser=float,
            vals=vals.Numbers(self._min_source_power, 25),
        )
        """Parameter power"""
        self.bandwidth: Parameter = self.add_parameter(
            name="bandwidth",
            label="Bandwidth",
            unit="Hz",
            get_cmd=f"SENS{n}:BAND?",
            set_cmd=self._set_bandwidth,
            get_parser=int,
            vals=vals.Enum(
                *np.append(10**6, np.kron([1, 1.5, 2, 3, 5, 7], 10 ** np.arange(6)))
            ),
            docstring="Measurement bandwidth of the IF filter. "
            "The inverse of this sets the integration "
            "time per point. "
            "There is an 'increased bandwidth option' "
            "(p. 4 of manual) that does not get taken "
            "into account here.",
        )
        """
        Measurement bandwidth of the IF filter.
        The inverse of this sets the integration time per point.
        There is an 'increased bandwidth option' (p. 4 of manual)
        that does not get taken into account here.
        """
        self.avg: Parameter = self.add_parameter(
            name="avg",
            label="Averages",
            unit="",
            get_cmd=f"SENS{n}:AVER:COUN?",
            set_cmd=f"SENS{n}:AVER:COUN {{:.4f}}",
            get_parser=int,
            vals=vals.Ints(1, 5000),
        )
        """Parameter avg"""
        self.start: Parameter = self.add_parameter(
            name="start",
            get_cmd=f"SENS{n}:FREQ:START?",
            set_cmd=self._set_start,
            get_parser=float,
            vals=vals.Numbers(self._parent._min_freq, self._parent._max_freq - 10),
        )
        """Parameter start"""
        self.stop: Parameter = self.add_parameter(
            name="stop",
            get_cmd=f"SENS{n}:FREQ:STOP?",
            set_cmd=self._set_stop,
            get_parser=float,
            vals=vals.Numbers(self._parent._min_freq + 1, self._parent._max_freq),
        )
        """Parameter stop"""
        self.center: Parameter = self.add_parameter(
            name="center",
            get_cmd=f"SENS{n}:FREQ:CENT?",
            set_cmd=self._set_center,
            get_parser=float,
            vals=vals.Numbers(
                self._parent._min_freq + 0.5, self._parent._max_freq - 10
            ),
        )
        """Parameter center"""
        self.span: Parameter = self.add_parameter(
            name="span",
            get_cmd=f"SENS{n}:FREQ:SPAN?",
            set_cmd=self._set_span,
            get_parser=float,
            vals=vals.Numbers(1, self._parent._max_freq - self._parent._min_freq),
        )
        """Parameter span"""
        self.npts: Parameter = self.add_parameter(
            name="npts",
            get_cmd=f"SENS{n}:SWE:POIN?",
            set_cmd=self._set_npts,
            get_parser=int,
        )
        """Parameter npts"""
        self.status: Parameter = self.add_parameter(
            name="status",
            get_cmd=f"CONF:CHAN{n}:MEAS?",
            set_cmd=f"CONF:CHAN{n}:MEAS {{}}",
            get_parser=int,
        )
        """Parameter status"""
        self.format: Parameter = self.add_parameter(
            name="format",
            get_cmd=partial(self._get_format, tracename=self._tracename),
            set_cmd=self._set_format,
            val_mapping={
                "dB": "MLOG\n",
                "Linear Magnitude": "MLIN\n",
                "Phase": "PHAS\n",
                "Unwr Phase": "UPH\n",
                "Polar": "POL\n",
                "Smith": "SMIT\n",
                "Inverse Smith": "ISM\n",
                "SWR": "SWR\n",
                "Real": "REAL\n",
                "Imaginary": "IMAG\n",
                "Delay": "GDEL\n",
                "Complex": "COMP\n",
            },
        )
        """Parameter format"""

        self.trace_mag_phase: FrequencySweepMagPhase = self.add_parameter(
            name="trace_mag_phase",
            start=self.start(),
            stop=self.stop(),
            npts=self.npts(),
            channel=n,
            parameter_class=FrequencySweepMagPhase,
        )
        """Parameter trace_mag_phase"""

        self.trace_db_phase: FrequencySweepDBPhase = self.add_parameter(
            name="trace_db_phase",
            start=self.start(),
            stop=self.stop(),
            npts=self.npts(),
            channel=n,
            parameter_class=FrequencySweepDBPhase,
        )
        """Parameter trace_db_phase"""
        self.trace: FrequencySweep = self.add_parameter(
            name="trace",
            start=self.start(),
            stop=self.stop(),
            npts=self.npts(),
            channel=n,
            parameter_class=FrequencySweep,
        )
        """Parameter trace"""
        self.electrical_delay: Parameter = self.add_parameter(
            name="electrical_delay",
            label="Electrical delay",
            get_cmd=f"SENS{n}:CORR:EDEL2:TIME?",
            set_cmd=f"SENS{n}:CORR:EDEL2:TIME {{}}",
            get_parser=float,
            unit="s",
        )
        """Parameter electrical_delay"""
        self.sweep_time: Parameter = self.add_parameter(
            name="sweep_time",
            label="Sweep time",
            get_cmd=f"SENS{n}:SWE:TIME?",
            get_parser=float,
            unit="s",
        )
        """Parameter sweep_time"""
        self.sweep_type: Parameter = self.add_parameter(
            name="sweep_type",
            get_cmd=f"SENS{n}:SWE:TYPE?",
            set_cmd=self._set_sweep_type,
            val_mapping={
                "Linear": "LIN\n",
                "Logarithmic": "LOG\n",
                "Power": "POW\n",
                "CW_Time": "CW\n",
                "CW_Point": "POIN\n",
                "Segmented": "SEGM\n",
            },
            docstring="The sweep_type parameter is used to set "
            "the type of measurement sweeps. It "
            "allows switching the default linear "
            "VNA sweep type to other types. Note that "
            "at the moment only the linear and "
            "CW_Point modes have supporting "
            "measurement parameters.",
        )
        """
        The sweep_type parameter is used to set the type of measurement sweeps.
        It allows switching the default linear VNA sweep type to other types.
        Note that at the moment only the linear and CW_Point modes
        have supporting measurement parameters.
        """
        self.cw_frequency: Parameter = self.add_parameter(
            name="cw_frequency",
            get_cmd=f"SENS{n}:FREQ:CW?",
            set_cmd=self._set_cw_frequency,
            get_parser=float,
            vals=vals.Numbers(
                self._parent._min_freq + 0.5, self._parent._max_freq - 10
            ),
            docstring="Parameter for setting frequency and "
            "querying for it when VNA sweep type is "
            "set to CW_Point mode.",
        )
        """
        Parameter for setting frequency and querying for it
        when VNA sweep type is set to CW_Point mode.
        """

        self.cw_check_sweep_first: ManualParameter = self.add_parameter(
            "cw_check_sweep_first",
            parameter_class=ManualParameter,
            initial_value=True,
            vals=vals.Bool(),
            docstring="Parameter that enables a few commands "
            "which are called before each get in "
            "continuous wave mode checking whether "
            "the vna is setup correctly. Is recommended "
            "to be turned, but can be turned off if "
            "one wants to minimize overhead in fast "
            "measurements. ",
        )
        """
        Parameter that enables a few commands which are called before each get
        in continuous wave mode checking whether the vna is setup correctly.
        Is recommended to be turned, but can be turned off if
        one wants to minimize overhead in fast measurements.
        """

        self.trace_fixed_frequency: FixedFrequencyTraceIQ = self.add_parameter(
            name="trace_fixed_frequency",
            npts=self.npts(),
            bandwidth=self.bandwidth(),
            parameter_class=FixedFrequencyTraceIQ,
        )
        """Parameter trace_fixed_frequency"""
        self.point_fixed_frequency: FixedFrequencyPointIQ = self.add_parameter(
            name="point_fixed_frequency", parameter_class=FixedFrequencyPointIQ
        )
        """Parameter point_fixed_frequency"""
        self.point_fixed_frequency_mag_phase: FixedFrequencyPointMagPhase = (
            self.add_parameter(
                name="point_fixed_frequency_mag_phase",
                parameter_class=FixedFrequencyPointMagPhase,
            )
        )
        """Parameter point_fixed_frequency_mag_phase"""
        self.averaging_enabled: Parameter = self.add_parameter(
            name="averaging_enabled",
            initial_value=False,
            get_cmd=None,
            set_cmd=self._enable_averaging,
            vals=vals.Bool(),
            val_mapping=create_on_off_val_mapping(on_val="ON", off_val="OFF"),
        )
        """Parameter averaging_enabled"""
        self.auto_sweep_time_enabled: Parameter = self.add_parameter(
            name="auto_sweep_time_enabled",
            initial_value=False,
            get_cmd=None,
            set_cmd=self._enable_auto_sweep_time,
            vals=vals.Bool(),
            val_mapping=create_on_off_val_mapping(on_val="ON", off_val="OFF"),
            docstring="When enabled, the (minimum) sweep time is "
            "calculated internally using the other channel settings "
            "and zero delay",
        )
        """
        When enabled, the (minimum) sweep time is calculated internally
        using the other channel settings and zero delay
        """

        self.add_function(
            "set_electrical_delay_auto", call_cmd=f"SENS{n}:CORR:EDEL:AUTO ONCE"
        )
        self.add_function(
            "autoscale",
            call_cmd=f"DISPlay:TRACe1:Y:SCALe:AUTO ONCE, {self._tracename}",
        )

    def _get_format(self, tracename: str) -> str:
        n = self._instrument_channel
        self.write(f"CALC{n}:PAR:SEL '{tracename}'")
        return self.ask(f"CALC{n}:FORM?")

    def _set_format(self, val: str) -> None:
        unit_mapping = {
            "MLOG\n": "dB",
            "MLIN\n": "",
            "PHAS\n": "rad",
            "UPH\n": "rad",
            "POL\n": "",
            "SMIT\n": "",
            "ISM\n": "",
            "SWR\n": "U",
            "REAL\n": "U",
            "IMAG\n": "U",
            "GDEL\n": "S",
            "COMP\n": "",
        }
        label_mapping = {
            "MLOG\n": "Magnitude",
            "MLIN\n": "Magnitude",
            "PHAS\n": "Phase",
            "UPH\n": "Unwrapped phase",
            "POL\n": "Complex Magnitude",
            "SMIT\n": "Complex Magnitude",
            "ISM\n": "Complex Magnitude",
            "SWR\n": "Standing Wave Ratio",
            "REAL\n": "Real Magnitude",
            "IMAG\n": "Imaginary Magnitude",
            "GDEL\n": "Delay",
            "COMP\n": "Complex Magnitude",
        }
        channel = self._instrument_channel
        self.write(f"CALC{channel}:PAR:SEL '{self._tracename}'")
        self.write(f"CALC{channel}:FORM {val}")
        self.trace.unit = unit_mapping[val]
        self.trace.label = f"{self.short_name} {label_mapping[val]}"

    @staticmethod
    def _strip(var: str) -> str:
        """Strip newline and quotes from instrument reply."""
        return var.rstrip()[1:-1]

    def _set_start(self, val: float) -> None:
        channel = self._instrument_channel
        self.write(f"SENS{channel}:FREQ:START {val:.7f}")
        stop = self.stop()
        if val >= stop:
            raise ValueError("Stop frequency must be larger than start frequency.")
        # we get start as the vna may not be able to set it to the
        # exact value provided.
        start = self.start()
        if abs(val - start) >= 1:
            log.warning(f"Could not set start to {val} setting it to {start}")
        self.update_lin_traces()

    def _set_stop(self, val: float) -> None:
        channel = self._instrument_channel
        start = self.start()
        if val <= start:
            raise ValueError("Stop frequency must be larger than start frequency.")
        self.write(f"SENS{channel}:FREQ:STOP {val:.7f}")
        # We get stop as the vna may not be able to set it to the
        # exact value provided.
        stop = self.stop()
        if abs(val - stop) >= 1:
            log.warning(f"Could not set stop to {val} setting it to {stop}")
        self.update_lin_traces()

    def _set_npts(self, val: int) -> None:
        channel = self._instrument_channel
        self.write(f"SENS{channel}:SWE:POIN {val:.7f}")
        if self.sweep_type().startswith("CW"):
            self.update_cw_traces()
        else:
            self.update_lin_traces()

    def _set_bandwidth(self, val: int) -> None:
        channel = self._instrument_channel
        self.write(f"SENS{channel}:BAND {val:.4f}")
        self.update_cw_traces()

    def _set_span(self, val: float) -> None:
        channel = self._instrument_channel
        self.write(f"SENS{channel}:FREQ:SPAN {val:.7f}")
        self.update_lin_traces()

    def _set_center(self, val: float) -> None:
        channel = self._instrument_channel
        self.write(f"SENS{channel}:FREQ:CENT {val:.7f}")
        self.update_lin_traces()

    def _set_sweep_type(self, val: str) -> None:
        channel = self._instrument_channel
        self.write(f"SENS{channel}:SWE:TYPE {val}")

    def _set_cw_frequency(self, val: float) -> None:
        channel = self._instrument_channel
        self.write(f"SENS{channel}:FREQ:CW {val:.7f}")

    def _enable_averaging(self, val: str) -> None:
        channel = self._instrument_channel
        self.write(f"SENS{channel}:AVER:STAT {val}")

    def _enable_auto_sweep_time(self, val: str) -> None:
        channel = self._instrument_channel
        self.write(f"SENS{channel}:SWE:TIME:AUTO {val}")

    def update_lin_traces(self) -> None:
        """
        Updates start, stop and npts of all trace parameters
        so that the x-coordinates are updated for the sweep.
        """
        start = self.start()
        stop = self.stop()
        npts = self.npts()
        for _, parameter in self.parameters.items():
            if isinstance(
                parameter,
                (FrequencySweep, FrequencySweepMagPhase, FrequencySweepDBPhase),
            ):
                try:
                    parameter.set_sweep(start, stop, npts)
                except AttributeError:
                    pass

    def update_cw_traces(self) -> None:
        """
        Updates the bandwidth and npts of all fixed frequency (CW) traces.
        """
        bandwidth = self.bandwidth()
        npts = self.npts()
        for _, parameter in self.parameters.items():
            if isinstance(parameter, FixedFrequencyTraceIQ):
                try:
                    parameter.set_cw_sweep(npts, bandwidth)
                except AttributeError:
                    pass
        self.sweep_time()

    def _get_sweep_data(self, force_polar: bool = False) -> np.ndarray:
        if not self._parent.rf_power():
            log.warning("RF output is off when getting sweep data")
        # It is possible that the instrument and QCoDeS disagree about
        # which parameter is measured on this channel.
        instrument_parameter = self.vna_parameter()
        if instrument_parameter != self._vna_parameter:
            raise RuntimeError(
                "Invalid parameter. Tried to measure "
                f"{self._vna_parameter} "
                f"got {instrument_parameter}"
            )
        self.averaging_enabled(True)
        self.write(f"SENS{self._instrument_channel}:AVER:CLE")

        # preserve original state of the znb
        with self.status.set_to(1):
            self.root_instrument.cont_meas_off()
            try:
                # if force polar is set, the SDAT data format will be used.
                # Here the data will be transferred as a complex number
                # independent of the set format in the instrument.
                if force_polar:
                    data_format_command = "SDAT"
                else:
                    data_format_command = "FDAT"

                with self.root_instrument.timeout.set_to(self._get_timeout()):
                    # instrument averages over its last 'avg' number of sweeps
                    # need to ensure averaged result is returned
                    for _ in range(self.avg()):
                        self.write(f"INIT{self._instrument_channel}:IMM; *WAI")
                    self.write(
                        f"CALC{self._instrument_channel}:PAR:SEL "
                        f"'{self._tracename}'"
                    )
                    data_str = self.ask(
                        f"CALC{self._instrument_channel}:DATA?"
                        f" {data_format_command}"
                    )
                data = np.array(data_str.rstrip().split(",")).astype("float64")
                if self.format() in ["Polar", "Complex", "Smith", "Inverse Smith"]:
                    data = data[0::2] + 1j * data[1::2]
            finally:
                self.root_instrument.cont_meas_on()
        return data

    def setup_cw_sweep(self) -> None:
        """
        This method sets the VNA to CW mode. CW Mode sweeps are performed at
        fixed frequency and allow to perform measurements versus time instead
        of versus frequency.
        See (https://www.rohde-schwarz.com/webhelp/ZNB_ZNBT_HTML_UserManual_en
        /ZNB_ZNBT_HTML_UserManual_en.htm) under GUI reference -> sweep softtool
        -> sweep type tab -> CW mode
        """

        # set the channel type to single point msmt
        self.sweep_type("CW_Point")
        # turn off average on the VNA since we want single point sweeps.
        self.averaging_enabled(False)
        # This format is required for getting both real and imaginary parts.
        self.format("Complex")
        # Set the sweep time to auto such that it sets the delay to zero
        # between each point (e.g msmt speed is optimized). Note that if one
        # would like to do a time sweep with time > npts/bandwidth, this is
        # where the delay would be set, but in general we want to measure as
        # fast as possible without artificial delays.
        self.auto_sweep_time_enabled(True)
        # Set cont measurement off here so we don't have to send that command
        # while measuring later.
        self.root_instrument.cont_meas_off()

    def setup_lin_sweep(self) -> None:
        """
        Setup the instrument into linear sweep mode.
        """
        self.sweep_type("Linear")
        self.averaging_enabled(True)
        self.root_instrument.cont_meas_on()

    def _check_cw_sweep(self) -> None:
        """
        Checks if all required settings are met to be able to measure in
        CW_point mode. Similar to what is done in get_sweep_data
        """
        if self.sweep_type() != "CW_Point":
            raise RuntimeError(
                f"Sweep type is not set to continuous wave "
                f"mode, instead it is: {self.sweep_type()}"
            )

        if not self.root_instrument.rf_power():
            log.warning("RF output is off when getting sweep data")

        # It is possible that the instrument and QCoDeS disagree about
        # which parameter is measured on this channel.
        instrument_parameter = self.vna_parameter()
        if instrument_parameter != self._vna_parameter:
            raise RuntimeError(
                "Invalid parameter. Tried to measure "
                f"{self._vna_parameter} "
                f"got {instrument_parameter}"
            )

        # Turn off average on the VNA since we want single point sweeps.
        self.averaging_enabled(False)
        # Set the format to complex.
        self.format("Complex")
        # Set cont measurement off.
        self.root_instrument.cont_meas_off()
        # Cache the sweep time so it is up to date when setting timeouts
        self.sweep_time()

    def _get_cw_data(self) -> tuple[np.ndarray, np.ndarray]:
        # Make the checking optional such that we can do super fast sweeps as
        # well, skipping the overhead of the other commands.
        if self.cw_check_sweep_first():
            self._check_cw_sweep()

        with self.status.set_to(1):
            with self.root_instrument.timeout.set_to(self._get_timeout()):
                self.write(f"INIT{self._instrument_channel}:IMM; *WAI")
                data_str = self.ask(f"CALC{self._instrument_channel}:DATA? SDAT")
            data = np.array(data_str.rstrip().split(",")).astype("float64")
            i = data[0::2]
            q = data[1::2]

        return i, q

    def _get_timeout(self) -> float:
        timeout = self.root_instrument.timeout() or float("+inf")
        timeout = max(self.sweep_time.cache.get() * 1.5, timeout)
        return timeout


ZNBChannel = RohdeSchwarzZNBChannel


class RohdeSchwarzZNBBase(VisaInstrument):
    """
    Base class for QCoDeS driver for the Rohde & Schwarz ZNB8 and ZNB20
    virtual network analyser. It can probably be extended to ZNB4 and 40
    without too much work. This class should not be instantiated directly
    the RohdeSchwarzZNB8 and RohdeSchwarzZNB20 should be used instead.

    Requires FrequencySweep parameter for taking a trace

    Args:
        name: instrument name
        address: Address of instrument probably in format
            'TCPIP0::192.168.15.100::inst0::INSTR'
        init_s_params: Automatically setup channels for all S parameters on the
            VNA.
        reset_channels: If True any channels defined on the VNA at the time
            of initialization are reset and removed.
        **kwargs: passed to base class

    Todo:
        - check initialisation settings and test functions
    """

    CHANNEL_CLASS = ZNBChannel

    def __init__(
        self,
        name: str,
        address: str,
        init_s_params: bool = True,
        reset_channels: bool = True,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ) -> None:
        super().__init__(name=name, address=address, **kwargs)

        # TODO(JHN) I could not find a way to get max and min freq from
        # the API, if that is possible replace below with that
        # See page 1025 in the manual. 7.3.15.10 for details of max/min freq
        # no attempt to support ZNB40, not clear without one how the format
        # is due to variants
        fullmodel = self.get_idn()["model"]
        if fullmodel is not None:
            model = fullmodel.split("-")[0]
        else:
            raise RuntimeError("Could not determine ZNB model")
        # format seems to be ZNB8-4Port
        m_frequency = {
            "ZNB4": (9e3, 4.5e9),
            "ZNB8": (9e3, 8.5e9),
            "ZNB20": (100e3, 20e9),
            "ZNB40": (10e6, 40e9),
        }
        if model not in m_frequency.keys():
            raise RuntimeError(f"Unsupported ZNB model {model}")
        self._min_freq: float
        self._max_freq: float
        self._min_freq, self._max_freq = m_frequency[model]

        self.num_ports: Parameter = self.add_parameter(
            name="num_ports", get_cmd="INST:PORT:COUN?", get_parser=int
        )
        """Parameter num_ports"""
        num_ports = self.num_ports()

        self.rf_power: Parameter = self.add_parameter(
            name="rf_power",
            get_cmd="OUTP1?",
            set_cmd="OUTP1 {}",
            val_mapping={True: "1\n", False: "0\n"},
        )
        """Parameter rf_power"""

        self.ref_osc_source: Parameter = self.add_parameter(
            name="ref_osc_source",
            label="Reference oscillator source",
            get_cmd="ROSC:SOUR?",
            set_cmd="ROSC:SOUR {}",
            # strip newline
            get_parser=lambda s: s.rstrip(),
            vals=vals.Enum("INT", "EXT", "int", "ext", "internal", "external"),
        )
        """Reference oscillator source"""

        self.ref_osc_external_freq: Parameter = self.add_parameter(
            name="ref_osc_external_freq",
            label="Reference oscillator frequency",
            docstring="Frequency of the external reference clock signal at REF IN",
            get_cmd="ROSC:EXT:FREQ?",
            set_cmd="ROSC:EXT:FREQ {}Hz",
            # The response contains the unit (Hz), so we have to strip it
            get_parser=lambda f: float(f.strip("Hz")),
            unit="Hz",
            # Data sheet: 1 MHz to 20 MHz, in steps of 1 MHz
            vals=vals.Enum(*np.linspace(1e6, 20e6, 20)),
        )
        """Frequency of the external reference clock signal at REF IN"""

        self.ref_osc_PLL_locked: Parameter = self.add_parameter(
            name="ref_osc_PLL_locked",
            label="Reference frequency PLL lock",
            get_cmd=self._get_PLL_locked,
            docstring="If an external reference signal or an internal high "
            "precision clock (option B4) is used, the local oscillator is "
            "phase locked to a reference signal. This parameter will be "
            "False if the phase locked loop (PLL) fails. "
            "\n"
            "For external reference: check frequency and level of the "
            "supplied reference signal.",
        )
        """
        If an external reference signal or an internal high precision clock
        (option B4) is used, the local oscillator is phase locked to a
        reference signal. This parameter will be False if the phase locked loop
        (PLL) fails.

        For external reference: check frequency and level of the supplied
        reference signal.
        """

        self.add_function("reset", call_cmd="*RST")
        self.add_function("tooltip_on", call_cmd="SYST:ERR:DISP ON")
        self.add_function("tooltip_off", call_cmd="SYST:ERR:DISP OFF")
        self.add_function("cont_meas_on", call_cmd="INIT:CONT:ALL ON")
        self.add_function("cont_meas_off", call_cmd="INIT:CONT:ALL OFF")
        self.add_function("update_display_once", call_cmd="SYST:DISP:UPD ONCE")
        self.add_function("update_display_on", call_cmd="SYST:DISP:UPD ON")
        self.add_function("update_display_off", call_cmd="SYST:DISP:UPD OFF")
        self.add_function(
            "display_sij_split",
            call_cmd=f"DISP:LAY GRID;:DISP:LAY:GRID {num_ports},{num_ports}",
        )
        self.add_function(
            "display_single_window", call_cmd="DISP:LAY GRID;:DISP:LAY:GRID 1,1"
        )
        self.add_function(
            "display_dual_window", call_cmd="DISP:LAY GRID;:DISP:LAY:GRID 2,1"
        )
        self.add_function("rf_off", call_cmd="OUTP1 OFF")
        self.add_function("rf_on", call_cmd="OUTP1 ON")
        if reset_channels:
            self.reset()
            self.clear_channels()
        channels = ChannelList(
            self, "VNAChannels", self.CHANNEL_CLASS, snapshotable=True
        )
        self.add_submodule("channels", channels)
        if init_s_params:
            for i in range(1, num_ports + 1):
                for j in range(1, num_ports + 1):
                    ch_name = "S" + str(i) + str(j)
                    self.add_channel(ch_name)
            self.display_sij_split()
            self.channels.autoscale()

        self.update_display_on()
        if reset_channels:
            self.rf_off()
        self.connect_message()

    def _get_PLL_locked(self) -> bool:
        # query the bits of the "questionable hardware integrity" register
        hw_integrity_bits = int(self.ask("STATus:QUEStionable:INTegrity:HARDware?"))
        # if bit number 1 is set, the PLL locking has failed
        pll_lock_failed = bool(hw_integrity_bits & 0b10)
        return not pll_lock_failed

    def display_grid(self, rows: int, cols: int) -> None:
        """
        Display a grid of channels rows by columns.
        """
        self.write(f"DISP:LAY GRID;:DISP:LAY:GRID {rows},{cols}")

    def add_channel(self, channel_name: str, **kwargs: Any) -> None:
        i_channel = len(self.channels) + 1
        channel = self.CHANNEL_CLASS(self, channel_name, i_channel, **kwargs)
        self.channels.append(channel)
        if i_channel == 1:
            self.display_single_window()
        if i_channel == 2:
            self.display_dual_window()
        # shortcut
        setattr(self, channel_name, channel)
        # initialising channel
        self.write(f"SENS{i_channel}:SWE:TYPE LIN")
        self.write(f"SENS{i_channel}:SWE:TIME:AUTO ON")
        self.write(f"TRIG{i_channel}:SEQ:SOUR IMM")
        self.write(f"SENS{i_channel}:AVER:STAT ON")

    def clear_channels(self) -> None:
        """
        Remove all channels from the instrument and channel list and
        unlock the channel list.
        """
        self.write("CALCulate:PARameter:DELete:ALL")
        for submodule in self.submodules.values():
            if isinstance(submodule, ChannelList):
                submodule.clear()


@deprecated(
    "The ZNB base class has been renamed RohdeSchwarzZNBBase",
    category=QCoDeSDeprecationWarning,
)
class ZNB(RohdeSchwarzZNBBase):
    pass
