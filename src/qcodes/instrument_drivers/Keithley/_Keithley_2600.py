from __future__ import annotations

import logging
import struct
import warnings
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

import qcodes.validators as vals
from qcodes.instrument import (
    Instrument,
    InstrumentChannel,
    VisaInstrument,
    VisaInstrumentKWArgs,
)
from qcodes.parameters import (
    ArrayParameter,
    Parameter,
    ParameterWithSetpoints,
    ParamRawDataType,
    create_on_off_val_mapping,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qcodes_loop.data.data_set import DataSet
    from typing_extensions import Unpack


log = logging.getLogger(__name__)


class LuaSweepParameter(ArrayParameter):
    """
    Parameter class to hold the data from a
    deployed Lua script sweep.
    """

    def __init__(self, name: str, instrument: Instrument, **kwargs: Any) -> None:
        super().__init__(
            name=name,
            shape=(1,),
            docstring="Holds a sweep",
            instrument=instrument,
            **kwargs,
        )

    def prepareSweep(self, start: float, stop: float, steps: int, mode: str) -> None:
        """
        Builds setpoints and labels

        Args:
            start: Starting point of the sweep
            stop: Endpoint of the sweep
            steps: No. of sweep steps
            mode: Type of sweep, either 'IV' (voltage sweep),
                'VI' (current sweep two probe setup) or
                'VIfourprobe' (current sweep four probe setup)

        """

        if mode not in ["IV", "VI", "VIfourprobe"]:
            raise ValueError('mode must be either "VI", "IV" or "VIfourprobe"')

        self.shape = (steps,)

        if mode == "IV":
            self.unit = "A"
            self.setpoint_names = ("Voltage",)
            self.setpoint_units = ("V",)
            self.label = "current"
            self._short_name = "iv_sweep"

        if mode == "VI":
            self.unit = "V"
            self.setpoint_names = ("Current",)
            self.setpoint_units = ("A",)
            self.label = "voltage"
            self._short_name = "vi_sweep"

        if mode == "VIfourprobe":
            self.unit = "V"
            self.setpoint_names = ("Current",)
            self.setpoint_units = ("A",)
            self.label = "voltage"
            self._short_name = "vi_sweep_four_probe"

        self.setpoints = (tuple(np.linspace(start, stop, steps)),)

        self.start = start
        self.stop = stop
        self.steps = steps
        self.mode = mode

    def get_raw(self) -> npt.NDArray:
        if self.instrument is not None:
            data = self.instrument._fast_sweep(
                self.start, self.stop, self.steps, self.mode
            )
        else:
            raise RuntimeError("No instrument attached to Parameter.")

        return data


class TimeTrace(ParameterWithSetpoints):
    """
    A parameter class that holds the data corresponding to the time dependence of
    current and voltage.
    """

    def _check_time_trace(self) -> None:
        """
        A helper function that compares the integration time with measurement
        interval for accurate results.

        Raises:
            RuntimeError: If no instrument attached to Parameter.

        """
        if self.instrument is None:
            raise RuntimeError("No instrument attached to Parameter.")

        dt = self.instrument.timetrace_dt()
        nplc = self.instrument.nplc()
        linefreq = self.instrument.linefreq()
        plc = 1 / linefreq
        if nplc * plc > dt:
            warnings.warn(
                f"Integration time of {nplc * plc * 1000:.1f} "
                f"ms is longer than {dt * 1000:.1f} ms set "
                "as measurement interval. Consider lowering "
                "NPLC or increasing interval.",
                UserWarning,
                2,
            )

    def _set_mode(self, mode: str) -> None:
        """
        A helper function to set correct units and labels.

        Args:
            mode: User defined mode for the timetrace. It can be either
            "current" or "voltage".

        """
        if mode == "current":
            self.unit = "A"
            self.label = "Current"
        if mode == "voltage":
            self.unit = "V"
            self.label = "Voltage"

    def _time_trace(self) -> npt.NDArray:
        """
        The function that prepares a Lua script for timetrace data acquisition.

        Raises:
            RuntimeError: If no instrument attached to Parameter.

        """

        if self.instrument is None:
            raise RuntimeError("No instrument attached to Parameter.")

        channel = self.instrument.channel
        npts = self.instrument.timetrace_npts()
        dt = self.instrument.timetrace_dt()
        mode = self.instrument.timetrace_mode()

        mode_map = {"current": "i", "voltage": "v"}

        script = [
            f"{channel}.measure.count={npts}",
            f"oldint={channel}.measure.interval",
            f"{channel}.measure.interval={dt}",
            f"{channel}.nvbuffer1.clear()",
            f"{channel}.measure.{mode_map[mode]}({channel}.nvbuffer1)",
            f"{channel}.measure.interval=oldint",
            f"{channel}.measure.count=1",
            "format.data = format.REAL32",
            "format.byteorder = format.LITTLEENDIAN",
            f"printbuffer(1, {npts}, {channel}.nvbuffer1.readings)",
        ]

        return self.instrument._execute_lua(script, npts)

    def get_raw(self) -> npt.NDArray:
        if self.instrument is None:
            raise RuntimeError("No instrument attached to Parameter.")

        self._check_time_trace()
        data = self._time_trace()
        return data


class TimeAxis(Parameter):
    """
    A simple :class:`.Parameter` that holds all the times (relative to the
    measurement start) at which the points of the time trace were acquired.
    """

    def get_raw(self) -> npt.NDArray:
        if self.instrument is None:
            raise RuntimeError("No instrument attached to Parameter.")

        npts = self.instrument.timetrace_npts()
        dt = self.instrument.timetrace_dt()
        return np.linspace(0, dt * npts, npts, endpoint=False)


class Keithley2600MeasurementStatus(StrEnum):
    """
    Keeps track of measurement status.
    """

    CURRENT_COMPLIANCE_ERROR = "Reached current compliance limit."
    VOLTAGE_COMPLIANCE_ERROR = "Reached voltage compliance limit."
    VOLTAGE_AND_CURRENT_COMPLIANCE_ERROR = (
        "Reached both voltage and current compliance limits."
    )
    NORMAL = "No error occured."
    COMPLIANCE_ERROR = "Reached compliance limit."  # deprecated, dont use it. It exists only for backwards compatibility


MeasurementStatus = Keithley2600MeasurementStatus
"Alias for backwards compatibility. Will eventually be deprecated and removed"

_from_bits_tuple_to_status = {
    (0, 0): Keithley2600MeasurementStatus.NORMAL,
    (1, 0): Keithley2600MeasurementStatus.VOLTAGE_COMPLIANCE_ERROR,
    (0, 1): Keithley2600MeasurementStatus.CURRENT_COMPLIANCE_ERROR,
    (1, 1): Keithley2600MeasurementStatus.VOLTAGE_AND_CURRENT_COMPLIANCE_ERROR,
}


class _ParameterWithStatus(Parameter):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self._measurement_status: Keithley2600MeasurementStatus | None = None

    @property
    def measurement_status(self) -> Keithley2600MeasurementStatus | None:
        return self._measurement_status

    @staticmethod
    def _parse_response(data: str) -> tuple[float, Keithley2600MeasurementStatus]:
        value, meas_status = data.split("\t")

        status_bits = [
            int(i)
            for i in bin(int(float(meas_status))).replace("0b", "").zfill(16)[::-1]
        ]

        status = _from_bits_tuple_to_status[(status_bits[0], status_bits[1])]  # pyright: ignore[reportArgumentType]

        return float(value), status

    def snapshot_base(
        self,
        update: bool | None = True,
        params_to_skip_update: Sequence[str] | None = None,
    ) -> dict[Any, Any]:
        snapshot = super().snapshot_base(
            update=update, params_to_skip_update=params_to_skip_update
        )

        if self._snapshot_value:
            snapshot["measurement_status"] = self.measurement_status

        return snapshot


class _MeasurementCurrentParameter(_ParameterWithStatus):
    def set_raw(self, value: ParamRawDataType) -> None:
        assert isinstance(self.instrument, Keithley2600Channel)
        assert isinstance(self.root_instrument, Keithley2600)

        smu_chan = self.instrument
        channel = smu_chan.channel

        smu_chan.write(f"{channel}.source.leveli={value:.12f}")

        smu_chan._reset_measurement_statuses_of_parameters()

    def get_raw(self) -> ParamRawDataType:
        assert isinstance(self.instrument, Keithley2600Channel)
        assert isinstance(self.root_instrument, Keithley2600)

        smu = self.instrument
        channel = self.instrument.channel

        data = smu.ask(
            f"{channel}.measure.i(), status.measurement.instrument.{channel}.condition"
        )
        value, status = self._parse_response(data)

        self._measurement_status = status

        return value


class _MeasurementVoltageParameter(_ParameterWithStatus):
    def set_raw(self, value: ParamRawDataType) -> None:
        assert isinstance(self.instrument, Keithley2600Channel)
        assert isinstance(self.root_instrument, Keithley2600)

        smu_chan = self.instrument
        channel = smu_chan.channel

        smu_chan.write(f"{channel}.source.levelv={value:.12f}")

        smu_chan._reset_measurement_statuses_of_parameters()

    def get_raw(self) -> ParamRawDataType:
        assert isinstance(self.instrument, Keithley2600Channel)
        assert isinstance(self.root_instrument, Keithley2600)

        smu = self.instrument
        channel = self.instrument.channel

        data = smu.ask(
            f"{channel}.measure.v(), status.measurement.instrument.{channel}.condition"
        )
        value, status = self._parse_response(data)

        self._measurement_status = status

        return value


class Keithley2600Channel(InstrumentChannel):
    """
    Class to hold the two Keithley channels, i.e.
    SMUA and SMUB.
    """

    def __init__(self, parent: Instrument, name: str, channel: str) -> None:
        """
        Args:
            parent: The Instrument instance to which the channel is
                to be attached.
            name: The 'colloquial' name of the channel
            channel: The name used by the Keithley, i.e. either
                'smua' or 'smub'

        """

        if channel not in ["smua", "smub"]:
            raise ValueError('channel must be either "smub" or "smua"')

        super().__init__(parent, name)
        self.model = self._parent.model
        self._extra_visa_timeout = 5000
        self._measurement_duration_factor = 2  # Ensures that we are always above
        # the expected time.
        vranges = self._parent._vranges
        iranges = self._parent._iranges
        vlimit_minmax = self.parent._vlimit_minmax
        ilimit_minmax = self.parent._ilimit_minmax

        self.volt: _MeasurementVoltageParameter = self.add_parameter(
            "volt",
            parameter_class=_MeasurementVoltageParameter,
            label="Voltage",
            unit="V",
            snapshot_get=False,
        )
        """Parameter volt"""

        self.curr: _MeasurementCurrentParameter = self.add_parameter(
            "curr",
            parameter_class=_MeasurementCurrentParameter,
            label="Current",
            unit="A",
            snapshot_get=False,
        )
        """Parameter curr"""

        self.res: Parameter = self.add_parameter(
            "res",
            get_cmd=f"{channel}.measure.r()",
            get_parser=float,
            set_cmd=False,
            label="Resistance",
            unit="Ohm",
        )
        """Parameter res"""

        self.mode: Parameter = self.add_parameter(
            "mode",
            get_cmd=f"{channel}.source.func",
            get_parser=float,
            set_cmd=f"{channel}.source.func={{:d}}",
            val_mapping={"current": 0, "voltage": 1},
            docstring="Selects the output source type. "
            "Can be either voltage or current.",
        )
        """Selects the output source type. Can be either voltage or current."""

        self.output: Parameter = self.add_parameter(
            "output",
            get_cmd=f"{channel}.source.output",
            get_parser=float,
            set_cmd=f"{channel}.source.output={{:d}}",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
        )
        """Parameter output"""

        self.linefreq: Parameter = self.add_parameter(
            "linefreq",
            label="Line frequency",
            get_cmd="localnode.linefreq",
            get_parser=float,
            set_cmd=False,
            unit="Hz",
        )
        """Parameter linefreq"""

        self.nplc: Parameter = self.add_parameter(
            "nplc",
            label="Number of power line cycles",
            set_cmd=f"{channel}.measure.nplc={{}}",
            get_cmd=f"{channel}.measure.nplc",
            get_parser=float,
            docstring="Number of power line cycles, used to perform measurements",
            vals=vals.Numbers(0.001, 25),
        )
        """Number of power line cycles, used to perform measurements"""
        # volt range
        # needs get after set (WilliamHPNielsen): why?
        self.sourcerange_v: Parameter = self.add_parameter(
            "sourcerange_v",
            label="voltage source range",
            get_cmd=f"{channel}.source.rangev",
            get_parser=float,
            set_cmd=self._set_sourcerange_v,
            unit="V",
            docstring="The range used when sourcing voltage "
            "This affects the range and the precision "
            "of the source.",
            vals=vals.Enum(*vranges[self.model]),
        )
        """The range used when sourcing voltage This affects the range and the precision of the source."""

        self.source_autorange_v_enabled: Parameter = self.add_parameter(
            "source_autorange_v_enabled",
            label="voltage source autorange",
            get_cmd=f"{channel}.source.autorangev",
            get_parser=float,
            set_cmd=f"{channel}.source.autorangev={{}}",
            docstring="Set autorange on/off for source voltage.",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
        )
        """Set autorange on/off for source voltage."""

        self.measurerange_v: Parameter = self.add_parameter(
            "measurerange_v",
            label="voltage measure range",
            get_cmd=f"{channel}.measure.rangev",
            get_parser=float,
            set_cmd=self._set_measurerange_v,
            unit="V",
            docstring="The range to perform voltage "
            "measurements in. This affects the range "
            "and the precision of the measurement. "
            "Note that if you both measure and "
            "source current this will have no effect, "
            "set `sourcerange_v` instead",
            vals=vals.Enum(*vranges[self.model]),
        )
        """
        The range to perform voltage measurements in. This affects the range and the precision of the measurement.
        Note that if you both measure and source current this will have no effect, set `sourcerange_v` instead
        """

        self.measure_autorange_v_enabled: Parameter = self.add_parameter(
            "measure_autorange_v_enabled",
            label="voltage measure autorange",
            get_cmd=f"{channel}.measure.autorangev",
            get_parser=float,
            set_cmd=f"{channel}.measure.autorangev={{}}",
            docstring="Set autorange on/off for measure voltage.",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
        )
        """Set autorange on/off for measure voltage."""
        # current range
        # needs get after set
        self.sourcerange_i: Parameter = self.add_parameter(
            "sourcerange_i",
            label="current source range",
            get_cmd=f"{channel}.source.rangei",
            get_parser=float,
            set_cmd=self._set_sourcerange_i,
            unit="A",
            docstring="The range used when sourcing current "
            "This affects the range and the "
            "precision of the source.",
            vals=vals.Enum(*iranges[self.model]),
        )
        """The range used when sourcing current This affects the range and the precision of the source."""

        self.source_autorange_i_enabled: Parameter = self.add_parameter(
            "source_autorange_i_enabled",
            label="current source autorange",
            get_cmd=f"{channel}.source.autorangei",
            get_parser=float,
            set_cmd=f"{channel}.source.autorangei={{}}",
            docstring="Set autorange on/off for source current.",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
        )
        """Set autorange on/off for source current."""

        self.measurerange_i: Parameter = self.add_parameter(
            "measurerange_i",
            label="current measure range",
            get_cmd=f"{channel}.measure.rangei",
            get_parser=float,
            set_cmd=self._set_measurerange_i,
            unit="A",
            docstring="The range to perform current "
            "measurements in. This affects the range "
            "and the precision of the measurement. "
            "Note that if you both measure and source "
            "current this will have no effect, set "
            "`sourcerange_i` instead",
            vals=vals.Enum(*iranges[self.model]),
        )
        """
        The range to perform current measurements in. This affects the range and the precision of the measurement.
        Note that if you both measure and source current this will have no effect, set `sourcerange_i` instead"""

        self.measure_autorange_i_enabled: Parameter = self.add_parameter(
            "measure_autorange_i_enabled",
            label="current autorange",
            get_cmd=f"{channel}.measure.autorangei",
            get_parser=float,
            set_cmd=f"{channel}.measure.autorangei={{}}",
            docstring="Set autorange on/off for measure current.",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
        )
        """Set autorange on/off for measure current."""
        # Compliance limit
        self.limitv: Parameter = self.add_parameter(
            "limitv",
            get_cmd=f"{channel}.source.limitv",
            get_parser=float,
            set_cmd=f"{channel}.source.limitv={{}}",
            docstring="Voltage limit e.g. the maximum voltage "
            "allowed in current mode. If exceeded "
            "the current will be clipped.",
            vals=vals.Numbers(
                vlimit_minmax[self.model][0], vlimit_minmax[self.model][1]
            ),
            unit="V",
        )
        """Voltage limit e.g. the maximum voltage allowed in current mode. If exceeded the current will be clipped."""
        # Compliance limit
        self.limiti: Parameter = self.add_parameter(
            "limiti",
            get_cmd=f"{channel}.source.limiti",
            get_parser=float,
            set_cmd=f"{channel}.source.limiti={{}}",
            docstring="Current limit e.g. the maximum current "
            "allowed in voltage mode. If exceeded "
            "the voltage will be clipped.",
            vals=vals.Numbers(
                ilimit_minmax[self.model][0], ilimit_minmax[self.model][1]
            ),
            unit="A",
        )
        """Current limit e.g. the maximum current allowed in voltage mode. If exceeded the voltage will be clipped."""

        self.fastsweep: LuaSweepParameter = self.add_parameter(
            "fastsweep", parameter_class=LuaSweepParameter
        )
        """Parameter fastsweep"""

        self.timetrace_npts: Parameter = self.add_parameter(
            "timetrace_npts",
            initial_value=500,
            label="Number of points",
            get_cmd=None,
            set_cmd=None,
        )
        """Parameter timetrace_npts"""

        self.timetrace_dt: Parameter = self.add_parameter(
            "timetrace_dt",
            initial_value=1e-3,
            label="Time resolution",
            unit="s",
            get_cmd=None,
            set_cmd=None,
        )
        """Parameter timetrace_dt"""

        self.time_axis: TimeAxis = self.add_parameter(
            name="time_axis",
            label="Time",
            unit="s",
            snapshot_value=False,
            vals=vals.Arrays(shape=(self.timetrace_npts,)),
            parameter_class=TimeAxis,
        )
        """Parameter time_axis"""

        self.timetrace: TimeTrace = self.add_parameter(
            "timetrace",
            vals=vals.Arrays(shape=(self.timetrace_npts,)),
            setpoints=(self.time_axis,),
            parameter_class=TimeTrace,
        )
        """Parameter timetrace"""

        self.timetrace_mode: Parameter = self.add_parameter(
            "timetrace_mode",
            initial_value="current",
            get_cmd=None,
            set_cmd=self.timetrace._set_mode,
            vals=vals.Enum("current", "voltage"),
        )
        """Parameter timetrace_mode"""

        self.channel = channel

    def _reset_measurement_statuses_of_parameters(self) -> None:
        assert isinstance(self.volt, _ParameterWithStatus)
        self.volt._measurement_status = None
        assert isinstance(self.curr, _ParameterWithStatus)
        self.curr._measurement_status = None

    def reset(self) -> None:
        """
        Reset instrument to factory defaults.
        This resets only the relevant channel.
        """
        self.write(f"{self.channel}.reset()")
        # remember to update all the metadata
        log.debug(f"Reset channel {self.channel}. Updating settings...")
        self.snapshot(update=True)

    def doFastSweep(self, start: float, stop: float, steps: int, mode: str) -> DataSet:
        """
        Perform a fast sweep using a deployed lua script and
        return a QCoDeS DataSet with the sweep.

        Args:
            start: starting sweep value (V or A)
            stop: end sweep value (V or A)
            steps: number of steps
            mode: Type of sweep, either 'IV' (voltage sweep),
                'VI' (current sweep two probe setup) or
                'VIfourprobe' (current sweep four probe setup)

        """
        try:
            # lazy import to avoid a geneal dependency on qcodes_loop
            from qcodes_loop.measure import Measure
        except ImportError as e:
            raise ImportError(
                "The doFastSweep method requires the "
                "qcodes_loop package to be installed."
            ) from e
        # prepare setpoints, units, name
        self.fastsweep.prepareSweep(start, stop, steps, mode)

        data = Measure(self.fastsweep).run()

        return data

    def _fast_sweep(
        self,
        start: float,
        stop: float,
        steps: int,
        mode: Literal["IV", "VI", "VIfourprobe"] = "IV",
    ) -> npt.NDArray:
        """
        Perform a fast sweep using a deployed Lua script.
        This is the engine that forms the script, uploads it,
        runs it, collects the data, and casts the data correctly.

        Args:
            start: starting voltage
            stop: end voltage
            steps: number of steps
            mode: Type of sweep, either 'IV' (voltage sweep),
                'VI' (current sweep two probe setup) or
                'VIfourprobe' (current sweep four probe setup)

        """

        channel = self.channel

        # an extra visa query, a necessary precaution
        # to avoid timing out when waiting for long
        # measurements
        nplc = self.nplc()

        dV = (stop - start) / (steps - 1)

        if mode == "IV":
            meas = "i"
            sour = "v"
            func = "1"
            sense_mode = "0"
        elif mode == "VI":
            meas = "v"
            sour = "i"
            func = "0"
            sense_mode = "0"
        elif mode == "VIfourprobe":
            meas = "v"
            sour = "i"
            func = "0"
            sense_mode = "1"
        else:
            raise ValueError(f"Invalid mode {mode}")

        script = [
            f"{channel}.measure.nplc = {nplc:.12f}",
            f"{channel}.source.output = 1",
            f"startX = {start:.12f}",
            f"dX = {dV:.12f}",
            f"{channel}.sense = {sense_mode}",
            f"{channel}.source.output = 1",
            f"{channel}.source.func = {func}",
            f"{channel}.measure.count = 1",
            f"{channel}.nvbuffer1.clear()",
            f"{channel}.nvbuffer1.appendmode = 1",
            f"for index = 1, {steps} do",
            "  target = startX + (index-1)*dX",
            f"  {channel}.source.level{sour} = target",
            f"  {channel}.measure.{meas}({channel}.nvbuffer1)",
            "end",
            "format.data = format.REAL32",
            "format.byteorder = format.LITTLEENDIAN",
            f"printbuffer(1, {steps}, {channel}.nvbuffer1.readings)",
        ]

        return self._execute_lua(script, steps)

    def _execute_lua(self, _script: list[str], steps: int) -> npt.NDArray:
        """
        This is the function that sends the Lua script to be executed and
        returns the corresponding data from the buffer.

        Args:
            _script: The Lua script to be executed.
            steps: Number of points.

        """
        nplc = self.nplc()
        linefreq = self.linefreq()
        _time_trace_extra_visa_timeout = self._extra_visa_timeout
        _factor = self._measurement_duration_factor
        estimated_measurement_duration = _factor * 1000 * steps * nplc / linefreq
        new_visa_timeout = (
            estimated_measurement_duration + _time_trace_extra_visa_timeout
        )

        self.write(self.root_instrument._scriptwrapper(program=_script, debug=True))

        # now poll all the data
        # The problem is that a '\n' character might by chance be present in
        # the data
        fullsize = 4 * steps + 3
        received = 0
        data = b""
        # we must wait for the script to execute
        with self.root_instrument.timeout.set_to(new_visa_timeout):
            while received < fullsize:
                data_temp = self.root_instrument.visa_handle.read_raw()
                received += len(data_temp)
                data += data_temp

        # From the manual p. 7-94, we know that a b'#0' is prepended
        # to the data and a b'\n' is appended
        data = data[2:-1]
        outdata = np.array(list(struct.iter_unpack("<f", data)))
        outdata = np.reshape(outdata, len(outdata))
        return outdata

    def _set_sourcerange_v(self, val: float) -> None:
        channel = self.channel
        self.source_autorange_v_enabled(False)
        self.write(f"{channel}.source.rangev={val}")

    def _set_measurerange_v(self, val: float) -> None:
        channel = self.channel
        self.measure_autorange_v_enabled(False)
        self.write(f"{channel}.measure.rangev={val}")

    def _set_sourcerange_i(self, val: float) -> None:
        channel = self.channel
        self.source_autorange_i_enabled(False)
        self.write(f"{channel}.source.rangei={val}")

    def _set_measurerange_i(self, val: float) -> None:
        channel = self.channel
        self.measure_autorange_i_enabled(False)
        self.write(f"{channel}.measure.rangei={val}")


class Keithley2600(VisaInstrument):
    """
    This is the base class for all  qcodes driver for the Keithley 2600 Source-Meter series.
    This class should not be instantiated directly. Rather one of the subclasses for a
    specific instrument should be used.
    """

    default_terminator = "\n"

    def __init__(
        self, name: str, address: str, **kwargs: Unpack[VisaInstrumentKWArgs]
    ) -> None:
        """
        Args:
            name: Name to use internally in QCoDeS
            address: VISA resource address
            **kwargs: kwargs are forwarded to base class.

        """
        super().__init__(name, address, **kwargs)

        model = self.ask("localnode.model")

        knownmodels = [
            "2601B",
            "2602A",
            "2602B",
            "2604B",
            "2611B",
            "2612B",
            "2614B",
            "2634B",
            "2635B",
            "2636B",
        ]
        if model not in knownmodels:
            kmstring = ("{}, " * (len(knownmodels) - 1)).format(*knownmodels[:-1])
            kmstring += f"and {knownmodels[-1]}."
            raise ValueError("Unknown model. Known model are: " + kmstring)

        self.model = model

        self._vranges = {
            "2601B": [0.1, 1, 6, 40],
            "2602A": [0.1, 1, 6, 40],
            "2602B": [0.1, 1, 6, 40],
            "2604B": [0.1, 1, 6, 40],
            "2611B": [0.2, 2, 20, 200],
            "2612B": [0.2, 2, 20, 200],
            "2614B": [0.2, 2, 20, 200],
            "2634B": [0.2, 2, 20, 200],
            "2635B": [0.2, 2, 20, 200],
            "2636B": [0.2, 2, 20, 200],
        }

        # TODO: In pulsed mode, models 2611B, 2612B, and 2614B
        # actually allow up to 10 A.
        self._iranges = {
            "2601B": [100e-9, 1e-6, 10e-6, 100e-6, 1e-3, 0.01, 0.1, 1, 3],
            "2602A": [100e-9, 1e-6, 10e-6, 100e-6, 1e-3, 0.01, 0.1, 1, 3],
            "2602B": [100e-9, 1e-6, 10e-6, 100e-6, 1e-3, 0.01, 0.1, 1, 3],
            "2604B": [100e-9, 1e-6, 10e-6, 100e-6, 1e-3, 0.01, 0.1, 1, 3],
            "2611B": [100e-9, 1e-6, 10e-6, 100e-6, 1e-3, 0.01, 0.1, 1, 1.5],
            "2612B": [100e-9, 1e-6, 10e-6, 100e-6, 1e-3, 0.01, 0.1, 1, 1.5],
            "2614B": [100e-9, 1e-6, 10e-6, 100e-6, 1e-3, 0.01, 0.1, 1, 1.5],
            "2634B": [
                1e-9,
                10e-9,
                100e-9,
                1e-6,
                10e-6,
                100e-6,
                1e-3,
                10e-6,
                100e-3,
                1,
                1.5,
            ],
            "2635B": [
                1e-9,
                10e-9,
                100e-9,
                1e-6,
                10e-6,
                100e-6,
                1e-3,
                10e-6,
                100e-3,
                1,
                1.5,
            ],
            "2636B": [
                1e-9,
                10e-9,
                100e-9,
                1e-6,
                10e-6,
                100e-6,
                1e-3,
                10e-6,
                100e-3,
                1,
                1.5,
            ],
        }

        self._vlimit_minmax = {
            "2601B": [10e-3, 40],
            "2602A": [10e-3, 40],
            "2602B": [10e-3, 40],
            "2604B": [10e-3, 40],
            "2611B": [20e-3, 200],
            "2612B": [20e-3, 200],
            "2614B": [20e-3, 200],
            "2634B": [20e-3, 200],
            "2635B": [20e-3, 200],
            "2636B": [20e-3, 200],
        }

        self._ilimit_minmax = {
            "2601B": [10e-9, 3],
            "2602A": [10e-9, 3],
            "2602B": [10e-9, 3],
            "2604B": [10e-9, 3],
            "2611B": [10e-9, 3],
            "2612B": [10e-9, 3],
            "2614B": [10e-9, 3],
            "2634B": [100e-12, 1.5],
            "2635B": [100e-12, 1.5],
            "2636B": [100e-12, 1.5],
        }
        # Add the channel to the instrument
        self.channels: list[Keithley2600Channel] = []
        for ch in ["a", "b"]:
            ch_name = f"smu{ch}"
            channel = Keithley2600Channel(self, ch_name, ch_name)
            self.add_submodule(ch_name, channel)
            self.channels.append(channel)

        # display
        self.display_settext: Parameter = self.add_parameter(
            "display_settext", set_cmd=self._display_settext, vals=vals.Strings()
        )
        """Parameter display_settext"""

        self.connect_message()

    def _display_settext(self, text: str) -> None:
        self.visa_handle.write(f'display.settext("{text}")')

    def get_idn(self) -> dict[str, str | None]:
        IDNstr = self.ask_raw("*IDN?")
        vendor, model, serial, firmware = map(str.strip, IDNstr.split(","))
        model = model[6:]

        IDN: dict[str, str | None] = {
            "vendor": vendor,
            "model": model,
            "serial": serial,
            "firmware": firmware,
        }
        return IDN

    def display_clear(self) -> None:
        """
        This function clears the display, but also leaves it in user mode
        """
        self.visa_handle.write("display.clear()")

    def display_normal(self) -> None:
        """
        Set the display to the default mode
        """
        self.visa_handle.write("display.screen = display.SMUA_SMUB")

    def exit_key(self) -> None:
        """
        Get back the normal screen after an error:
        send an EXIT key press event
        """
        self.visa_handle.write("display.sendkey(75)")

    def reset(self) -> None:
        """
        Reset instrument to factory defaults.
        This resets both channels.
        """
        self.write("reset()")
        # remember to update all the metadata
        log.debug("Reset instrument. Re-querying settings...")
        self.snapshot(update=True)

    def ask(self, cmd: str) -> str:
        """
        Override of normal ask. This is important, since queries to the
        instrument must be wrapped in 'print()'
        """
        return super().ask(f"print({cmd:s})")

    @staticmethod
    def _scriptwrapper(program: list[str], debug: bool = False) -> str:
        """
        Wraps a program so that the output can be put into
        visa_handle.write and run.
        The script will run immediately as an anonymous script.

        Args:
            program: A list of program instructions. One line per
                list item, e.g. ['for ii = 1, 10 do', 'print(ii)', 'end' ]
            debug: log additional debug output

        """
        mainprog = "\r\n".join(program) + "\r\n"
        wrapped = f"loadandrunscript\r\n{mainprog}endscript"
        if debug:
            log.debug("Wrapped the following script:")
            log.debug(wrapped)
        return wrapped
