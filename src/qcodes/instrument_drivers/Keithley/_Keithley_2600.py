from __future__ import annotations

import logging
import struct
import warnings
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

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
    Parameter,
    ParameterBase,
    ParameterWithSetpoints,
    ParamRawDataType,
    create_on_off_val_mapping,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Unpack


log = logging.getLogger(__name__)


class _LinSweepLike(Protocol):
    """
    Protocol for linear sweep objects that can be used with ``setup_fastsweep``.

    Any object implementing this protocol can be used to configure fast sweeps.
    The canonical example is :class:`qcodes.dataset.LinSweep`.

    Required attributes:
        param: The parameter being swept (e.g., ``keith.smua.volt``).
        delay: Time in seconds to wait after setting each point before measuring.
        num_points: Number of sweep points.

    Required methods:
        get_setpoints: Returns the array of setpoint values for the sweep.
    """

    @property
    def param(self) -> ParameterBase: ...

    @property
    def delay(self) -> float: ...

    @property
    def num_points(self) -> int: ...

    def get_setpoints(self) -> npt.NDArray: ...


@dataclass
class _FastSweepConfig:
    """Internal configuration for fastsweep."""

    inner_start: float
    inner_stop: float
    inner_npts: int
    inner_delay: float = 0.0
    inner_param_name: str = "Voltage"
    inner_param_unit: str = "V"
    inner_param_full_name: str | None = None  # Original parameter's full_name
    inner_channel: str = "smua"  # Lua channel name for inner sweep
    outer_start: float | None = None
    outer_stop: float | None = None
    outer_npts: int | None = None
    outer_delay: float = 0.0
    outer_param_name: str = "Voltage"
    outer_param_unit: str = "V"
    outer_param_full_name: str | None = None  # Original parameter's full_name
    outer_channel: str = "smub"  # Lua channel name for outer sweep
    mode: Literal["IV", "VI", "VIfourprobe"] = "IV"

    @property
    def is_2d(self) -> bool:
        return self.outer_npts is not None

    @property
    def total_points(self) -> int:
        if self.is_2d:
            assert self.outer_npts is not None
            return self.inner_npts * self.outer_npts
        return self.inner_npts

    def get_inner_setpoints(self) -> npt.NDArray:
        return np.linspace(self.inner_start, self.inner_stop, self.inner_npts)

    def get_outer_setpoints(self) -> npt.NDArray:
        if not self.is_2d:
            raise RuntimeError("No outer setpoints for 1D sweep")
        assert self.outer_start is not None
        assert self.outer_stop is not None
        assert self.outer_npts is not None
        return np.linspace(self.outer_start, self.outer_stop, self.outer_npts)


class _FastSweepInnerSetpoints(Parameter[npt.NDArray, "Keithley2600Channel"]):
    """Parameter that returns the inner axis setpoints for a fastsweep."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._source_full_name: str | None = None

    @property
    def register_name(self) -> str:
        """Return source parameter's full_name for dataset registration."""
        if self._source_full_name is not None:
            return self._source_full_name
        return self.full_name

    def get_raw(self) -> npt.NDArray:
        if self.instrument is None:
            raise RuntimeError("No instrument attached to Parameter.")
        config = self.instrument._fastsweep_config
        if config is None:
            raise RuntimeError("Fastsweep not configured. Call setup_fastsweep first.")
        return config.get_inner_setpoints()


class _FastSweepOuterSetpoints(Parameter[npt.NDArray, "Keithley2600Channel"]):
    """Parameter that returns the outer axis setpoints for a 2D fastsweep."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._source_full_name: str | None = None

    @property
    def register_name(self) -> str:
        """Return source parameter's full_name for dataset registration."""
        if self._source_full_name is not None:
            return self._source_full_name
        return self.full_name

    def get_raw(self) -> npt.NDArray:
        if self.instrument is None:
            raise RuntimeError("No instrument attached to Parameter.")
        config = self.instrument._fastsweep_config
        if config is None:
            raise RuntimeError("Fastsweep not configured. Call setup_fastsweep first.")
        return config.get_outer_setpoints()


class LuaSweepParameter(ParameterWithSetpoints[npt.NDArray, "Keithley2600Channel"]):
    """
    Parameter class to perform fast sweeps using Lua scripts on the Keithley 2600.

    Supports both 1D and 2D sweeps. Configure using the channel's
    ``setup_fastsweep`` method with sweep objects that implement the
    ``_LinSweepLike`` protocol (e.g., :class:`qcodes.dataset.LinSweep`).

    For 1D sweeps, returns a 1D array. For 2D sweeps, returns a 2D array
    with shape (outer_npts, inner_npts).

    For more information on writing Lua scripts for the Keithley2600, please see
    https://www.tek.com/en/documents/application-note/how-to-write-scripts-for-test-script-processing-(tsp)
    """

    def _update_metadata(self, config: _FastSweepConfig) -> None:
        """Update parameter metadata based on sweep configuration."""
        mode = config.mode

        match mode:
            case "IV":
                self.unit = "A"
                self.label = "Current"
            case "VI" | "VIfourprobe":
                self.unit = "V"
                self.label = "Voltage"

        # Build labels that include original parameter info for traceability
        inner_full_label = config.inner_param_name
        if config.inner_param_full_name:
            inner_full_label = (
                f"{config.inner_param_name} ({config.inner_param_full_name})"
            )

        if self.instrument is not None:
            # Set source name so dataset uses original parameter's name
            self.instrument.fastsweep_inner_setpoints._source_full_name = (
                config.inner_param_full_name
            )
            self.instrument.fastsweep_inner_setpoints.unit = config.inner_param_unit
            self.instrument.fastsweep_inner_setpoints.label = inner_full_label
            self.instrument.fastsweep_inner_setpoints.vals = vals.Arrays(
                shape=(config.inner_npts,)
            )

            if config.is_2d:
                assert config.outer_npts is not None
                outer_full_label = config.outer_param_name
                if config.outer_param_full_name:
                    outer_full_label = (
                        f"{config.outer_param_name} ({config.outer_param_full_name})"
                    )

                # Set source name so dataset uses original parameter's name
                self.instrument.fastsweep_outer_setpoints._source_full_name = (
                    config.outer_param_full_name
                )
                self.instrument.fastsweep_outer_setpoints.unit = config.outer_param_unit
                self.instrument.fastsweep_outer_setpoints.label = outer_full_label
                self.instrument.fastsweep_outer_setpoints.vals = vals.Arrays(
                    shape=(config.outer_npts,)
                )
                self.setpoints = (
                    self.instrument.fastsweep_outer_setpoints,
                    self.instrument.fastsweep_inner_setpoints,
                )
                self.setpoint_names = (outer_full_label, inner_full_label)
                self.setpoint_units = (config.outer_param_unit, config.inner_param_unit)
                self.vals = vals.Arrays(shape=(config.outer_npts, config.inner_npts))
            else:
                self.setpoints = (self.instrument.fastsweep_inner_setpoints,)
                self.setpoint_names = (inner_full_label,)  # type: ignore[assignment]
                self.setpoint_units = (config.inner_param_unit,)  # type: ignore[assignment]
                self.vals = vals.Arrays(shape=(config.inner_npts,))

    def _build_1d_script(self, config: _FastSweepConfig) -> list[str]:
        """Build Lua script for 1D sweep."""
        if self.instrument is None:
            raise RuntimeError("No instrument attached to Parameter.")

        channel = config.inner_channel
        nplc = self.instrument.nplc()

        dX = (config.inner_stop - config.inner_start) / (config.inner_npts - 1)

        match config.mode:
            case "IV":
                meas, source, func, sense_mode = "i", "v", "1", "0"
            case "VI":
                meas, source, func, sense_mode = "v", "i", "0", "0"
            case "VIfourprobe":
                meas, source, func, sense_mode = "v", "i", "0", "1"

        script = [
            f"{channel}.measure.nplc = {nplc:.12f}",
            f"{channel}.sense = {sense_mode}",
            f"{channel}.source.func = {func}",
            f"{channel}.source.output = 1",
            f"{channel}.measure.count = 1",
            f"startX = {config.inner_start:.12f}",
            f"dX = {dX:.12f}",
            f"{channel}.nvbuffer1.clear()",
            f"{channel}.nvbuffer1.appendmode = 1",
            f"for index = 1, {config.inner_npts} do",
            "  target = startX + (index-1)*dX",
            f"  {channel}.source.level{source} = target",
        ]

        if config.inner_delay > 0:
            script.append(f"  delay({config.inner_delay})")

        script.extend(
            [
                f"  {channel}.measure.{meas}({channel}.nvbuffer1)",
                "end",
                "format.data = format.REAL32",
                "format.byteorder = format.LITTLEENDIAN",
                f"printbuffer(1, {config.inner_npts}, {channel}.nvbuffer1.readings)",
            ]
        )
        return script

    def _build_2d_script(self, config: _FastSweepConfig) -> list[str]:
        """Build Lua script for 2D sweep."""
        if self.instrument is None:
            raise RuntimeError("No instrument attached to Parameter.")

        inner_channel = config.inner_channel
        outer_channel = config.outer_channel
        nplc = self.instrument.nplc()

        assert config.outer_start is not None
        assert config.outer_stop is not None
        assert config.outer_npts is not None

        dX_inner = (config.inner_stop - config.inner_start) / (config.inner_npts - 1)
        dX_outer = (config.outer_stop - config.outer_start) / (config.outer_npts - 1)

        match config.mode:
            case "IV":
                meas, source, func, sense_mode = "i", "v", "1", "0"
                outer_source, outer_func = "v", "1"
            case "VI":
                meas, source, func, sense_mode = "v", "i", "0", "0"
                outer_source, outer_func = "i", "0"
            case "VIfourprobe":
                meas, source, func, sense_mode = "v", "i", "0", "1"
                outer_source, outer_func = "i", "0"

        script = [
            # Set up inner channel (fast sweep)
            f"{inner_channel}.measure.nplc = {nplc:.12f}",
            f"{inner_channel}.sense = {sense_mode}",
            f"{inner_channel}.source.func = {func}",
            f"{inner_channel}.source.output = 1",
            f"{inner_channel}.measure.count = 1",
            # Set up outer channel
            f"{outer_channel}.source.func = {outer_func}",
            f"{outer_channel}.source.output = 1",
            # Initialize variables
            f"startX_inner = {config.inner_start:.12f}",
            f"dX_inner = {dX_inner:.12f}",
            f"startX_outer = {config.outer_start:.12f}",
            f"dX_outer = {dX_outer:.12f}",
            # Clear buffer
            f"{inner_channel}.nvbuffer1.clear()",
            f"{inner_channel}.nvbuffer1.appendmode = 1",
            # Outer loop (slow axis on other channel)
            f"for outer_idx = 1, {config.outer_npts} do",
            "  outer_target = startX_outer + (outer_idx-1)*dX_outer",
            f"  {outer_channel}.source.level{outer_source} = outer_target",
        ]

        if config.outer_delay > 0:
            script.append(f"  delay({config.outer_delay})")

        script.extend(
            [
                f"  for inner_idx = 1, {config.inner_npts} do",
                "    inner_target = startX_inner + (inner_idx-1)*dX_inner",
                f"    {inner_channel}.source.level{source} = inner_target",
            ]
        )

        if config.inner_delay > 0:
            script.append(f"    delay({config.inner_delay})")

        script.extend(
            [
                f"    {inner_channel}.measure.{meas}({inner_channel}.nvbuffer1)",
                "  end",
                "end",
                "format.data = format.REAL32",
                "format.byteorder = format.LITTLEENDIAN",
                f"printbuffer(1, {config.total_points}, {inner_channel}.nvbuffer1.readings)",
            ]
        )
        return script

    def get_raw(self) -> npt.NDArray:
        if self.instrument is None:
            raise RuntimeError("No instrument attached to Parameter.")

        config = self.instrument._fastsweep_config
        if config is None:
            raise RuntimeError("Fastsweep not configured. Call setup_fastsweep first.")

        if config.is_2d:
            script = self._build_2d_script(config)
            data = self.instrument._execute_lua(script, config.total_points)
            assert config.outer_npts is not None
            return data.reshape(config.outer_npts, config.inner_npts)
        else:
            script = self._build_1d_script(config)
            return self.instrument._execute_lua(script, config.total_points)


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

        status = _from_bits_tuple_to_status[(status_bits[0], status_bits[1])]

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

        # Internal fastsweep configuration - set via setup_fastsweep()
        self._fastsweep_config: _FastSweepConfig | None = None

        # Setpoint parameters for fastsweep
        self.fastsweep_inner_setpoints: _FastSweepInnerSetpoints = self.add_parameter(
            name="fastsweep_inner_setpoints",
            label="Sweep setpoints",
            snapshot_value=False,
            vals=vals.Arrays(shape=(1,)),  # Placeholder, updated by setup_fastsweep
            parameter_class=_FastSweepInnerSetpoints,
        )
        """Holds inner axis setpoints for fastsweep."""

        self.fastsweep_outer_setpoints: _FastSweepOuterSetpoints = self.add_parameter(
            name="fastsweep_outer_setpoints",
            label="Outer sweep setpoints",
            snapshot_value=False,
            vals=vals.Arrays(shape=(1,)),  # Placeholder, updated by setup_fastsweep
            parameter_class=_FastSweepOuterSetpoints,
        )
        """Holds outer axis setpoints for 2D fastsweep."""

        self.fastsweep: LuaSweepParameter = self.add_parameter(
            "fastsweep",
            vals=vals.Arrays(shape=(1,)),  # Placeholder, updated by setup_fastsweep
            setpoints=(self.fastsweep_inner_setpoints,),  # Updated by setup_fastsweep
            parameter_class=LuaSweepParameter,
            docstring="Performs a fast sweep using on-instrument Lua scripts. "
            "Configure using setup_fastsweep() with LinSweep-like object(s). "
            "For 1D sweeps, returns a 1D array. "
            "For 2D sweeps, returns a 2D array with shape (outer_npts, inner_npts).",
        )
        """
        Performs a fast sweep. Configure with setup_fastsweep() before use.
        Call fastsweep on the **inner** channel (the one from the first LinSweep).

        Example 1D:
            >>> from qcodes.dataset import LinSweep
            >>> keith.smua.setup_fastsweep(LinSweep(keith.smua.volt, 0, 1, 100))
            >>> ds, _, _ = do0d(keith.smua.fastsweep)

        Example 2D (inner=smub, outer=smua):
            >>> keith.smua.setup_fastsweep(
            ...     LinSweep(keith.smub.volt, 0, 1, 100),  # inner
            ...     LinSweep(keith.smua.volt, 0, 0.5, 20),  # outer
            ... )
            >>> ds, _, _ = do0d(keith.smub.fastsweep)  # call on inner channel
        """

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

    def setup_fastsweep(
        self,
        inner: _LinSweepLike,
        outer: _LinSweepLike | None = None,
        mode: Literal["IV", "VI", "VIfourprobe"] = "IV",
    ) -> None:
        """
        Configure a 1D or 2D fastsweep using sweep objects.

        Accepts any object implementing the ``_LinSweepLike`` protocol.
        The canonical example is :class:`qcodes.dataset.LinSweep`.

        Both 1D and 2D sweeps execute entirely on the instrument via Lua scripts,
        minimizing communication overhead.

        For 1D sweeps, provide only the inner sweep.
        For 2D sweeps, provide both inner and outer sweeps. The inner sweep
        runs to completion for each step of the outer sweep.

        The channels are determined by the sweep parameters you provide.
        After calling setup_fastsweep, call ``fastsweep`` on the **inner** channel
        (the channel from the first sweep object) to execute the measurement.

        Args:
            inner: Sweep object for the inner (fast) axis. Must have ``param``,
                   ``delay``, ``num_points`` attributes and a ``get_setpoints()``
                   method. See :class:`qcodes.dataset.LinSweep` for an example.
                   The channel is determined from ``param.instrument.channel``.
                   Measurement is performed on this channel.
            outer: Optional sweep object for the outer (slow) axis.
                   If provided, performs a 2D sweep.
            mode: Sweep mode - 'IV' (sweep voltage, measure current),
                  'VI' (sweep current, measure voltage), or
                  'VIfourprobe' (four-probe VI measurement).

        Example 1D:

            >>> from qcodes.dataset import LinSweep
            >>> keith.smua.setup_fastsweep(LinSweep(keith.smua.volt, 0, 1, 100))
            >>> ds, _, _ = do0d(keith.smua.fastsweep)

        Example 2D (inner=smua, outer=smub):

            >>> keith.smua.setup_fastsweep(
            ...     LinSweep(keith.smua.volt, 0, 1, 100),  # inner
            ...     LinSweep(keith.smub.volt, 0, 0.5, 20),  # outer
            ... )
            >>> ds, _, _ = do0d(keith.smua.fastsweep)  # call on inner channel

        Example 2D (inner=smub, outer=smua):

            >>> keith.smua.setup_fastsweep(
            ...     LinSweep(keith.smub.volt, 0, 1, 100),  # inner
            ...     LinSweep(keith.smua.volt, 0, 0.5, 20),  # outer
            ... )
            >>> ds, _, _ = do0d(keith.smub.fastsweep)  # call on inner channel

        """

        # Helper to extract channel name from parameter
        def get_channel(param: ParameterBase) -> str:
            """Extract Lua channel name (smua/smub) from parameter."""
            inst = param.instrument

            if not isinstance(inst, Keithley2600Channel):
                raise ValueError(
                    f"Parameter '{param.name}' must belong to a Keithley2600Channel. "
                    f"Got instrument of type {type(inst).__name__}."
                )

            return inst.channel

        # Get setpoints from inner sweep to derive start/stop
        inner_setpoints = inner.get_setpoints()
        inner_start = float(inner_setpoints[0])
        inner_stop = float(inner_setpoints[-1])
        inner_param = cast("Parameter", inner.param)
        inner_channel = get_channel(inner_param)

        # Build the configuration
        config = _FastSweepConfig(
            inner_start=inner_start,
            inner_stop=inner_stop,
            inner_npts=inner.num_points,
            inner_delay=inner.delay,
            inner_param_name=inner_param.label,
            inner_param_unit=inner_param.unit,
            inner_param_full_name=inner_param.full_name,
            inner_channel=inner_channel,
            mode=mode,
        )

        # Add outer sweep configuration if provided
        if outer is not None:
            outer_setpoints = outer.get_setpoints()
            outer_start = float(outer_setpoints[0])
            outer_stop = float(outer_setpoints[-1])
            outer_param = cast("Parameter", outer.param)
            outer_channel = get_channel(outer_param)

            config.outer_start = outer_start
            config.outer_stop = outer_stop
            config.outer_npts = outer.num_points
            config.outer_delay = outer.delay
            config.outer_param_name = outer_param.label
            config.outer_param_unit = outer_param.unit
            config.outer_param_full_name = outer_param.full_name
            config.outer_channel = outer_channel

        # Get the inner channel object where fastsweep should be called from
        # (measurement happens on the inner channel)
        inner_channel_obj: Keithley2600Channel = getattr(
            self.root_instrument, inner_channel
        )

        # Store configuration on the inner channel - users call fastsweep there
        inner_channel_obj._fastsweep_config = config

        # Update fastsweep parameter metadata on the inner channel
        inner_channel_obj.fastsweep._update_metadata(config)

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
