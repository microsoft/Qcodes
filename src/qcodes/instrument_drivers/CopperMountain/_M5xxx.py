from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from qcodes.instrument import VisaInstrument, VisaInstrumentKWArgs
from qcodes.parameters import (
    ManualParameter,
    MultiParameter,
    Parameter,
    ParamRawDataType,
    create_on_off_val_mapping,
)
from qcodes.validators import Bool, Enum, Ints, Numbers

if TYPE_CHECKING:
    from typing_extensions import Unpack


class CopperMountainM5xxx(VisaInstrument):
    """
    Base class for QCoDeS drivers for Copper Mountain M-series VNAs.

    Not to be instantiated directly. Use model specific subclass.
    https://coppermountaintech.com/help-s2/index.html

    Note: Currently this driver only expects a single channel on the PNA. We
          can handle multiple traces, but using traces across multiple channels
          may have unexpected results.
    """

    default_terminator = "\n"

    def __init__(
        self,
        name: str,
        address: str,
        min_freq: float,
        max_freq: float,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ):
        """
        QCoDeS driver for Copper Mountain M-series VNA (M5xxx).
        This driver supports only a single channel.

        Args:
            name: Identifier for the instrument instance.
            address: VISA address of the instrument.
            min_freq: Minimum frequency supported by the instrument (in Hz).
            max_freq: Maximum frequency supported by the instrument (in Hz).
            **kwargs: Additional keyword arguments for VisaInstrument.

        """

        super().__init__(name=name, address=address, **kwargs)
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_points = 2
        self.max_points = 200001

        self.output: Parameter = self.add_parameter(
            name="output",
            label="Output",
            get_parser=int,
            get_cmd="OUTP:STAT?",
            set_cmd="OUTP:STAT {}",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
        )
        """Use to check state of RF signal output (ON/OFF) and turns the RF signal output ON/OFF"""

        self.power: Parameter = self.add_parameter(
            name="power",
            label="Power",
            get_parser=float,
            get_cmd="SOUR:POW?",
            set_cmd="SOUR:POW {}",
            unit="dBm",
            docstring="Sets or reads out the power level for the frequency sweep type in dBm.",
            vals=Numbers(min_value=-55, max_value=5),
        )
        """Sets or reads out the power level for the frequency sweep type in dBm."""

        self.if_bandwidth: Parameter = self.add_parameter(
            name="if_bandwidth",
            label="IF Bandwidth",
            get_parser=float,
            get_cmd="SENS1:BWID?",
            set_cmd="SENS1:BWID {}",
            unit="Hz",
            vals=Enum(
                *np.append(
                    np.kron([1, 1.5, 2, 3, 5, 7], 10 ** np.arange(5)),
                    np.kron([1, 1.5, 2, 3], 10**5),
                )
            ),
        )
        """Sets or reads out the IF bandwidth in Hz."""

        self.averages_enabled: Parameter = self.add_parameter(
            name="averages_enabled",
            label="Averages Status",
            get_cmd="SENS1:AVER:STAT?",
            set_cmd=self._set_averages_enabled,
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0"),
        )
        """Turns the measurement averaging function ON/OFF on channel 1."""

        self.averages_trigger_enabled: Parameter = self.add_parameter(
            "averages_trigger_enabled",
            label="Trigger average status",
            get_cmd="TRIG:SEQ:AVER?",
            set_cmd="TRIG:SEQ:AVER {}",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0"),
        )
        """Turns the averaging trigger function ON/OFF."""

        self.averages: Parameter = self.add_parameter(
            "averages",
            label="Averages",
            get_cmd="SENS1:AVER:COUN?",
            set_cmd="SENS1:AVER:COUN {}",
            get_parser=int,
            set_parser=int,
            unit="",
            docstring="Sets or reads out the averaging factor "
            "when the averaging function is turned on.",
            vals=Numbers(min_value=1, max_value=999),
        )
        """Sets or reads out the averaging factor
        when the averaging function is turned on."""

        self.electrical_delay: Parameter = self.add_parameter(
            "electrical_delay",
            label="Electrical delay",
            get_cmd="CALC1:CORR:EDEL:TIME?",
            set_cmd="CALC1:CORR:EDEL:TIME {}",
            get_parser=float,
            set_parser=float,
            unit="s",
            docstring="Sets or reads out the value of the electrical delay in seconds.",
            vals=Numbers(-10, 10),
        )
        """Sets or reads out the value of the electrical delay in seconds."""

        self.electrical_distance: Parameter = self.add_parameter(
            "electrical_distance",
            label="Electrical distance",
            get_cmd="CALC1:CORR:EDEL:DIST?",
            set_cmd="CALC1:CORR:EDEL:DIST {}",
            get_parser=float,
            set_parser=float,
            docstring="Sets or reads out the value of the equivalent "
            "distance in the electrical delay function.",
            vals=Numbers(),
        )
        """Sets or reads out the value of the equivalent
        distance in the electrical delay function."""

        self.electrical_distance_units: Parameter = self.add_parameter(
            "electrical_distance_units",
            label="Electrical distance units",
            get_cmd="CALC1:CORR:EDEL:DIST:UNIT?",
            set_cmd="CALC1:CORR:EDEL:DIST:UNIT {}",
            get_parser=str,
            docstring="Sets or reads out the distance units in the electrical delay function.",
            vals=Enum("MET", "FEET", "INCH"),
        )
        """Sets or reads out the distance units in the electrical delay function."""

        self.clock_source: Parameter = self.add_parameter(
            name="clock_source",
            label="Clock source",
            get_cmd="SENSe1:ROSCillator:SOURce?",
            set_cmd="SENSe1:ROSCillator:SOURce {}",
            get_parser=str,
            set_parser=str,
            docstring="Sets or reads out an internal or external "
            "source of the 10 MHz reference frequency.",
            vals=Enum(
                "int",
                "Int",
                "INT",
                "internal",
                "Internal",
                "INTERNAL",
                "ext",
                "Ext",
                "EXT",
                "external",
                "External",
                "EXTERNAL",
            ),
        )
        """Sets or reads out an internal or external
        source of the 10 MHz reference frequency."""

        self.start: Parameter = self.add_parameter(
            name="start",
            label="Start Frequency",
            get_parser=float,
            get_cmd="SENS1:FREQ:STAR?",
            set_cmd=self._set_start,
            unit="Hz",
            docstring="Sets or reads out the stimulus start value "
            "of the sweep range for linear or logarithmic sweep type.",
            vals=Numbers(min_value=self.min_freq, max_value=self.max_freq - 1),
        )
        """Sets or reads out the stimulus start value of the sweep
        range for linear or logarithmic sweep type."""

        self.stop: Parameter = self.add_parameter(
            name="stop",
            label="Stop Frequency",
            get_parser=float,
            get_cmd="SENS1:FREQ:STOP?",
            set_cmd=self._set_stop,
            unit="Hz",
            docstring="Sets or reads out the stimulus stop value of the "
            "sweep range for linear or logarithmic sweep type.",
            vals=Numbers(min_value=self.min_freq + 1, max_value=self.max_freq),
        )
        """Sets or reads out the stimulus stop value of the sweep
        range for linear or logarithmic sweep type."""

        self.center: Parameter = self.add_parameter(
            name="center",
            label="Center Frequency",
            get_parser=float,
            get_cmd="SENS1:FREQ:CENT?",
            set_cmd=self._set_center,
            unit="Hz",
            docstring="Sets or reads out the stimulus center value of "
            "the sweep range for linear or logarithmic sweep type.",
            vals=Numbers(min_value=self.min_freq + 1, max_value=self.max_freq - 1),
        )
        """Sets or reads out the stimulus center value of the sweep range for
        linear or logarithmic sweep type."""

        self.span: Parameter = self.add_parameter(
            name="span",
            label="Frequency Span",
            get_parser=float,
            get_cmd="SENS1:FREQ:SPAN?",
            set_cmd=self._set_span,
            unit="Hz",
            docstring="Sets or reads out the stimulus span value of the "
            "sweep range for linear or logarithmic sweep type.",
            vals=Numbers(min_value=1, max_value=self.max_freq - 1),
        )
        """Sets or reads out the stimulus span value of the sweep range
        for linear or logarithmic sweep type."""

        self.number_of_points: Parameter = self.add_parameter(
            "number_of_points",
            label="Number of points",
            get_parser=int,
            set_parser=int,
            get_cmd="SENS1:SWE:POIN?",
            set_cmd=self._set_number_of_points,
            docstring="Sets or reads out the number of measurement points.",
            vals=Ints(min_value=self.min_points, max_value=self.max_points),
        )
        """Sets or reads out the number of measurement points."""

        self.number_of_traces: Parameter = self.add_parameter(
            name="number_of_traces",
            label="Number of traces",
            get_parser=int,
            set_parser=int,
            get_cmd="CALC1:PAR:COUN?",
            set_cmd="CALC1:PAR:COUN {}",
            unit="",
            docstring="Sets or reads out the number of traces in the channel.",
            vals=Ints(min_value=1, max_value=16),
        )
        """Sets or reads out the number of traces in the channel."""

        self.trigger_source: Parameter = self.add_parameter(
            name="trigger_source",
            label="Trigger source",
            get_parser=str,
            get_cmd=self._get_trigger,
            set_cmd=self._set_trigger,
            docstring="Selects the trigger source.",
            vals=Enum("bus", "external", "internal", "manual"),
        )
        """Selects the trigger source"""

        self.data_transfer_format: Parameter = self.add_parameter(
            name="data_transfer_format",
            label="Data format during transfer",
            get_parser=str,
            get_cmd="FORM:DATA?",
            set_cmd="FORM:DATA {}",
            docstring="Sets or reads out the data transfer format "
            "when responding to certain queries.",
            vals=Enum("ascii"),
        )
        """Sets or reads out the data transfer format when
        responding to certian queries."""

        self.s11: FrequencySweepMagPhase = self.add_parameter(
            name="s11",
            start=self.start(),
            stop=self.stop(),
            number_of_points=self.number_of_points(),
            parameter_class=FrequencySweepMagPhase,
            docstring="Input reflection.",
        )
        """Input reflection."""

        self.s12: FrequencySweepMagPhase = self.add_parameter(
            name="s12",
            start=self.start(),
            stop=self.stop(),
            number_of_points=self.number_of_points(),
            parameter_class=FrequencySweepMagPhase,
            docstring="Reverse transmission.",
        )
        """Reverse transmission."""

        self.s21: FrequencySweepMagPhase = self.add_parameter(
            name="s21",
            start=self.start(),
            stop=self.stop(),
            number_of_points=self.number_of_points(),
            parameter_class=FrequencySweepMagPhase,
            docstring="Forward transmission",
        )
        """Forward transmission"""

        self.s22: FrequencySweepMagPhase = self.add_parameter(
            name="s22",
            start=self.start(),
            stop=self.stop(),
            number_of_points=self.number_of_points(),
            parameter_class=FrequencySweepMagPhase,
            docstring="Output reflection",
        )
        """Output reflection"""

        self.point_s11: PointMagPhase = self.add_parameter(
            name="point_s11", parameter_class=PointMagPhase
        )

        self.point_s12: PointMagPhase = self.add_parameter(
            name="point_s12", parameter_class=PointMagPhase
        )

        self.point_s21: PointMagPhase = self.add_parameter(
            name="point_s21", parameter_class=PointMagPhase
        )

        self.point_s22: PointMagPhase = self.add_parameter(
            name="point_s22", parameter_class=PointMagPhase
        )

        self.point_s11_iq: PointIQ = self.add_parameter(
            name="point_s11_iq", parameter_class=PointIQ
        )

        self.point_s12_iq: PointIQ = self.add_parameter(
            name="point_s12_iq", parameter_class=PointIQ
        )

        self.point_s21_iq: PointIQ = self.add_parameter(
            name="point_s21_iq", parameter_class=PointIQ
        )

        self.point_s22_iq: PointIQ = self.add_parameter(
            name="point_s22_iq", parameter_class=PointIQ
        )

        self.point_check_sweep_first: ManualParameter = self.add_parameter(
            name="point_check_sweep_first",
            parameter_class=ManualParameter,
            initial_value=True,
            vals=Bool(),
            docstring="Parameter that enables a few commands, which are called"
            "before each get of a point_sxx parameter checking whether the vna"
            "is setup correctly. Is recommended to be True, but can be turned"
            "off if one wants to minimize overhead.",
        )

        # Electrical distance default units.
        self.electrical_distance_units("MET")

        self.connect_message()

    def reset(self) -> None:
        self.write("*RST")

    def _set_start(self, val: float) -> None:
        """Sets the start frequency and updates linear trace parameters.

        Args:
            val: start frequency to be set

        Raises:
            ValueError: If start > stop

        """
        stop = self.stop()
        if val >= stop:
            raise ValueError("Stop frequency must be larger than start frequency.")
        self.write(f"SENS1:FREQ:STAR {val}")
        # we get start as the vna may not be able to set it to the
        # exact value provided.
        start = self.start()
        if abs(val - start) >= 1:
            self.log.info(f"Could not set start to {val} setting it to {start}")
        self.update_lin_traces()

    def _set_stop(self, val: float) -> None:
        """Sets the start frequency and updates linear trace parameters.

        Args:
            val: start frequency to be set

        Raises:
            ValueError: If stop < start

        """
        start = self.start()
        if val <= start:
            raise ValueError("Stop frequency must be larger than start frequency.")
        self.write(f"SENS1:FREQ:STOP {val}")
        # We get stop as the vna may not be able to set it to the
        # exact value provided.
        stop = self.stop()
        if abs(val - stop) >= 1:
            self.log.info(f"Could not set stop to {val} setting it to {stop}")
        self.update_lin_traces()

    def _set_averages_enabled(self, averages_enabled: str) -> None:
        """Set averages_trigger_enabled along with averages_enabled or
        else triggering won't work properly

        Args:
            averages_enabled: value mapping from parameter ("ON" or "OFF")

        """
        self.write(f"SENS1:AVER:STAT {averages_enabled}")
        self.averages_trigger_enabled("ON")

    def _set_span(self, val: float) -> None:
        """Sets frequency span and updates linear trace parameters.

        Args:
            val: frequency span to be set

        """
        self.write(f"SENS1:FREQ:SPAN {val}")
        self.update_lin_traces()

    def _set_center(self, val: float) -> None:
        """Sets center frequency and updates linear trace parameters.

        Args:
            val: center frequency to be set

        """
        self.write(f"SENS1:FREQ:CENT {val}")
        self.update_lin_traces()

    def _set_number_of_points(self, val: int) -> None:
        """Sets number of points and updates linear trace parameters.

        Args:
            val: number of points to be set.

        """
        self.write(f"SENS1:SWE:POIN {val}")
        self.update_lin_traces()

    def _get_trigger(self) -> str:
        """Gets trigger source.

        Returns:
            str: Trigger source.

        """
        r = self.ask("TRIG:SOUR?")

        if r.lower() == "int":
            return "internal"
        elif r.lower() == "ext":
            return "external"
        elif r.lower() == "man":
            return "manual"
        else:
            return "bus"

    def _set_trigger(
        self, trigger: Literal["external", "internal", "manual", "bus"]
    ) -> None:
        """Sets trigger source.

        Args:
            trigger: Trigger source

        """
        self.write("TRIG:SOUR " + trigger.upper())

    def _set_trace_formats_to_polar(self, traces: list[int]) -> None:
        """
        Sets the format of the specified traces to SMITH (real + imaginary).

        Args:
            traces: A list of trace indices to set the format for.

        Returns:
            None

        """

        for trace in traces:
            self.write(f"CALC1:TRAC{trace}:FORM POLar")

    def get_s_parameters(
        self, expected_measurement_duration: float = 600
    ) -> tuple[
        "NDArray",
        "NDArray",
        "NDArray",
        "NDArray",
        "NDArray",
        "NDArray",
        "NDArray",
        "NDArray",
        "NDArray",
    ]:
        """
        Return all S parameters as magnitude in dB and phase in rad.

        Args:
            expected_measurement_duration: Expected duration of the measurement in seconds.

        Returns:
            Tuple[NDArray]: frequency [GHz],
            s11 magnitude [dB], s11 phase [rad],
            s12 magnitude [dB], s12 phase [rad],
            s21 magnitude [dB], s21 phase [rad],
            s22 magnitude [dB], s22 phase [rad]

        """
        timeout = self.timeout()
        current_timeout = timeout if timeout is not None else float("inf")

        with self.timeout.set_to(max(current_timeout, expected_measurement_duration)):
            self.write("CALC1:PAR:COUN 4")  # 4 trace
            self.write("CALC1:PAR1:DEF S11")  # Choose S11 for trace 1
            self.write("CALC1:PAR2:DEF S12")  # Choose S12 for trace 2
            self.write("CALC1:PAR3:DEF S21")  # Choose S21 for trace 3
            self.write("CALC1:PAR4:DEF S22")  # Choose S22 for trace 4
            self._set_trace_formats_to_polar(traces=[1, 2, 3, 4])
            self.write("TRIG:SEQ:SING")  # Trigger a single sweep
            self.ask("*OPC?")  # Wait for measurement to complete

            # Get data as string
            freq_raw = self.ask("SENS1:FREQ:DATA?")
            s11_raw = self.ask("CALC1:TRAC1:DATA:FDAT?")
            s12_raw = self.ask("CALC1:TRAC2:DATA:FDAT?")
            s21_raw = self.ask("CALC1:TRAC3:DATA:FDAT?")
            s22_raw = self.ask("CALC1:TRAC4:DATA:FDAT?")

        # Get data as numpy array
        freq = np.fromstring(freq_raw, dtype=float, sep=",")
        s11 = np.fromstring(s11_raw, dtype=float, sep=",")
        s11 = s11[0::2] + 1j * s11[1::2]
        s12 = np.fromstring(s12_raw, dtype=float, sep=",")
        s12 = s12[0::2] + 1j * s12[1::2]
        s21 = np.fromstring(s21_raw, dtype=float, sep=",")
        s21 = s21[0::2] + 1j * s21[1::2]
        s22 = np.fromstring(s22_raw, dtype=float, sep=",")
        s22 = s22[0::2] + 1j * s22[1::2]

        return (
            np.array(freq),
            self._db(s11),
            np.array(np.angle(s11)),
            self._db(s12),
            np.array(np.angle(s12)),
            self._db(s21),
            np.array(np.angle(s21)),
            self._db(s22),
            np.array(np.angle(s22)),
        )

    def update_lin_traces(self) -> None:
        """
        Updates start, stop and number_of_points of all trace parameters so that the
        setpoints and shape are updated for the sweep.
        """
        start = self.start()
        stop = self.stop()
        number_of_points = self.number_of_points()
        for _, parameter in self.parameters.items():
            if isinstance(parameter, (FrequencySweepMagPhase)):
                try:
                    parameter.set_sweep(start, stop, number_of_points)
                except AttributeError:
                    pass

    def reset_averages(self) -> None:
        """
        Resets average count to 0
        """
        self.write("SENS1.AVER.CLE")

    @staticmethod
    def _db(data: "NDArray") -> "NDArray":
        """
        Return dB from magnitude

        Args:
            data: data to be transformed into dB.

        Returns:
            data: data transformed in dB.

        """

        return 20.0 * np.log10(np.abs(data))


class FrequencySweepMagPhase(
    MultiParameter[tuple[NDArray, NDArray], CopperMountainM5xxx]
):
    """
    Sweep that returns magnitude and phase.
    """

    def __init__(
        self,
        name: str,
        start: float,
        stop: float,
        number_of_points: int,
        instrument: CopperMountainM5xxx,
        expected_measurement_duration: float = 600,
        **kwargs: Any,
    ) -> None:
        """
        Linear frequency sweep that returns magnitude and phase for a single
        trace.

        Args:
            name: Name of the linear frequency sweep
            start: Start frequency of linear sweep
            stop: Stop frequency of linear sweep
            number_of_points: Number of points of linear sweep
            instrument: Instrument to which sweep is bound to.
            expected_measurement_duration: Adjusts instrument timeout (seconds). Defaults to 600 seconds.
            **kwargs: Any

        """

        self.expected_measurement_duration = expected_measurement_duration

        super().__init__(
            name,
            instrument=instrument,
            names=(
                f"{instrument.short_name}_{name}_magnitude",
                f"{instrument.short_name}_{name}_phase",
            ),
            labels=(
                f"{instrument.short_name} {name} magnitude",
                f"{instrument.short_name} {name} phase",
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
                (number_of_points,),
                (number_of_points,),
            ),
            **kwargs,
        )
        self.set_sweep(start, stop, number_of_points)

    def set_sweep(self, start: float, stop: float, number_of_points: int) -> None:
        """Updates the setpoints and shapes based on start, stop and number_of_points.

        Args:
            start: start frequency
            stop: stop frequency
            number_of_points: number of points

        """
        f = tuple(np.linspace(int(start), int(stop), num=number_of_points))
        self.setpoints = ((f,), (f,))
        self.shapes = ((number_of_points,), (number_of_points,))

    def get_raw(self) -> tuple[NDArray, NDArray]:
        """Gets data from instrument

        Returns:
            Tuple[ParamRawDataType, ...]: magnitude, phase

        """

        timeout = self.instrument.timeout()
        current_timeout = timeout if timeout is not None else float("inf")

        with self.instrument.timeout.set_to(
            max(current_timeout, self.expected_measurement_duration)
        ):
            self.instrument.write("CALC1:PAR:COUN 1")  # 1 trace
            self.instrument.write(f"CALC1:PAR1:DEF {self.name}")

            # ensure correct format
            self.instrument._set_trace_formats_to_polar(traces=[1])
            self.instrument.trigger_source("bus")  # set the trigger to bus
            self.instrument.write("INIT")  # put in wait for trigger mode
            self.instrument.write("TRIG:SEQ:SING")  # Trigger a single sweep
            self.instrument.ask("*OPC?")  # Wait for measurement to complete

            # get data from instrument
            sxx_raw = self.instrument.ask("CALC1:TRAC1:DATA:FDAT?")

        # Get data as numpy array
        sxx = np.fromstring(sxx_raw, dtype=float, sep=",")
        sxx = sxx[0::2] + 1j * sxx[1::2]

        return self.instrument._db(sxx), np.angle(sxx)


class PointMagPhase(
    MultiParameter[tuple[np.floating, np.floating], CopperMountainM5xxx]
):
    """
    Returns the average Sxx of a frequency sweep.
    Work around for a CW mode where only one point is read.
    number_of_points=2 and stop = start + 1 (in Hz) is required.
    """

    def __init__(
        self,
        name: str,
        instrument: CopperMountainM5xxx,
        expected_measurement_duration: float = 600,
        **kwargs: Any,
    ) -> None:
        """Magnitude and phase measurement of a single point at start
        frequency.

        Args:
            name: Name of point measurement
            instrument:  Instrument to which parameter is bound to.
            expected_measurement_duration: Adjusts instrument timeout (seconds). Defaults to 600 seconds.
            **kwargs: Any

        """

        self.expected_measurement_duration = expected_measurement_duration

        super().__init__(
            name,
            instrument=instrument,
            names=(
                f"{instrument.short_name}_{name}_magnitude",
                f"{instrument.short_name}_{name}_phase",
            ),
            labels=(
                f"{instrument.short_name} {name} magnitude",
                f"{instrument.short_name} {name} phase",
            ),
            units=("dB", "rad"),
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

    def get_raw(self) -> tuple[ParamRawDataType, ParamRawDataType]:
        """Gets data from instrument

        Returns:
            Tuple[ParamRawDataType, ...]: magnitude, phase

        """
        # check that number_of_points, start and stop fullfill requirements if point_check_sweep_first is True.
        if self.instrument.point_check_sweep_first():
            if self.instrument.number_of_points() != 2:
                raise ValueError(
                    f"number_of_points is not 2 but {self.instrument.number_of_points()}. Please set it to 2"
                )
            if self.instrument.stop() - self.instrument.start() != 1:
                raise ValueError(
                    f"Stop-start is not 1 Hz but {self.instrument.stop() - self.instrument.start()} Hz. "
                    "Please adjust start or stop."
                )

        timeout = self.instrument.timeout()
        current_timeout = timeout if timeout is not None else float("inf")

        with self.instrument.timeout.set_to(
            max(current_timeout, self.expected_measurement_duration)
        ):
            self.instrument.write("CALC1:PAR:COUN 1")  # 1 trace
            self.instrument.write(f"CALC1:PAR1:DEF {self.name[-3:]}")

            # ensure correct format
            self.instrument._set_trace_formats_to_polar(traces=[1])
            self.instrument.trigger_source("bus")  # set the trigger to bus
            self.instrument.write("INIT")  # put in wait for trigger mode
            self.instrument.write("TRIG:SEQ:SING")  # Trigger a single sweep
            self.instrument.ask("*OPC?")  # Wait for measurement to complete

            # get data from instrument
            sxx_raw = self.instrument.ask("CALC1:TRAC1:DATA:FDAT?")

        # Get data as numpy array
        sxx = np.fromstring(sxx_raw, dtype=float, sep=",")
        sxx = sxx[0::2] + 1j * sxx[1::2]

        # Return the average of the trace, which will have "start" as
        # its setpoint
        sxx_mean = np.mean(sxx)
        return 20 * np.log10(abs(sxx_mean)), (np.angle(sxx_mean))


class PointIQ(MultiParameter[tuple[np.floating, np.floating], CopperMountainM5xxx]):
    """
    Returns the average Sxx of a frequency sweep, in terms of I and Q.
    Work around for a CW mode where only one point is read.
    number_of_points=2 and stop = start + 1 (in Hz) is required.
    """

    def __init__(
        self,
        name: str,
        instrument: CopperMountainM5xxx,
        expected_measurement_duration: float = 600,
        **kwargs: Any,
    ) -> None:
        """I and Q measurement of a single point at start
        frequency.

        Args:
            name: Name of point measurement
            instrument:  Instrument to which parameter is bound to.
            expected_measurement_duration: Adjusts instrument timeout (seconds). Defaults to 600 seconds.
            **kwargs: Any

        """

        self.expected_measurement_duration = expected_measurement_duration

        super().__init__(
            name,
            instrument=instrument,
            names=(
                f"{instrument.short_name}_{name}_i",
                f"{instrument.short_name}_{name}_q",
            ),
            labels=(
                f"{instrument.short_name} {name} i",
                f"{instrument.short_name} {name} q",
            ),
            units=("V", "V"),
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

    def get_raw(self) -> tuple[ParamRawDataType, ParamRawDataType]:
        """Gets data from instrument

        Returns:
            Tuple[ParamRawDataType, ...]: I, Q

        """
        # check that number_of_points, start and stop fullfill requirements if point_check_sweep_first is True.
        if self.instrument.point_check_sweep_first():
            if self.instrument.number_of_points() != 2:
                raise ValueError(
                    f"number_of_points is not 2 but {self.instrument.number_of_points()}. Please set it to 2"
                )
            if self.instrument.stop() - self.instrument.start() != 1:
                raise ValueError(
                    f"Stop-start is not 1 Hz but {self.instrument.stop() - self.instrument.start()} Hz. Please adjust start or stop."
                )

        timeout = self.instrument.timeout()
        current_timeout = timeout if timeout is not None else float("inf")

        with self.instrument.timeout.set_to(
            max(current_timeout, self.expected_measurement_duration)
        ):
            self.instrument.write("CALC1:PAR:COUN 1")  # 1 trace
            self.instrument.write(f"CALC1:PAR1:DEF {self.name[-3:]}")

            # ensure correct format
            self.instrument._set_trace_formats_to_polar(traces=[1])
            self.instrument.trigger_source("bus")  # set the trigger to bus
            self.instrument.write("INIT")  # put in wait for trigger mode
            self.instrument.write("TRIG:SEQ:SING")  # Trigger a single sweep
            self.instrument.ask("*OPC?")  # Wait for measurement to complete

            # get data from instrument
            sxx_raw = self.instrument.ask("CALC1:TRAC1:DATA:FDAT?")

        # Get data as numpy array
        sxx = np.fromstring(sxx_raw, dtype=float, sep=",")

        # Return the average of the trace, which will have "start" as
        # its setpoint
        return np.mean(sxx[0::2]), np.mean(sxx[1::2])
