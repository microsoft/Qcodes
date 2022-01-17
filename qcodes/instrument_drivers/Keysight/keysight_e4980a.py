from typing import Any, Sequence, Tuple, Union, cast

from packaging import version
from pyvisa.errors import VisaIOError

from qcodes import InstrumentChannel, VisaInstrument
from qcodes.instrument.group_parameter import Group, GroupParameter
from qcodes.instrument.parameter import (
    ManualParameter,
    MultiParameter,
    ParamRawDataType,
)
from qcodes.utils.helpers import create_on_off_val_mapping
from qcodes.utils.installation_info import convert_legacy_version_to_supported_version
from qcodes.utils.validators import Bool, Enum, Ints, Numbers


class MeasurementPair(MultiParameter):
    """
    Data class for E4980A measurement, which will always return two items
    at once.

    The two items are for two different parameters, depending on the measurement
    function. Hence, the names of the two attributes are created from the
    "names" tuple of the measurement functions.

    Examples:
        To create a measurement data with capacitance=1.2, and
        dissipation_factor=3.4.

        >>> data = MeasurementPair(name="CPD",
                                    names=("capacitance", "dissipation_factor"),
                                    units=("F", ""))
        >>> data.set((1.2, 3.4))
        >>> data.get()
        (1.2, 3.4)
    """
    value: Tuple[float, float] = (0., 0.)

    def __init__(self,
                 name: str,
                 names: Sequence[str],
                 units: Sequence[str],
                 **kwargs: Any):
        super().__init__(name=name,
                         names=names,
                         shapes=((), ()),
                         units=units,
                         setpoints=((), ()),
                         **kwargs)
        self.__dict__.update(
            {names[0]: 0,
             names[1]: 0}
        )

    def set_raw(self, value: Tuple[float, float]) -> None:
        self.value = value
        setattr(self, self.names[0], value[0])
        setattr(self, self.names[1], value[1])

    def get_raw(self) -> Tuple[ParamRawDataType, ...]:
        return self.value


class E4980AMeasurements:
    """
    All the measurement function for E4980A LCR meter. See user's guide P353
    https://literature.cdn.keysight.com/litweb/pdf/E4980-90230.pdf?id=789356
    """
    CPD = MeasurementPair(
        "CPD", ("capacitance", "dissipation_factor"), ("F", "")
    )
    CPQ = MeasurementPair(
        "CPQ", ("capacitance", "quality_factor"), ("F", "")
    )
    CPG = MeasurementPair(
        "CPG", ("capacitance", "conductance"), ("F", "S")
    )
    CPRP = MeasurementPair(
        "CPRP", ("capacitance", "resistance"), ("F", "Ohm")
    )
    CSD = MeasurementPair(
        "CSD", ("capacitance", "dissipation_factor"), ("F", "")
    )
    CSQ = MeasurementPair(
        'CSQ', ("capacitance", "quality_factor"), ("F", "")
    )
    CSRS = MeasurementPair(
        "CSRS", ("capacitance", "resistance"), ("F", "Ohm")
    )
    LPD = MeasurementPair(
        "LPD", ("inductance", "dissipation_factor"), ("H", "")
    )
    LPQ = MeasurementPair(
        "LPQ", ("inductance", "quality_factor"), ("H", "")
    )
    LPG = MeasurementPair(
        "LPG", ("inductance", "conductance"), ("H", "S")
    )
    LPRP = MeasurementPair(
        "LPRP", ("inductance", "resistance"), ("H", "Ohm")
    )
    LSD = MeasurementPair(
        "LSD", ("inductance", "dissipation_factor"), ("H", "")
    )
    LSQ = MeasurementPair(
        "LSQ", ("inductance", "quality_factor"), ("H", "")
    )
    LSRS = MeasurementPair(
        "LSRS", ("inductance", "resistance"), ("H", "Ohm")
    )
    LSRD = MeasurementPair(
        "LSRD", ("inductance", "resistance"), ("H", "Ohm")
    )
    RX = MeasurementPair(
        "RX", ("resistance", "reactance"), ("Ohm", "Ohm")
    )
    ZTD = MeasurementPair(
        "ZTD", ("impedance", "theta"), ("Ohm", "Degree")
    )
    ZTR = MeasurementPair(
        "ZTR", ("impedance", "theta"), ("Ohm", "Radiant")
    )
    GB = MeasurementPair(
        "GB", ("conductance", "susceptance"), ("S", "S")
    )
    YTD = MeasurementPair(
        "YTD", ("admittance", "theta"), ("Y", "Degree")
    )
    YTR = MeasurementPair(
        "YTR", ("admittance", "theta"), ("Y", "Radiant")
    )
    VDID = MeasurementPair(
        "VDID", ("voltage", "current"), ("V", "A")
    )


class Correction4980A(InstrumentChannel):
    """
    Module for correction settings.
    """
    def __init__(
            self,
            parent: VisaInstrument,
            name: str,
    ) -> None:
        super().__init__(parent, name)

        self.add_parameter(
            "open",
            set_cmd=":CORRection:OPEN",
            docstring="Executes OPEN correction based on all frequency points."
        )

        self.add_parameter(
            "open_state",
            get_cmd=":CORRection:OPEN:STATe?",
            set_cmd=":CORRection:OPEN:STATe {}",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0"),
            docstring="Enables or disable OPEN correction"
        )

        self.add_parameter(
            "short",
            set_cmd=":CORRection:SHORt",
            docstring="Executes SHORT correction based on all frequency points."
        )

        self.add_parameter(
            "short_state",
            get_cmd=":CORRection:SHORt:STATe?",
            set_cmd=":CORRection:SHORt:STATe {}",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0"),
            docstring="Enables or disable SHORT correction."
        )


class KeysightE4980A(VisaInstrument):
    """
    QCodes driver for E4980A Precision LCR Meter
    """
    def __init__(self,
                 name: str,
                 address: str,
                 terminator: str = '\n',
                 **kwargs: Any):
        """
        Create an instance of the instrument.

        Args:
            name: Name of the instrument instance
            address: Visa-resolvable instrument address.
        """
        super().__init__(name, address, terminator=terminator, **kwargs)

        idn = self.IDN.get()

        self.has_firmware_a_02_10_or_above = version.parse(
            convert_legacy_version_to_supported_version(idn["firmware"])
        ) >= version.parse(convert_legacy_version_to_supported_version("A.02.10"))

        self.has_option_001 = '001' in self._options()
        self._dc_bias_v_level_range: Union[Numbers, Enum]
        if self.has_option_001:
            self._v_level_range = Numbers(0, 20)
            self._i_level_range = Numbers(0, 0.1)
            self._imp_range = Enum(0.1, 1, 10, 100, 300, 1000, 3000, 10000,
                                   30000, 100000)
            self._dc_bias_v_level_range = Numbers(-40, 40)
        else:
            self._v_level_range = Numbers(0, 2)
            self._i_level_range = Numbers(0, 0.02)
            self._imp_range = Enum(1, 10, 100, 300, 1000, 3000, 10000, 30000,
                                   100000)
            self._dc_bias_v_level_range = Enum(0, 1.5, 2)

        self._measurement_pair = MeasurementPair(
            "CPD",
            ("capacitance", "dissipation_factor"),
            ("F", "")
        )

        self.add_parameter(
            "frequency",
            get_cmd=":FREQuency?",
            set_cmd=":FREQuency {}",
            get_parser=float,
            unit="Hz",
            vals=Numbers(20, 2E6),
            docstring="Gets and sets the frequency for normal measurement."
        )

        self.add_parameter(
            "current_level",
            get_cmd=self._get_current_level,
            set_cmd=self._set_current_level,
            unit="A",
            vals=self._i_level_range,
            docstring="Gets and sets the current level for measurement signal."
        )

        self.add_parameter(
            "voltage_level",
            get_cmd=self._get_voltage_level,
            set_cmd=self._set_voltage_level,
            unit="V",
            vals=self._v_level_range,
            docstring="Gets and sets the AC bias voltage level for measurement "
                      "signal."
        )

        self.add_parameter(
            "measurement_function",
            get_cmd=":FUNCtion:IMPedance?",
            set_cmd=self._set_measurement
        )

        self.add_parameter(
            "range",
            get_cmd=":FUNCtion:IMPedance:RANGe?",
            set_cmd=self._set_range,
            unit='Ohm',
            vals=self._imp_range,
            docstring="Selects the impedance measurement range, also turns "
                      "the auto range function OFF."
        )

        self.add_parameter(
            "imp_autorange_enabled",
            get_cmd=":FUNCtion:IMPedance:RANGe:AUTO?",
            set_cmd=":FUNCtion:IMPedance:RANGe:AUTO {}",
            val_mapping=create_on_off_val_mapping(on_val="1",
                                                  off_val="0"),
            docstring="Enables the auto-range for impedance measurement."
        )

        self.add_parameter(
            "dc_bias_enabled",
            get_cmd=":BIAS:STATe?",
            set_cmd=":BIAS:STATe {}",
            vals=Bool(),
            val_mapping=create_on_off_val_mapping(on_val="1",
                                                  off_val="0"),
            docstring="Enables DC bias. DC bias is automatically turned "
                      "off after recalling the state from memory."
        )

        self.add_parameter(
            "dc_bias_voltage_level",
            get_cmd=":BIAS:VOLTage:LEVel?",
            set_cmd=":BIAS:VOLTage:LEVel {}",
            get_parser=float,
            unit="V",
            vals=self._dc_bias_v_level_range,
            docstring="Sets the DC bias voltage. Setting does not "
                      "implicitly turn the DC bias ON."
        )

        self.add_parameter(
            "meas_time_mode",
            val_mapping={"short": "SHOR", "medium": "MED", "long": "LONG"},
            parameter_class=GroupParameter
        )

        self.add_parameter(
            "averaging_rate",
            vals=Ints(1, 256),
            parameter_class=GroupParameter,
            get_parser=int,
            docstring="Averaging rate for the measurement."
        )

        self._aperture_group = Group(
            [self.meas_time_mode,
             self.averaging_rate],
            set_cmd=":APERture {meas_time_mode},{averaging_rate}",
            get_cmd=":APERture?"
        )

        if self.has_firmware_a_02_10_or_above:
            self.add_parameter(
                "dc_bias_autorange_enabled",
                get_cmd=":BIAS:RANGe:AUTO?",
                set_cmd=":BIAS:RANGe:AUTO {}",
                vals=Bool(),
                val_mapping=create_on_off_val_mapping(on_val="1",
                                                      off_val="0"),
                docstring="Enables DC Bias range AUTO setting. When DC bias "
                          "range is fixed (not AUTO), '#' is displayed in "
                          "the BIAS field of the display."
            )

        self.add_parameter(
            "signal_mode",
            initial_value=None,
            vals=Enum("Voltage", "Current", None),
            parameter_class=ManualParameter,
            docstring="This parameter tracks the signal mode which is being "
                      "set."
        )

        self.add_submodule(
            "_correction",
            Correction4980A(self, "correction")
        )
        self._set_signal_mode_on_driver_initialization()
        self.connect_message()

    @property
    def correction(self) -> Correction4980A:
        submodule = self.submodules['_correction']
        return cast(Correction4980A, submodule)

    @property
    def measure_impedance(self) -> MeasurementPair:
        return self._get_complex_impedance()

    @property
    def measurement(self) -> MeasurementPair:
        return self._measurement()

    def _set_range(self, val: str) -> None:
        self.write(f":FUNCtion:IMPedance:RANGe {val}")
        self.imp_autorange_enabled.get()

    def _get_complex_impedance(self) -> MeasurementPair:
        """
        Returns the impedance in the format of (R, X), where R is the
        resistance, and X is the reactance.
        """
        measurement = self.ask(":FETCH:IMPedance:CORRected?")
        r, x = (float(n) for n in measurement.split(","))
        measurement_pair = MeasurementPair(
            name="RX",
            names=("resistance", "reactance"),
            units=("Ohm", "Ohm")
        )
        measurement_pair.set((r, x))
        return measurement_pair

    def _measurement(self) -> MeasurementPair:
        """
        Returns a measurement result with the selected measurement function.
        """
        measurement = self.ask(":FETCH:IMPedance:FORMatted?")
        val1, val2, _ = (float(n) for n in measurement.split(","))
        measurement_pair = MeasurementPair(
            name=self._measurement_pair.name,
            names=self._measurement_pair.names,
            units=self._measurement_pair.units
        )
        measurement_pair.set((val1, val2))
        return measurement_pair

    def _set_measurement(self,
                         measurement_pair: MeasurementPair) -> None:
        """
        Selects the measurement function.
        """
        self._measurement_pair = measurement_pair
        self.write(f":FUNCtion:IMPedance {measurement_pair.name}")

    def _get_voltage_level(self) -> float:
        """
        Gets voltage level if signal is set with voltage level parameter
        otherwise raises an error.
        """
        if self.signal_mode() == "Current":
            raise RuntimeError("Cannot get voltage level as signal is set "
                               "with current level parameter.")

        v_level = self.ask(":VOLTage:LEVel?")

        return float(v_level)

    def _set_voltage_level(self, val: str) -> None:
        """
        Sets voltage level
        """
        self.signal_mode("Voltage")
        self.voltage_level.snapshot_exclude = False
        self.current_level.snapshot_exclude = True

        self.write(f":VOLTage:LEVel {val}")

    def _set_current_level(self, val: str) -> None:
        """
        Sets current level
        """
        self.signal_mode("Current")
        self.voltage_level.snapshot_exclude = True
        self.current_level.snapshot_exclude = False

        self.write(f":CURRent:LEVel {val}")

    def _get_current_level(self) -> float:
        """
        Gets current level if signal is set with current level parameter
        otherwise raises an error.
        """
        if self.signal_mode() == "Voltage":
            raise RuntimeError("Cannot get current level as signal is set "
                               "with voltage level parameter.")

        i_level = self.ask(":CURRent:LEVel?")

        return float(i_level)

    def _is_signal_mode_voltage_on_driver_initialization(self) -> bool:
        """
        Checks if signal is set with voltage_level param at instrument driver
        initialization
        """
        assert self.signal_mode() is None
        try:
            self.voltage_level()
            return True
        except VisaIOError:
            return False

    def _set_signal_mode_on_driver_initialization(self) -> None:
        """
        Sets signal mode on driver initialization
        """
        if self._is_signal_mode_voltage_on_driver_initialization():
            self.signal_mode("Voltage")
            self.voltage_level.snapshot_exclude = False
            self.current_level.snapshot_exclude = True
        else:
            self.signal_mode("Current")
            self.voltage_level.snapshot_exclude = True
            self.current_level.snapshot_exclude = False

    def _options(self) -> Tuple[str, ...]:
        """
        Returns installed options numbers. Combinations of different installed
        options are possible. Two of the possible options are Power/DC Bias
        Enhance (option 001) and Bias Current Interface (option 002).
        """
        options_raw = self.ask('*OPT?')
        return tuple(options_raw.split(','))

    def system_errors(self) -> str:
        """
        Returns the oldest unread error message from the event log and removes
        it from the log.
        """
        return self.ask(":SYSTem:ERRor?")

    def clear_status(self) -> None:
        """
        Clears the following:
            Error Queue
            Status Byte Register
            Standard Event Status Register
            Operation Status Event Register
            Questionable Status Event Register (No Query)
        """
        self.write('*CLS')

    def reset(self) -> None:
        """
        Resets the instrument settings.
        """
        self.write('*RST')
