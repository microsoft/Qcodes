from typing import Tuple, Sequence, cast, Any

from qcodes import VisaInstrument, InstrumentChannel
from qcodes.instrument.parameter import MultiParameter
from qcodes.utils.helpers import create_on_off_val_mapping
from qcodes.utils.validators import Enum, Numbers


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

    def get_raw(self) -> tuple:
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
            get_cmd=":CURRent:LEVel?",
            set_cmd=":CURRent:LEVel {}",
            get_parser=float,
            unit="A",
            vals=Numbers(0, 0.1),
            docstring="Gets and sets the current level for measurement signal."
        )

        self.add_parameter(
            "voltage_level",
            get_cmd=":VOLTage:LEVel?",
            set_cmd=":VOLTage:LEVel {}",
            get_parser=float,
            unit="V",
            vals=Numbers(0, 20),
            docstring="Gets and sets the voltage level for measurement signal."
        )

        self.add_parameter(
            "measurement_function",
            get_cmd=":FUNCtion:IMPedance?",
            set_cmd=self._set_measurement
        )

        self.add_parameter(
            "range",
            get_cmd=":FUNCtion:IMPedance:RANGe?",
            set_cmd=":FUNCtion:IMPedance:RANGe {}",
            unit='Ohm',
            vals=Enum(0.1, 1, 10, 100, 300, 1000, 3000, 10000, 30000, 100000),
            docstring="Selects the impedance measurement range, also turns "
                      "the auto range function OFF."
        )

        self.add_submodule(
            "_correction",
            Correction4980A(self, "correction")
        )

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

    def _get_complex_impedance(self) -> MeasurementPair:
        """
        Returns the impedance in the format of (R, X), where R is the
        resistance, and X is the reactance.
        """
        measurement = self.ask(":FETCH:IMPedance:CORRected?")
        r, x = [float(n) for n in measurement.split(",")]
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
        val1, val2, _ = [float(n) for n in measurement.split(",")]
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
