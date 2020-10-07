from typing import Tuple

from qcodes import VisaInstrument, InstrumentChannel
from qcodes.instrument.parameter import MultiParameter
from qcodes.utils.validators import Enum, Numbers


class MeasurementFunction(MultiParameter):
    def __init__(self,
                 name: str,
                 names: tuple,
                 units: tuple):
        super().__init__(name=name,
                         names=names,
                         shapes=((), ()),
                         units=units)

    def set_raw(self, value: any) -> None:
        raise ValueError("Measurement function can not be modified.")

    def get_raw(self) -> str:
        return self.name


class MeasurementPair(MultiParameter):
    value = ()

    def __init__(self,
                 measurement_function: MeasurementFunction):
        super().__init__(name=measurement_function.name,
                         names=measurement_function.names,
                         shapes=((), ()),
                         units=measurement_function.units)
        self.__dict__.update(
            {measurement_function.names[0]: 0,
             measurement_function.names[1]: 0}
        )

    def set_raw(self, value: Tuple[float, float]) -> None:
        self.value = value
        setattr(self, self.names[0], value[0])
        setattr(self, self.names[1], value[1])

    def get_raw(self):
        return self.value


class E4980AMeasurements:
    CPD = MeasurementFunction("CPD", ("capacitance", "dissipation_factor"), ("F", ""))
    CPQ = MeasurementFunction("CPQ", ("capacitance", "quality_factor"), ("F", ""))
    CPG = MeasurementFunction("CPG", ("capacitance", "conductance"), ("F", "S"))
    CPRP = MeasurementFunction("CPRP", ("capacitance", "resistance"), ("F", "Ohm"))
    CSD = MeasurementFunction("CSD", ("capacitance", "dissipation_factor"), ("F", ""))
    CSQ = MeasurementFunction('CSQ', ("capacitance", "quality_factor"), ("F", ""))
    CSRS = MeasurementFunction("CSRS", ("capacitance", "resistance"), ("F", "Ohm"))
    LPD = MeasurementFunction("LPD", ("inductance", "dissipation_factor"), ("H", ""))
    LPQ = MeasurementFunction("LPQ", ("inductance", "quality_factor"), ("H", ""))
    LPG = MeasurementFunction("LPG", ("inductance", "conductance"), ("H", "S"))
    LPRP = MeasurementFunction("LPRP", ("inductance", "resistance"), ("H", "Ohm"))
    LSD = MeasurementFunction("LSD", ("inductance", "dissipation_factor"), ("H", ""))
    LSQ = MeasurementFunction("LSQ", ("inductance", "quality_factor"), ("H", ""))
    LSRS = MeasurementFunction("LSRS", ("inductance", "resistance"), ("H", "Ohm"))
    LSRD = MeasurementFunction("LSRD", ("inductance", "resistance"), ("H", "Ohm"))
    RX = MeasurementFunction("RX", ("resistance", "reactance"), ("Ohm", "Ohm"))
    ZTD = MeasurementFunction("ZTD", ("impedance", "theta"), ("Ohm", "Degree"))
    ZTR = MeasurementFunction("ZTR", ("impedance", "theta"), ("Ohm", "Radiant"))
    GB = MeasurementFunction("GB", ("conductance", "susceptance"), ("S", "S"))
    YTD = MeasurementFunction("YTD", ("admittance", "theta"), ("Y", "Degree"))
    YTR = MeasurementFunction("YTR", ("admittance", "theta"), ("Y", "Radiant"))
    VDID = MeasurementFunction("VDID", ("voltage", "current"), ("V", "A"))


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
            val_mapping={
                'off': 0,
                'on': 1,
            },
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
            val_mapping={
                'off': 0,
                'on': 1,
            },
            docstring="Enables or disable SHORT correction."
        )


class KeysightE4980A(VisaInstrument):
    """
    QCodes driver for E4980A Precision LCR Meter
    """
    def __init__(self, name, address, terminator='\n', **kwargs):
        """
        Create an instance of the instrument.

        Args:
            name: Name of the instrument instance
            address: Visa-resolvable instrument address.
        """
        super().__init__(name, address, terminator=terminator, **kwargs)

        self._measurement_function = E4980AMeasurements.CPD

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
            "impedance",
            get_cmd=self._get_complex_impedance,
        )

        self.add_parameter(
            "measurement_function",
            get_cmd=":FUNCtion:IMPedance?",
            set_cmd=self._set_measurement
        )

        self.add_parameter(
            "measure",
            get_cmd=self._measurement,
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

        self.add_parameter(
            "system_errors",
            get_cmd=":SYSTem:ERRor?",
            docstring="Returns the oldest unread error message from the event "
                      "log and removes it from the log."
        )

        self.add_submodule(
            "_correction",
            Correction4980A(self, "correction")
        )

        self.connect_message()

    @property
    def correction(self):
        return self.submodules['_correction']

    def _get_complex_impedance(self) -> MeasurementPair:
        """
        Returns a complex measurement result (R-X format).
        """
        measurement = self.ask(":FETCH:IMPedance:CORRected?")
        r, x = [float(n) for n in measurement.split(",")]
        measurement_pair = MeasurementPair(E4980AMeasurements.RX)
        measurement_pair.set((r, x))
        return measurement_pair

    def _measurement(self) -> MeasurementPair:
        """
        Returns a measurement result with the selected measurement function.
        """
        measurement = self.ask(":FETCH:IMPedance:FORMatted?")
        # p1, p2 = self._get_parameters_from_measurement_function()
        val1, val2, _ = [float(n) for n in measurement.split(",")]
        measurement_pair = MeasurementPair(self._measurement_function)
        measurement_pair.set((val1, val2))
        return measurement_pair

    def _set_measurement(self,
                         measurement_function: MeasurementFunction) -> None:
        """
        Selects the measurement function.
        """
        self._measurement_function = measurement_function
        self.write(f":FUNCtion:IMPedance {measurement_function.name}")

    def clear_status(self) -> None:
        """
        Clears the following:
            • Error Queue
            • Status Byte Register
            • Standard Event Status Register
            • Operation Status Event Register
            • Questionable Status Event Register (No Query)
        """
        self.write('*CLS')

    def reset(self) -> None:
        """
        Resets the instrument settings.
        """
        self.write('*RST')
