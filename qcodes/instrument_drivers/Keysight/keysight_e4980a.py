from qcodes import VisaInstrument, InstrumentChannel
from qcodes.utils.validators import Enum, Numbers


Measurement_Function = {
    # CP vs CS: P means measured using parallel equivalent circuit model,
    #           S means measured using series equivalent circuit model.
    # Same for LP and LS
    # RP vs RS: Equivalent parallel/series resistance
    "CPD": "Capacitance - Dissipation factor",
    "CPQ": "Capacitance - Quality factor",
    "CPG": "Capacitance - Conductance",
    "CPRP": "Capacitance - Resistance",
    "CSD": "Capacitance - Dissipation factor",
    "CSQ": "Capacitance - Quality factor",
    "CSRS": "Capacitance - Resistance",
    "LPD": "Inductance - Dissipation factor",
    "LPQ": "Inductance - Quality factor",
    "LPG": "Inductance - Conductance",
    "LPRP": "Inductance - Resistance",
    "LPRD": "Inductance - DC resistance",
    "LSD": "Inductance - Dissipation factor",
    "LSQ": "Inductance - Quality factor",
    "LSRS": "Inductance - Resistance",
    "LSRD": "Inductance - DC resistance",
    "RX": "Resistance - Reactance",
    "ZTD": "Absolute value of impedance - thd",
    "ZTR": "Absolute value of impedance - thr",
    "GB": "Conductance - Sustenance",
    "YTD": "Absolute value of admittance - thd",
    "YTR": "Absolute value of admittance - thr",
    "VDID": "DC voltage - DC current"
}


class Correction4980A(InstrumentChannel):
    """

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

        self._measurement_function = "CPD"

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
            get_cmd=self._get_complex_impedance
        )

        self.add_parameter(
            "measurement_function",
            get_cmd=":FUNCtion:IMPedance?",
            set_cmd=self._set_measurement,
            vals=Enum("CPD", "CPD", "CPQ", "CPG", "CPRP", "CSD", "CSQ", "CSRS",
                      "LPD", "LPQ", "LPG", "LPRP", "LPRD", "LSD", "LSQ",
                      "LSRS", "LSRD", "RX", "ZTD", "ZTR", "GB", "YTD", "YTR",
                      "VDID", "list")

        )

        self.add_parameter(
            "measure",
            get_cmd=self._measurement
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

    def _get_complex_impedance(self) -> dict:
        """
        Returns a complex measurement result (R-X format).
        """
        measurement = self.ask(":FETCH:IMPedance:CORRected?")
        r, x = [float(n) for n in measurement.split(",")]
        return {"Resistance": r, "Reactance": x}

    def _measurement(self) -> dict:
        """
        Returns a measurement result with the selected measurement function.
        """
        measurement = self.ask(":FETCH:IMPedance:FORMatted?")
        func = self._measurement_function
        val1, val2, _ = [float(n) for n in measurement.split(",")]
        key1, key2 = [
            key.strip() for key in Measurement_Function[func].split('-')
        ]
        return {key1: val1, key2: val2}

    def _set_measurement(self, measurement_function: str) -> None:
        """
        Selects the measurement function.
        """
        if measurement_function == 'list':
            for key, value in Measurement_Function.items():
                print(f"{key}: {value}")
        else:
            self._measurement_function = measurement_function
            self.write(f":FUNCtion:IMPedance {measurement_function}")

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
