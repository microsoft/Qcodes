from typing import Any, Dict, Optional
import numpy as np

from qcodes.instrument.visa import VisaInstrument
from qcodes.instrument.parameter import Parameter, DelegateParameter
from qcodes.utils import validators as vals


class SP983A(VisaInstrument):
    """
    A driver for Basel Preamp's (SP983C) Remote Instrument - Model SP983A.

    Args:
        name: name for your instrument driver instance
        address: address of the connected remote controller of basel preamp
        input_offset_voltage: (Optional) A source input offset voltage
            parameter. The range for input is -10 to 10 Volts and it is
            user's responsibility to ensure this. This source parameter is
            used to set offset voltage parameter of the preamp and the
            source parameter should represent a voltage source that is
            connected to the "Offset Input Volgate" connector of the SP983C.
    """
    def __init__(self,
                 name: str,
                 address: str,
                 input_offset_voltage: Optional[Parameter] = None,
                 terminator: str = "\r\n",
                 **kwargs: Any) -> None:
        super().__init__(name, address, terminator=terminator, **kwargs)

        self.connect_message()

        self.add_parameter(
            "gain",
            label="Gain",
            unit="V/A",
            set_cmd=self._set_gain,
            get_cmd=self._get_gain,
            vals=vals.Enum(1e5, 1e6, 1e7, 1e8, 1e9),
        )
        self.add_parameter(
            "fcut",
            unit="Hz",
            label="Filter Cut-Off Frequency",
            get_cmd=self._get_filter,
            get_parser=self._parse_filter_value,
            set_cmd=self._set_filter,
            val_mapping={30: '30', 100: '100', 300: '300', 1000: '1k',
                         3000: '3k', 10e3: '10k', 30e3: '30k',
                         100e3: '100k', 1e6: 'FULL'},
        )
        self.add_parameter(
            "overload_status",
            label="Overload Status",
            set_cmd=False,
            get_cmd="GET O"
        )

        self.add_parameter(
            "offset_voltage",
            label="Offset Voltage for SP983C",
            unit="V",
            vals=vals.Numbers(-0.1, 0.1),
            scale=100,
            source=input_offset_voltage,
            parameter_class=DelegateParameter)

    def get_idn(self) -> Dict[str, Optional[str]]:
        vendor = 'Physics Basel'
        model = 'SP 983A'
        serial = None
        firmware = None
        return {'vendor': vendor, 'model': model,
                'serial': serial, 'firmware': firmware}

    def _set_gain(self, value: float) -> None:
        r = self.ask(f"SET G 1E{int(np.log10(value))}")
        if r != "OK":
            raise ValueError(f"Expected OK return but got: {r}")

    def _get_gain(self) -> float:
        s = self.ask("GET G")
        r = s.split("Gain: ")[1]
        return float(r)

    def _set_filter(self, value: str) -> None:
        r = self.ask(f"SET F {value}")
        if r != "OK":
            raise ValueError(f"Expected OK return but got: {r}")

    def _get_filter(self) -> str:
        s = self.ask("GET F")
        return s.split("Filter: ")[1]

    @staticmethod
    def _parse_filter_value(val: str) -> str:
        if val.startswith("F"):
            return "FULL"
        elif val[-3::] == "kHz":
            return str(int(val[0:-3]))+'k'

        elif val[-2::] == "Hz":
            return str(int(val[0:-2]))
        else:
            raise ValueError(f"Could not interpret result. Got: {val}")
