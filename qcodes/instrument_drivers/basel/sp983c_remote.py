from typing import Any, Union, Dict, Optional
import numpy as np

from qcodes.instrument.visa import VisaInstrument
from qcodes.utils import validators as vals


class SP983A(VisaInstrument):
    def __init__(self, name: str, address: str, **kwargs: Any) -> None:
        super().__init__(name, address, terminator="\r\n", **kwargs)

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
            vals={30: '30', 100: '100', 300: '300', 1000: '1000', 3000: '3000',
                  10e3: '10k', 30e3: '30k', 100e3: '100k', 'Full': 'FULL'},
        )
        self.add_parameter(
            "overload_status",
            label="Overload Status",
            set_cmd=None,
            get_cmd="GET O"
        )

    def get_idn(self) -> Dict[str, Optional[str]]:
        vendor = 'Physics Basel'
        model = 'SP 983(a)'
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
    def _parse_filter_value(val: str) -> Union[str, float]:
        if val.startswith("F"):
            return "Full"
        elif val[-3::] == "kHz":
            return float(val[0:-3]) * 1e3
        elif val[-2::] == "Hz":
            return float(val[0:-2])
        else:
            raise ValueError(f"Could not interpret result. Got: {val}")
