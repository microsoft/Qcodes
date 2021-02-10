import numpy as np
from qcodes.instrument.visa import VisaInstrument
from qcodes.utils import validators as vals


class SP893c(VisaInstrument):
    def __init__(self, name, address):
        super().__init__(name, address, terminator="\r\n")

        self.add_parameter(
            "gain",
            label="Gain",
            set_cmd=self.set_gain,
            get_cmd=self.get_gain,
            vals=vals.Enum(1e5, 1e6, 1e7, 1e8, 1e9),
        )
        self.add_parameter(
            "filter_f",
            unit="Hz",
            label="Filter Cut-Off Frequency",
            set_cmd=self.set_filter,
            get_cmd=self.get_filter,
            vals=vals.Enum(30, 100, 300, 1000, 3000, 30_000, 100_000, "FULL"),
        )
        self.add_parameter(
            "overload_status", label="Overload Status", set_cmd=None, get_cmd="GET O"
        )

    def set_gain(self, value):
        r = self.ask(f"SET G 1E{int(np.log10(value))}")
        if r != "OK":
            raise ValueError(f"Expected OK return but got: {r}")

    def get_gain(self):
        s = self.ask("GET G")
        r = s.split("Gain: ")[1]
        return float(r)

    def set_filter(self, value):
        r = self.ask(f"SET F {value}")
        if r != "OK":
            raise ValueError(f"Expected OK return but got: {r}")

    def get_filter(self):
        s = self.ask("GET F")
        r = s.split("Filter: ")[1]
        if r == "Full":
            return r.upper()
        elif r[-3::] == "kHz":
            return float(r[0:-3]) * 1e3
        elif r[-2::] == "Hz":
            return float(r[0:-2])
        else:
            raise ValueError(f"Could not interpret result. Got: {s}")
