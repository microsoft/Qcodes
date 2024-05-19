
from typing import TYPE_CHECKING

from qcodes.instrument import VisaInstrument, VisaInstrumentKWArgs
from qcodes.parameters import Parameter
from qcodes.validators import Enum

if TYPE_CHECKING:
    from typing_extensions import Unpack


class Agilent344xxA(VisaInstrument):
    """
    This is the base class for QCoDeS driver for the Agilent 34400A DMM Series.
    This class should not be instantiated directly but a specific sub model should
    be used.

    Note that most models are better supported by the Keysight 33xxA drivers.
    """

    default_terminator = "\n"

    def __init__(
        self,
        name: str,
        address: str,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ) -> None:
        super().__init__(name, address, **kwargs)

        idn = self.IDN.get()
        self.model = idn["model"]

        NPLC_list = {
            "34401A": [0.02, 0.2, 1, 10, 100],
            "34410A": [0.006, 0.02, 0.06, 0.2, 1, 2, 10, 100],
            "34411A": [0.001, 0.002, 0.006, 0.02, 0.06, 0.2, 1, 2, 10, 100],
        }[self.model]

        self._resolution_factor = {
            "34401A": [1e-4, 1e-5, 3e-6, 1e-6, 3e-7],
            "34410A": [6e-06, 3e-06, 1.5e-06, 7e-07, 3e-07, 2e-07, 1e-07],
            "34411A": [
                3e-05,
                1.5e-05,
                6e-06,
                3e-06,
                1.5e-06,
                7e-07,
                3e-07,
                2e-07,
                1e-07,
                3e-08,
            ],
        }[self.model]

        self.resolution = Parameter(
            "resolution",
            instrument=self,
            get_cmd="VOLT:DC:RES?",
            get_parser=float,
            set_cmd=self._set_resolution,
            label="Resolution",
            unit="V",
        )
        """Resolution """

        self.volt: Parameter = self.add_parameter(
            "volt", get_cmd="READ?", label="Voltage", get_parser=float, unit="V"
        )
        """Parameter volt"""

        self.fetch: Parameter = self.add_parameter(
            "fetch",
            get_cmd="FETCH?",
            label="Voltage",
            get_parser=float,
            unit="V",
            snapshot_get=False,
            docstring=(
                "Reads the data you asked for, i.e. "
                "after an `init_measurement()` you can "
                "read the data with fetch.\n"
                "Do not call this when you did not ask "
                "for data in the first place!"
            ),
        )
        """Parameter fetch"""

        self.NPLC: Parameter = self.add_parameter(
            "NPLC",
            get_cmd="VOLT:NPLC?",
            get_parser=float,
            set_cmd=self._set_nplc,
            vals=Enum(*NPLC_list),
            label="Integration time",
            unit="NPLC",
        )
        """Parameter NPLC"""

        self.terminals: Parameter = self.add_parameter(
            "terminals", get_cmd="ROUT:TERM?"
        )
        """Parameter terminals"""

        self.range_auto: Parameter = self.add_parameter(
            "range_auto",
            get_cmd="VOLT:RANG:AUTO?",
            set_cmd="VOLT:RANG:AUTO {:d}",
            val_mapping={"on": 1, "off": 0},
        )
        """Parameter range_auto"""

        self.range: Parameter = self.add_parameter(
            "range",
            get_cmd="SENS:VOLT:DC:RANG?",
            get_parser=float,
            set_cmd="SENS:VOLT:DC:RANG {:f}",
            vals=Enum(0.1, 1.0, 10.0, 100.0, 1000.0),
        )
        """Parameter range"""

        self.connect_message()

    def _set_nplc(self, value: float) -> None:
        self.write(f"VOLT:NPLC {value:f}")
        # resolution settings change with NPLC
        self.resolution.get()

    def _set_resolution(self, value: float) -> None:
        rang = self.range.get()
        # convert both value*range and the resolution factors
        # to strings with few digits, so we avoid floating point
        # rounding errors.
        res_fac_strs = [f"{v * rang:.1e}" for v in self._resolution_factor]
        if f"{value:.1e}" not in res_fac_strs:
            raise ValueError(
                f"Resolution setting {value:.1e} ({value} at range {rang}) "
                "does not exist. "
                f"Possible values are {res_fac_strs}"
            )
        self.write(f"VOLT:DC:RES {value:.1e}")
        # NPLC settings change with resolution
        self.NPLC.get()

    def _set_range(self, value: float) -> None:
        self.write(f"SENS:VOLT:DC:RANG {value:f}")
        # resolution settings change with range
        self.resolution.get()

    def clear_errors(self) -> None:
        while True:
            err = self.ask("SYST:ERR?")
            if "No error" in err:
                return
            print(err)

    def init_measurement(self) -> None:
        self.write("INIT")

    def display_clear(self) -> None:
        if self.model in ["34401A"]:
            lines = ["WIND"]
        elif self.model in ["34410A", "34411A"]:
            lines = ["WIND1", "WIND2"]
        else:
            raise ValueError("unrecognized model: " + str(self.model))

        for line in lines:
            self.write("DISP:" + line + ":TEXT:CLE")
            self.write("DISP:" + line + ":STAT 1")

    def reset(self) -> None:
        self.write("*RST")
