import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

import qcodes.validators as vals
from qcodes.instrument import VisaInstrument, VisaInstrumentKWArgs
from qcodes.parameters import Parameter, create_on_off_val_mapping

if TYPE_CHECKING:
    from typing_extensions import Unpack


class AgilentE8257D(VisaInstrument):
    """
    This is the QCoDeS driver for the Agilent E8257D signal generator.
    This driver will most likely work for multiple Agilent sources.
    This driver does not contain all commands available for the E8257D but
    only the ones most commonly used.
    """

    default_terminator = "\n"

    def __init__(
        self,
        name: str,
        address: str,
        step_attenuator: bool | None = None,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ) -> None:
        super().__init__(name, address, **kwargs)

        if step_attenuator is not None:
            warnings.warn(
                "step_attenuator argument to E8257D is deprecated "
                "and has no effect. It will be removed in the "
                "future."
            )

        # Query installed options
        self._options = self.ask_raw("DIAG:CPU:INFO:OPT:DET?")

        # Determine installed frequency option
        frequency_option = None
        for f_option in ["513", "520", "521", "532", "540", "550", "567"]:
            if f_option in self._options:
                frequency_option = f_option
        if frequency_option is None:
            raise RuntimeError("Could not determine the frequency option")

        # convert installed frequency option to frequency ranges, based on:
        # https://www.keysight.com/us/en/assets/7018-01233/configuration-guides
        # /5989-1325.pdf
        # the frequency range here is the max range and not the specified
        # (calibrated) one
        f_options_dict = {
            "513": (100e3, 13e9),
            "520": (100e3, 20e9),
            "521": (10e6, 20e9),
            "532": (100e3, 31.8e9),
            "540": (100e3, 40e9),
            "550": (100e3, 50e9),
            "567": (100e3, 70e9),
        }

        # assign min and max frequencies
        self._min_freq: float
        self._max_freq: float
        self._min_freq, self._max_freq = f_options_dict[frequency_option]

        # Based on installed frequency option and presence/absence of step
        # attenuator (option '1E1') determine power range based on:
        # https://www.keysight.com/us/en/assets/7018-01211/data-sheets
        # /5989-0698.pdf

        # assign min and max powers
        self._min_power: float
        self._max_power: float

        if "1E1" in self._options:
            if frequency_option in ["513", "520", "521", "532", "540"]:
                self._min_power = -135
                self._max_power = 10
            else:
                self._min_power = -110
                self._max_power = 5
        elif frequency_option in ["513", "520", "521", "532", "540"]:
            self._min_power = -20
            self._max_power = 10
        else:
            # default minimal power is -20 dBm
            self._min_power = -20
            self._max_power = 5

        self.frequency: Parameter = self.add_parameter(
            name="frequency",
            label="Frequency",
            unit="Hz",
            get_cmd="FREQ:CW?",
            set_cmd="FREQ:CW" + " {:.4f}",
            get_parser=float,
            set_parser=float,
            vals=vals.Numbers(self._min_freq, self._max_freq),
        )
        """Parameter frequency"""

        self.phase: Parameter = self.add_parameter(
            name="phase",
            label="Phase",
            unit="deg",
            get_cmd="PHASE?",
            set_cmd="PHASE" + " {:.8f}",
            get_parser=self.rad_to_deg,
            set_parser=self.deg_to_rad,
            vals=vals.Numbers(-180, 180),
        )
        """Parameter phase"""

        self.power: Parameter = self.add_parameter(
            name="power",
            label="Power",
            unit="dBm",
            get_cmd="POW:AMPL?",
            set_cmd="POW:AMPL" + " {:.4f}",
            get_parser=float,
            set_parser=float,
            vals=vals.Numbers(self._min_power, self._max_power),
        )
        """Parameter power"""

        self.output_enabled: Parameter = self.add_parameter(
            "output_enabled",
            get_cmd=":OUTP?",
            set_cmd="OUTP {}",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0"),
        )
        """Parameter output_enabled"""

        self.connect_message()

    def on(self) -> None:
        self.output_enabled.set("on")

    def off(self) -> None:
        self.output_enabled.set("off")

    # functions to convert between rad and deg
    @staticmethod
    def deg_to_rad(
        angle_deg: float | str | np.floating | np.integer,
    ) -> "np.floating[Any]":
        return np.deg2rad(float(angle_deg))

    @staticmethod
    def rad_to_deg(
        angle_rad: float | str | np.floating | np.integer,
    ) -> "np.floating[Any]":
        return np.rad2deg(float(angle_rad))
