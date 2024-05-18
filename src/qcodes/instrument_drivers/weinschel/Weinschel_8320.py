from typing import TYPE_CHECKING

import numpy as np

from qcodes import validators as vals
from qcodes.instrument import VisaInstrument, VisaInstrumentKWArgs
from qcodes.parameters import Parameter

if TYPE_CHECKING:
    from typing_extensions import Unpack


class Weinschel8320(VisaInstrument):
    """
    QCodes driver for the Weinschel 8320 stepped attenuator.

    Weinschel is formerly known as Aeroflex/Weinschel
    """

    default_terminator = "\r"

    def __init__(
        self, name: str, address: str, **kwargs: "Unpack[VisaInstrumentKWArgs]"
    ):
        super().__init__(name, address, **kwargs)
        self.attenuation = Parameter(
            "attenuation",
            unit="dB",
            set_cmd="ATTN ALL {0:0=2d}",
            get_cmd="ATTN? 1",
            vals=vals.Enum(*np.arange(0, 60.1, 2).tolist()),
            instrument=self,
            get_parser=float,
        )
        """Control the attenuation"""

        self.connect_message()


class Weinschel_8320(Weinschel8320):
    """
    Alias for Weinschel8320 will eventually be deprecated and removed.
    """
