from typing import Any

import numpy as np

from qcodes.instrument.parameter import Parameter
from qcodes.instrument.visa import VisaInstrument
from qcodes.utils import validators as vals


class Weinschel_8320(VisaInstrument):
    """
    QCodes driver for the stepped attenuator
    Weinschel is formerly known as Aeroflex/Weinschel
    """

    def __init__(self, name: str, address: str, **kwargs: Any):
        super().__init__(name, address, terminator='\r', **kwargs)
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
