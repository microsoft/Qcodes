from typing import TYPE_CHECKING

from ._s2vna_base import CopperMountainS2VnaBase

if TYPE_CHECKING:
    from typing import Unpack

    from qcodes.instrument import VisaInstrumentKWArgs


class CopperMountainM5180(CopperMountainS2VnaBase):
    """This is the QCoDeS driver for the M5180 VNA from Copper Mountain Technologies"""

    def __init__(
        self,
        name: str,
        address: str,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ):
        super().__init__(
            name,
            address,
            min_freq=300e3,
            max_freq=18e9,
            min_power=-40,
            max_power=10,
            max_points_per_sweep=200_001,
            **kwargs,
        )
