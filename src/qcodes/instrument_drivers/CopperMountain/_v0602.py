from typing import TYPE_CHECKING

from ._s2vna_base import CopperMountainS2VnaBase

if TYPE_CHECKING:
    from typing import Unpack

    from qcodes.instrument import VisaInstrumentKWArgs


class CopperMountainV0602(CopperMountainS2VnaBase):
    """This is the QCoDeS driver for the V0602 VNA from Copper Mountain Technologies"""

    def __init__(
        self,
        name: str,
        address: str,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ):
        super().__init__(
            name,
            address,
            min_freq=100e3,
            max_freq=6.5e9,
            min_power=-50.0,
            max_power=5.0,
            max_points_per_sweep=500_001,
            **kwargs,
        )
