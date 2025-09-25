from typing import TYPE_CHECKING

from ._M5xxx import CopperMountainM5xxx

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.instrument import VisaInstrumentKWArgs


class CopperMountainM5065(CopperMountainM5xxx):
    """This is the QCoDeS driver for the M5065 VNA from Copper Mountain Technologies"""

    def __init__(
        self,
        name: str,
        address: str,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ):
        super().__init__(name, address, min_freq=300e3, max_freq=6.5e9, **kwargs)
