from typing import TYPE_CHECKING

from . import N52xx

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.instrument import VisaInstrumentKWArgs

#  This is not the same class of Keysight devices but seems to work well...


class KeysightP9374A(N52xx.KeysightPNAxBase):
    def __init__(
        self, name: str, address: str, **kwargs: "Unpack[VisaInstrumentKWArgs]"
    ):
        super().__init__(
            name,
            address,
            min_freq=300e6,
            max_freq=20e9,
            min_power=-43,
            max_power=20,
            nports=2,
            **kwargs
        )

        self.get_options()
