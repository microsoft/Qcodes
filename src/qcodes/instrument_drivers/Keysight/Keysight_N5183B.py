from typing import TYPE_CHECKING

from qcodes.instrument_drivers.Keysight.N51x1 import KeysightN51x1

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.instrument import VisaInstrumentKWArgs


class KeysightN5183B(KeysightN51x1):
    def __init__(
        self, name: str, address: str, **kwargs: "Unpack[VisaInstrumentKWArgs]"
    ):
        super().__init__(name, address, min_power=-20, max_power=19, **kwargs)


class N5183B(KeysightN5183B):
    """ "
    Alias for backwards compatiblitly

    """
