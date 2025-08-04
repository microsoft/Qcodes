from typing import TYPE_CHECKING

from . import N52xx

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.instrument import VisaInstrumentKWArgs


class KeysightP5004B(N52xx.KeysightPNAxBase):
    """
    Driver for the Keysight P5004B Vector Network Analyzer. (see: https://www.keysight.com/us/en/assets/3121-1235/data-sheets/Streamline-Series-Vector-Network-Analyzer-B-models.pdf for datasheet.)
    Power range is -100 dBm to +20 dBm (see "Table 22. Power Resolution, Maximum/minimum Settable Power" on page 23 of the datasheet).
    Frequency range is 9 kHz to 20 GHz (see https://www.keysight.com/us/en/product/P5004B/streamline-vector-network-analyzer-9-khz-to-20-ghz-2-port.html )
    """

    def __init__(
        self, name: str, address: str, **kwargs: "Unpack[VisaInstrumentKWArgs]"
    ):
        super().__init__(
            name,
            address,
            min_freq=9e3,
            max_freq=20e9,
            min_power=-100,
            max_power=20,
            nports=2,
            **kwargs,
        )
