from typing import TYPE_CHECKING

from qcodes.instrument_drivers.stanford_research.SR86x import SR86x

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.instrument import VisaInstrumentKWArgs


class SR865(SR86x):
    """
    QCoDeS driver for the Stanford Research Systems SR865 Lock-in Amplifier.

    The SR865 instrument is almost equal to the SR860, except for the max frequency
    """

    def __init__(
        self,
        name: str,
        address: str,
        reset: bool = False,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ) -> None:
        super().__init__(name, address, max_frequency=2e6, reset=reset, **kwargs)
