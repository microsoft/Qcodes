from typing import TYPE_CHECKING

from .AWG70000A import TektronixAWG70000Base

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.instrument import VisaInstrumentKWArgs


class TektronixAWG70002B(TektronixAWG70000Base):
    """
    The QCoDeS driver for Tektronix AWG70002B series AWG's.

    All the actual driver meat is in the superclass AWG70000A.
    """

    default_timeout = 10

    def __init__(
        self,
        name: str,
        address: str,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ) -> None:
        """
        Args:
            name: The name used internally by QCoDeS in the DataSet
            address: The VISA resource name of the instrument
            **kwargs: kwargs are forwarded to base class.
        """

        super().__init__(name, address, num_channels=2, **kwargs)
