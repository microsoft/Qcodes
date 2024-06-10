from typing import TYPE_CHECKING

from .AWG70000A import TektronixAWG70000Base

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.instrument import VisaInstrumentKWArgs


class TektronixAWG5208(TektronixAWG70000Base):
    """
    The QCoDeS driver for Tektronix AWG5208
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

        super().__init__(name, address, num_channels=8, **kwargs)


class AWG5208(TektronixAWG5208):
    """
    Alias with non-conformant name left for backwards compatibility
    """

    pass
