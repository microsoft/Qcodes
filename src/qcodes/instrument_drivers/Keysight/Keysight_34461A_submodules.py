from typing import TYPE_CHECKING

from typing_extensions import deprecated

from qcodes.utils import QCoDeSDeprecationWarning

from .private.Keysight_344xxA_submodules import Keysight344xxA

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.instrument import VisaInstrumentKWArgs


class Keysight34461A(Keysight344xxA):
    """
    This is the qcodes driver for the Keysight 34461A Multimeter
    """

    def __init__(
        self,
        name: str,
        address: str,
        silent: bool = False,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ):
        super().__init__(name, address, silent, **kwargs)


@deprecated("Use Keysight34461A", category=QCoDeSDeprecationWarning, stacklevel=2)
class Keysight_34461A(Keysight344xxA):
    """
    Alias for backwards compatibility.
    """
