from typing import TYPE_CHECKING

from qcodes.instrument import VisaInstrumentKWArgs
from qcodes.validators import Strings

from ._Agilent_344xxA import Agilent344xxA

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.instrument import VisaInstrumentKWArgs
    from qcodes.parameters import Parameter


class Agilent34410A(Agilent344xxA):
    """
    This is the QCoDeS driver for the Agilent 34410A DMM.
    """

    def __init__(
        self,
        name: str,
        address: str,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ) -> None:
        super().__init__(name, address, **kwargs)

        self.display_text: Parameter = self.add_parameter(
            "display_text",
            get_cmd="DISP:WIND1:TEXT?",
            set_cmd='DISP:WIND1:TEXT "{}"',
            vals=Strings(),
        )
        """Parameter display_text"""

        self.display_text_2: Parameter = self.add_parameter(
            "display_text_2",
            get_cmd="DISP:WIND2:TEXT?",
            set_cmd='DISP:WIND2:TEXT "{}"',
            vals=Strings(),
        )
        """Parameter display_text_2"""
