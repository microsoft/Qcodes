"""
This file holds the QCoDeS driver for the Galil DMC-41x3 motor controllers,
colloquially known as the "stepper motors".
"""
from typing import Any, Dict, Optional, List

from qcodes.instrument.visa import VisaInstrument


class DMC41x3(VisaInstrument):
    """
    Driver for Galil DMC-41x3 Motor Controller
    """

    def __init__(self, name: str, address: str, **kwargs: Any) -> None:
        super().__init__(name=name, address=address, **kwargs)

        self.connect_message()

    def get_idn(self) -> Dict[str, Optional[str]]:
        with self.timeout.set_to(5):
            self.write("ID")
            data = str(self.visa_handle.read_raw()).split(" ")

        idparts: List[Optional[str]] = ["Galil Motion Control, Inc.",
                                        data[0], None, data[2]]

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))
