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

        self.add_parameter(
            name="motor_pos",
            label="Current Motor Position",
            get_cmd=self._get_motor_position,
            docstring="Gets current position of the motor."
        )

        self.connect_message()

    def _get_motor_position(self) -> Dict[str, float]:
        """
        Gets current position of the motor.
        """
        self.write("TP")
        data = str(self.visa_handle.read_raw()).split(", ")
        return {"A": float(data[0]), "B": float(data[1]), "C": float(data[2])}

    def get_idn(self) -> Dict[str, Optional[str]]:
        with self.timeout.set_to(5):
            self.write("ID")
            data = str(self.visa_handle.read_raw()).split(" ")

        idparts: List[Optional[str]] = ["Galil Motion Control, Inc.",
                                        data[0], None, data[2]]

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))
