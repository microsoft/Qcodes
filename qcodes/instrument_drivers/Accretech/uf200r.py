"""
This file holds the QCoDeS driver for the Accretech UF200R Wafer probing
machine, colloquially known as the "autoprober"
"""

from typing import Any

from qcodes.instrument.visa import VisaInstrument


class UF200R(VisaInstrument):
    """
    The class that is the QCoDeS driver for the Accretech UF200R Wafer probing
    machine
    """

    def __init__(self, name: str, address: str, **kwargs: Any):
        super().__init__(name=name, address=address, **kwargs)

        self.add_parameter(
            name="chuck",
            label="Chuck status",
            val_mapping={"Z DOWN": "D",
                         "Z UP": "Z"},
            set_cmd=self._set_chuck
        )

    def _set_chuck(self, target: str) -> None:

        expected_stb: int = {"Z": 67, "D": 68}[target]

        self.write(target)
        stb: int = self.visa_handle.read_stb()

        if expected_stb != stb:
            raise RuntimeError(
                f'Could not reach desired state: {target}. '
                f'Expected status byte {expected_stb}, but got {stb}.'
                )
