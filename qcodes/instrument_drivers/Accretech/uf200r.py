"""
This file holds the QCoDeS driver for the Accretech UF200R Wafer probing
machine, colloquially known as the "autoprober"
"""

import logging
from typing import Any, Optional, Dict

from qcodes.instrument.visa import VisaInstrument


logger = logging.getLogger(__name__)


class UF200R(VisaInstrument):
    """
    The class that is the QCoDeS driver for the Accretech UF200R Wafer probing
    machine
    """

    def __init__(self, name: str, address: str, **kwargs: Any):
        super().__init__(name=name, address=address, **kwargs)
        self.stb: Optional[int] = None
        self.error_type_code_contents: Dict[str, str] = {"S": "System error",
                                                         "E": "Error",
                                                         "O": "Operator call",
                                                         "W": "Warning error",
                                                         "I": "Information"}

        self.add_parameter(
            name="chuck",
            label="Chuck status",
            val_mapping={"Z DOWN": "D",
                         "Z UP": "Z"},
            set_cmd=self._set_chuck
        )
        self.add_parameter(
            name="moveXY",
            label="Move chuck along XY axis",
            set_cmd=self._move_xy_axis
        )
        self.add_parameter(
            name="proberID",
            label="Request prober ID",
            set_cmd=self._get_prober_id
        )
        self.add_parameter(
            name="errorCode",
            label="Request error code",
            set_smd=self._get_error_code
        )

    def _set_chuck(self, target: str) -> None:

        expected_stb: int = {"Z": 67, "D": 68}[target]

        self.write(target)
        self.stb = self.visa_handle.read_stb()

        if expected_stb != self.stb:
            raise RuntimeError(
                f'Could not reach desired state: {target}. '
                f'Expected status byte {expected_stb}, but got {self.stb}.'
                )

    def _move_xy_axis(self, x: int, y: int) -> None:

        cmd = "A"
        abs_y = abs(y)
        if abs_y == y:
            cmd += f"Y+{abs_y:06}"
        else:
            cmd += f"Y-{abs_y:06}"
        abs_x = abs(x)
        if abs_x == x:
            cmd += f"X+{abs_x:06}"
        else:
            cmd += f"X-{abs_x:06}"

        self.write(cmd)
        stb = self.visa_handle.read_stb()

        if stb == 74:
            self.stb = 74
            raise RuntimeError("Movement destination out of probing area.")
        elif stb == self.stb == 67:
            logger.info("Movement complete and chuck back to UP position.")
        elif stb == 65:
            self.stb = 65
            logger.info("End of XY axis movement.")
        else:
            self.stb = stb
            raise RuntimeError(f"Couldn't reach desired state during XY axis "
                               f"movement. Got state byte {stb}.")

    def _get_prober_id(self) -> str:

        self.write("B")
        data = str(self.visa_handle.read_raw(size=11))

        if data[0] != "B":
            raise RuntimeError(f"Didn't receive required data. Instead "
                               f"received {data}.")

        return data[1:-2]

    def _get_error_code(self) -> str:

        assert self.stb is 76
        self.write("E")
        data = str(self.visa_handle.read_raw(size=8))

        if data[0] != "E":
            raise RuntimeError(f"Didn't receive required data. Instead "
                               f"received {data}.")

        return self.error_type_code_contents[data[1]]
