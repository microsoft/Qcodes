"""
This file holds the QCoDeS driver for the Accretech UF200R Wafer probing
machine, colloquially known as the "autoprober"
"""

import logging
from typing import Any, Optional, Dict, List

from qcodes.instrument.visa import VisaInstrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals


logger = logging.getLogger(__name__)


class UF200R(VisaInstrument):
    """
    The class that is the QCoDeS driver for the Accretech UF200R Wafer probing
    machine
    """

    def __init__(self, name: str, address: str, up_down_time: float,
                 receive_time: float, move_time: float, **kwargs: Any) -> None:
        super().__init__(name=name, address=address, **kwargs)
        self.up_down_time: float = up_down_time
        self.receive_time: float = receive_time
        self.move_time: float = move_time
        self.stb: Optional[int] = None
        self.error_type_code_contents: Dict[str, str] = {"S": "System error",
                                                         "E": "Error",
                                                         "O": "Operator call",
                                                         "W": "Warning error",
                                                         "I": "Information"}
        self._current_chuck_status: Optional[str] = None

        self.add_parameter(
            name="chuck",
            label="Move chuck in Up or Down position",
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
            get_cmd=self._get_prober_id
        )
        self.add_parameter(
            name="errorCode",
            label="Request error code",
            get_cmd=self._get_error_code
        )
        self.add_parameter(
            name="location_number",
            label="Request location no. for tester",
            get_cmd=self._request_location_number()
        )
        self.add_parameter(
            name="chars_in_coordinator_val",
            parameter_class=ManualParameter,
            initial_value="Normal",
            vals=vals.Enum("Normal", "4 characters"),
            docstring="This parameter value should be same as [The number of "
                      "characters of Coordinator for GP-IB] parameter on the "
                      "instrument."
        )
        self.add_parameter(
            name="die_loc",
            label="Request coordinator values of the current probing target",
            get_cmd=self._request_current_probing_target_location
        )

    def get_idn(self) -> Dict[str, Optional[str]]:

        with self.timeout.set_to(self.receive_time):
            self.write("PV")
            data = str(self.visa_handle.read_raw(size=17))

        if data[0:2] != "PV":
            raise RuntimeError(f"Expecting to receive instrument details. "
                               f"Instead received {data}")
        idparts: List[Optional[str]] = [None, data[2:8], None, data[8:15]]

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))

    def move_to_start_die(self) -> None:

        # during probing
        with self.timeout.set_to(self.move_time):
            self.write("G")
            self.stb = int(self.visa_handle.read_stb())

        if self._current_chuck_status == "Up" and self.stb == 67:
            logger.info("Movement to start die complete and chuck back to UP "
                        "position.")
        elif self._current_chuck_status == "Down" and self.stb == 70:
            logger.info("At first die. End of wafer loading.")
        else:
            raise RuntimeError(f"Couldn't reach desired state during move "
                               f"to start die. Got state byte {self.stb}.")

    def move_to_next_die(self) -> None:

        # during probing
        with self.timeout.set_to(self.move_time):
            self.write("J")
            self.stb = int(self.visa_handle.read_stb())

        if self._current_chuck_status == "Up" and self.stb == 67:
            logger.info("Movement to next die complete and chuck back to UP "
                        "position.")
        elif self._current_chuck_status == "Down" and self.stb == 66:
            logger.info("End of movement to next die.")
        elif self.stb == 81:
            logger.info("Next die is not in probing area. Wafer end.")
        else:
            raise RuntimeError(f"Couldn't reach desired state during move "
                               f"to next die. Got state byte {self.stb}.")

    def move_to_next_subdie(self) -> None:

        # during probing
        with self.timeout.set_to(self.move_time):
            self.write("JS")
            self.stb = int(self.visa_handle.read_stb())

        if self._current_chuck_status == "Up" and self.stb == 67:
            logger.info("Movement to next sub die complete and chuck back to "
                        "UP position.")
        elif self._current_chuck_status == "Down" and self.stb == 66:
            logger.info("End of movement to next sub die.")
        elif self.stb == 81:
            logger.info("All sub dies are finished. End of sub die.")
        elif self.stb == 76:
            raise RuntimeError(f"Not a normal end for move to next sub die. An "
                               f"error has occurred. Got state byte "
                               f"{self.stb}.")
        else:
            raise RuntimeError(f"Couldn't reach desired state during move "
                               f"to next sub die. Got state byte {self.stb}.")

    def move_to_specified_die(self, x: int, y: int) -> None:

        # during probing
        cmd = self._create_cmd("J", x, y)
        with self.timeout.set_to(self.move_time):
            self.write(cmd)
            self.stb = self.visa_handle.read_stb()

        if self._current_chuck_status == "Up" and self.stb == 67:
            logger.info("Movement complete and chuck back to UP position.")
        elif self._current_chuck_status == "Down" and self.stb == 66:
            logger.info(f"End of movement to specified die location x: {x}, "
                        f"y: {y}.")
        elif self.stb == 74:
            raise RuntimeError("Movement destination out of probing area.")
        else:
            raise RuntimeError(f"Couldn't reach desired state during move to "
                               f"specified die. Got state byte {self.stb}.")

    def load_wafer_with_alignment(self) -> None:

        # Wafer sensing has already been performed
        with self.timeout.set_to(self.up_down_time + self.move_time):
            self.write("L")
            self.stb = int(self.visa_handle.read_stb())

        if self.stb == 94:
            logger.info("No wafer in load source cassette. End of lot process.")
        elif self.stb == 70:
            self._current_chuck_status = "Down"
            logger.info("At first die. End of wafer loading.")
        else:
            raise RuntimeError(f"Couldn't reach desired state during load "
                               f"wafer with alignment. Got state byte"
                               f" {self.stb}.")

    def _set_chuck(self, target: str) -> None:

        expected_stb: int = {"Z": 67, "D": 68}[target]

        with self.timeout.set_to(self.up_down_time):
            self.write(target)
            self.stb = self.visa_handle.read_stb()

        if expected_stb == self.stb and target == "Z":
            self._current_chuck_status = "Up"
        elif expected_stb == self.stb and target == "D":
            self._current_chuck_status = "Down"
        else:
            raise RuntimeError(
                f'Could not reach desired state: {target}. '
                f'Expected status byte {expected_stb}, but got {self.stb}.'
                )

    def _move_xy_axis(self, x: int, y: int) -> None:

        abs_x = abs(x)
        abs_y = abs(y)
        if len(str(abs_x)) > 6 or len(str(abs_y)) > 6:
            raise RuntimeError("Co-ordinate values cannot have more than 6 "
                               "digits.")

        cmd = "A"
        if abs_y == y:
            cmd += f"Y+{abs_y:06}"
        else:
            cmd += f"Y-{abs_y:06}"

        if abs_x == x:
            cmd += f"X+{abs_x:06}"
        else:
            cmd += f"X-{abs_x:06}"

        with self.timeout.set_to(self.move_time):
            self.write(cmd)
            self.stb = self.visa_handle.read_stb()

        if self._current_chuck_status == "Up" and self.stb == 67:
            logger.info("Movement complete and chuck back to UP position.")
        elif self._current_chuck_status == "Down" and self.stb == 65:
            logger.info("End of XY axis movement.")
        elif self.stb == 74:
            raise RuntimeError("Movement destination out of probing area.")
        else:
            raise RuntimeError(f"Couldn't reach desired state during XY axis "
                               f"movement. Got state byte {self.stb}.")

    def _get_prober_id(self) -> str:

        with self.timeout.set_to(self.receive_time):
            self.write("B")
            data = str(self.visa_handle.read_raw(size=11))

        if data[0] != "B":
            raise RuntimeError(f"Didn't receive prober ID. Instead received "
                               f"{data}.")

        return data[1:-2]

    def _get_error_code(self) -> str:

        assert self.stb == 76

        with self.timeout.set_to(self.receive_time):
            self.write("E")
            data = str(self.visa_handle.read_raw(size=8))

        if data[0] != "E":
            raise RuntimeError(f"Didn't receive error information. Instead "
                               f"received {data}.")

        return self.error_type_code_contents[data[1]]

    def _request_location_number(self) -> int:

        # after initialization
        with self.timeout.set_to(self.receive_time):
            self.write("kh")
            data = str(self.visa_handle.read_raw(size=6))

        if data[0:2] != "kh":
            raise RuntimeError(f"Didn't receive location number. Instead "
                               f"received {data}.")

        return int(data[2:4])

    def _request_current_probing_target_location(self) -> Dict[str, int]:

        # during probing
        with self.timeout.set_to(self.receive_time):
            self.write("Q")
            if self.chars_in_coordinator_val() == "Normal":
                data = str(self.visa_handle.read_raw(size=11))
            else:
                data = str(self.visa_handle.read_raw(size=13))

        if data[0] != "Q":
            raise RuntimeError(f"Didn't receive coordinator value. Instead "
                               f"received {data}.")

        return self._parse_coordinator_value(data)

    def _parse_coordinator_value(self, data: str) -> Dict[str, int]:

        if self.chars_in_coordinator_val() == "Normal":
            x_val = int(data[6:9]) if data[6] != '-' else -1*int(data[7:9])
            y_val = int(data[2:5]) if data[2] != '-' else -1*int(data[3:5])
        else:
            x_val = int(data[7:11]) if data[7] != '-' else -1*int(data[8:11])
            y_val = int(data[2:6]) if data[2] != '-' else -1*int(data[3:6])

        return {'x': x_val, 'y': y_val}

    def _create_cmd(self, cmd: str, x: int, y: int) -> str:

        if self.chars_in_coordinator_val() == "Normal":
            val_len = 3
            if len(str(x)) > 3 or len(str(y)) > 3:
                raise RuntimeError(f"Only 3 chars allowed for x or y "
                                   f"e.g. 999 or -56. Got values x: {x}, "
                                   f"y: {y}")
        else:
            val_len = 4
            if len(str(x)) > 4 or len(str(y)) > 4:
                raise RuntimeError(f"Only 4 chars allowed for x or y "
                                   f"e.g. -745 or 8359. Got values x: {x}, "
                                   f"y: {y}")

        return self._concat_str(cmd, x, y, val_len)

    def _concat_str(self, cmd: str, x: int, y: int, val_len: int) -> str:

        abs_x = abs(x)
        abs_y = abs(y)

        if abs_y == y:
            cmd += f"Y{abs_y:0{val_len}}"
        else:
            cmd += f"Y-{abs_y:0{val_len-1}}"

        if abs_x == x:
            cmd += f"X{abs_x:0{val_len}}"
        else:
            cmd += f"X-{abs_x:0{val_len-1}}"

        return cmd
