import struct
from typing import Dict

from qcodes import Instrument

from .USBHIDMixin import USBHIDMixin


class RUDAT_13G_90(USBHIDMixin, Instrument):
    """
    Args:
        name (str)
        instance_id (str): The id of the instrument we want to connect. If
            there is only one instrument then this is an optional argument.
            If we have more then one instrument, quarry their ID's by calling
            the class method 'enumerate_devices'
        timeout (float): Specify a timeout for this instrument
    """
    packet_size = 64
    vendor_id = 0x20ce
    product_id = 0x0023

    def __init__(self, name: str, instance_id: str=None, timeout: float=2,
                 *args, **kwargs) ->None:

        super().__init__(
            name=name,
            instance_id=instance_id,
            timeout=timeout,
            *args,
            **kwargs
        )

        self.add_parameter(
            "model_name",
            get_cmd=":MN?"
        )

        self.add_parameter(
            "serial_number",
            get_cmd=":SN?"
        )

        self.add_parameter(
            "firmware",
            get_cmd=":FIRMWARE?"
        )

        self.add_parameter(
            "attenuation",
            set_cmd=":SETATT={}",
            get_cmd=":ATT?",
            get_parser=float
        )

        self.add_parameter(
            "startup_attenuation",
            set_cmd=":STARTUPATT:VALUE:{}",
            get_cmd=":STARTUPATT:VALUE?",
            get_parser=float
        )

        self.add_parameter(
            "hop_points",
            get_cmd="HOP:POINTS?",
            get_parser=int
        )

        self._usb_endpoint = 0
        self._end_of_message = 0x00

        self.connect_message()

    def _send_scpi(self, scpi_str: str) ->str:
        """
        Send a SCPI command to the instrument

        Args:
            scpi_str (string)
        """
        str_len = len(scpi_str)
        command_number = 1
        packed_data = struct.pack(
            f"B{str_len}s", command_number, scpi_str.encode("ascii")
        )

        try:
            response = self.ask_hid(self._usb_endpoint, packed_data)
        except TimeoutError:
            raise TimeoutError(f"Timeout while sending command {scpi_str}")

        reply_string = ""
        # clip the first byte off because it reflects the command
        for char in response[1:]:
            if char == self._end_of_message:
                break

            reply_string += chr(char)

        return reply_string

    def write_raw(self, cmd: str)->None:
        self._send_scpi(cmd)

    def ask_raw(self, cmd: str)->str:
        return self._send_scpi(cmd)

    def get_idn(self) -> Dict[str, str]:
        model = self.model_name()
        serial = self.serial_number()
        firmware = self.firmware()

        return {'vendor': 'Mini-Circuits',
                'model': model,
                'serial': serial,
                'firmware': firmware}
