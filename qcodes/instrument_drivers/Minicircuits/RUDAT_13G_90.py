import struct
from typing import Dict

from .USBHIDInstrument import USBHIDInstrument


class RUDAT_13G_90(USBHIDInstrument):
    """
    Args:
        name (str)
        instance_id (str): The id of the instrument we want to connect. If
            there is only one instrument then this is an optional argument.
            If we have more then one instrument, quarry their ID's by calling
            the class method 'enumerate_devices'
        timeout (float): Specify a timeout for this instrument
    """
    vendor_id = 0x20ce
    product_id = 0x0023

    def __init__(self, name: str, instance_id: str=None, timeout: float=2,
                 *args, **kwargs) ->None:

        super().__init__(name, instance_id, timeout, *args, **kwargs)

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
        self._end_of_message = b"\x00"
        self.packet_size = 64

        self.connect_message()

    def _pack_string(self, scpi_str: str) ->bytes:
        """
        Pack a string to a binary format such that it can be send to the
        HID.

        Args:
            scpi_str (str)
        """
        str_len = len(scpi_str)
        pad_len = self.packet_size - str_len

        if pad_len < 0:
            raise ValueError(f"Length of data exceeds {self.packet_size} B")

        command_number = 1
        packed_data = struct.pack(
            f"BB{str_len}s{pad_len}x",
            self._usb_endpoint, command_number, scpi_str.encode("ascii")
        )

        return packed_data

    def _unpack_string(self, response: bytes) ->str:
        """
        Unpack data from the instrument to a string

        Args:
            response (bytes)
        """
        usb_end_point, command_number, reply_data = struct.unpack(
            f"BB{self.packet_size-1}s", bytes(response)
        )
        span = reply_data.find(self._end_of_message)
        return str(reply_data[:span])

    def get_idn(self) -> Dict[str, str]:
        model = self.model_name()
        serial = self.serial_number()
        firmware = self.firmware()

        return {'vendor': 'Mini-Circuits',
                'model': model,
                'serial': serial,
                'firmware': firmware}
