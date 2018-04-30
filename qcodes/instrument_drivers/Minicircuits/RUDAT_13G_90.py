import struct

from qcodes import Instrument

from .USBHIDMixin import USBHIDMixin


class RUDAT_13G_90(USBHIDMixin, Instrument):
    packet_size = 64
    vendor_id = 0x20ce
    product_id = 0x0023

    def __init__(self, name, instance_id=None, timeout=2, *args, **kwargs):

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

    def _send_scpi(self, scpi_str):
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

    def write_raw(self, cmd):
        self._send_scpi(cmd)

    def ask_raw(self, cmd):
        return self._send_scpi(cmd)
