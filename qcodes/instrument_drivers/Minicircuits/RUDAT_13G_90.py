import pywinusb.hid as hid
import time
import struct

from qcodes import Instrument


class RUDAT_13G_90(Instrument):
    vendor_id = 0x20ce
    product_id = 0x0023

    def __init__(self, name, timeout=2):
        super().__init__(name)
        self._timeout = timeout

        devs = hid.HidDeviceFilter(
            product_id=RUDAT_13G_90.product_id,
            vendor_id=RUDAT_13G_90.vendor_id
        ).get_devices()

        if len(devs) == 0:
            raise RuntimeError("No instruments found!")

        self._data_buffer = None
        self._device = devs[0]  # We assume we only have one instrument
        self._device.open()
        self._device.set_raw_data_handler(self._handler)

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

        print("connected to ", self.get_idn())

    def _handler(self, data):
        self._data_buffer = data

    def _get_data_buffer(self):
        data = self._data_buffer
        self._data_buffer = None
        return data

    def _get_device_reply(self):
        tries_per_second = 5
        number_of_tries = tries_per_second * self._timeout

        response = None
        for _ in range(number_of_tries):
            time.sleep(1 / tries_per_second)
            response = self._get_data_buffer()
            if response is not None:
                break

        if response is None:
            raise TimeoutError("")

        reply_string = ""
        for char in response[2:]:
            if char == 0x00:
                break

            reply_string += chr(char)

        return reply_string

    def get_model(self):
        data = struct.pack("BB65x", 0, 40)
        self._device.send_output_report(data)
        return self._get_device_reply()

    def get_serial(self):
        data = struct.pack("BB65x", 0, 41)
        self._device.send_output_report(data)
        return self._get_device_reply()

    def get_idn(self):
        model_name = self.get_model()
        serial_number = self.get_serial()
        return f"{model_name} SN: {serial_number}"

    def _send_scpi(self, scpi_str):
        n = len(scpi_str)
        m = 65 - n
        data = struct.pack(f"BB{n}s{m}x", 0, 1, scpi_str.encode("ascii"))
        self._device.send_output_report(data)

        try:
            return self._get_device_reply()
        except TimeoutError:
            raise TimeoutError(f"Timeout while sending command {scpi_str}")

    def write_raw(self, cmd):
        self._send_scpi(cmd)

    def ask_raw(self, cmd):
        return self._send_scpi(cmd)
