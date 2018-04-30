"""
A mixin module for USB Human Interface Device instruments
"""
import pywinusb.hid as hid
import time
import struct


class USBHIDMixin:
    # The following class attributes are set by subclasses
    packet_size = 0
    vendor_id = 0x0000
    product_id = 0x0000

    def __init__(self, instance_id=None, timeout=2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        devs = hid.HidDeviceFilter(
            product_id=self.product_id,
            vendor_id=self.vendor_id,
            instance_id=instance_id
        ).get_devices()

        if len(devs) == 0:
            raise RuntimeError("No instruments found!")
        elif len(devs) > 1:
            raise RuntimeError("Multiple HID devices detected! Please supply "
                               "a instance id")

        self._device = devs[0]  # We assume we only have one instrument
        self._device.open()
        self._data_buffer = None
        self._timeout = timeout
        self._device.set_raw_data_handler(self._handler)

    def _handler(self, data):
        self._data_buffer = data

    def _get_data_buffer(self):
        data = self._data_buffer
        self._data_buffer = None
        return data

    def send_hid(self, feature_id, data):
        data_len = len(data)
        pad_len = self.packet_size - data_len

        if pad_len < 0:
            raise ValueError(f"Length of data exceeds {self.packet_size} B")

        packed_data = struct.pack(f"B{data_len}s{pad_len}x", feature_id, data)
        result = self._device.send_output_report(packed_data)
        if not result:
            raise RuntimeError("Communication with device failed")

    def ask_hid(self, feature_id, data):
        self.send_hid(feature_id, data)

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
        # The first bit is the feature id
        return response[1:]

    @classmethod
    def enumerate_devices(cls):
        """
        This method returns the 'instance_id's of all connected decives for with
        the given product and vendor IDs.
        """
        devs = hid.HidDeviceFilter(
            porduct_id=cls.product_id,
            vendor_id=cls.vendor_id
        ).get_devices()

        return [dev.instance_id for dev in devs]
