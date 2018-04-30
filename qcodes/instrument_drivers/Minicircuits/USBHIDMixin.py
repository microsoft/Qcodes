"""
A mixin module for USB Human Interface Device instruments
"""
import os
import time
import struct


class USBHIDMixin:
    """
    Args:
        instance_id (str): The id of the instrument we want to connect. If
            there is only one instrument then this is an optional argument.
            If we have more then one instrument, quarry their ID's by calling
            the class method 'enumerate_devices'
        timeout (float): Specify a timeout for this instrument
    """
    # The following class attributes are set by subclasses
    packet_size = 0
    vendor_id = 0x0000
    product_id = 0x0000

    def __init__(self, instance_id: str=None, timeout: float=2, *args,
                 **kwargs) ->None:
        super().__init__(*args, **kwargs)

        if os.name != 'nt':
            raise ImportError("""This driver only works in Windows.""")

        try:
            import pywinusb.hid as hid
        except ImportError:
            raise ImportError("pywinusb is not installed. Please install "
                              "it by typing 'pip install pywinusb' in a"
                              "qcodes environment terminal")

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

        self._device = devs[0]
        self._device.open()
        self._data_buffer = None
        self._timeout = timeout
        self._device.set_raw_data_handler(self._handler)

    def _handler(self, data: bytes) ->None:
        self._data_buffer = data

    def _get_data_buffer(self)->bytearray:
        data = self._data_buffer
        self._data_buffer = None
        return data

    def send_hid(self, feature_id: int, data: bytes):
        """
        Send binary data to the human interface device

        Args:
            feature_id (int): The 'address' of the device we want to send the
                meassage to
            data (bytearray)
        """
        data_len = len(data)
        pad_len = self.packet_size - data_len

        if pad_len < 0:
            raise ValueError(f"Length of data exceeds {self.packet_size} B")

        packed_data = struct.pack(f"B{data_len}s{pad_len}x", feature_id, data)
        result = self._device.send_output_report(packed_data)
        if not result:
            raise RuntimeError("Communication with device failed")

    def ask_hid(self, feature_id: int, data: bytes) ->bytes:
        """
        Send binary data to the human interface device and wait for a reply

        Args:
            feature_id (int): The 'address' of the device we want to send the
                meassage to
            data (bytearray)
        """
        self.send_hid(feature_id, data)

        tries_per_second = 5
        number_of_tries = int(tries_per_second * self._timeout)

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
        This method returns the 'instance_id's of all connected devices for
        with the given product and vendor IDs.
        """
        devs = hid.HidDeviceFilter(
            porduct_id=cls.product_id,
            vendor_id=cls.vendor_id
        ).get_devices()

        return [dev.instance_id for dev in devs]
