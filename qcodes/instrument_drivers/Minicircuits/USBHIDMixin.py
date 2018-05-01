"""
A mixin module for USB Human Interface Device instruments
"""
import os
import time

try:
    import pywinusb.hid as hid
except ImportError:
    # We will raise a proper error when we attempt to instantiate a driver.
    # Raising an exception here will cause CI to fail under Linux
    hid = None

from qcodes.instrument.base import Instrument


class USBHIDMixin(Instrument):
    """
    Args:
        instance_id (str): The id of the instrument we want to connect. If
            there is only one instrument then this is an optional argument.
            If we have more then one instrument, quarry their ID's by calling
            the class method 'enumerate_devices'
        timeout (float): Specify a timeout for this instrument
    """
    # The following class attributes are set by subclasses
    vendor_id = 0x0000
    product_id = 0x0000

    @staticmethod
    def _check_hid_import():

        if os.name != 'nt':
            raise ImportError("""This driver only works in Windows.""")

        if hid is None:

            raise ImportError(
                "pywinusb is not installed. Please install it by typing "
                "'pip install pywinusb' in a qcodes environment terminal"
            )

    def __init__(self, name, instance_id: str=None, timeout: float=2,
                 **kwargs) ->None:

        super().__init__(name, **kwargs)
        self._check_hid_import()

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
        self._data_buffer: bytes = None
        self._timeout = timeout
        self._device.set_raw_data_handler(self._handler)

    def _handler(self, data: bytes) ->None:
        self._data_buffer = data

    def _get_data_buffer(self)->bytes:
        data = self._data_buffer
        self._data_buffer = None
        return data

    def _pack_string(self, cmd: str) ->bytes:
        raise NotImplementedError("Please subclass")

    def _unpack_string(self, response: bytes) ->str:
        raise NotImplementedError("Please subclass")

    def write_raw(self, cmd: str) ->None:
        """
        Send binary data to the human interface device

        Args:
           cmd (str)
        """
        data = self._pack_string(cmd)

        result = self._device.send_output_report(data)
        if not result:
            raise RuntimeError("Communication with device failed")

    def ask_raw(self, cmd: str) ->str:
        """
        Send binary data to the human interface device and wait for a reply

        Args:
            cmd (str)
        """
        self.write_raw(cmd)

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

        return self._unpack_string(response)

    def close(self):
        self._device.close()

    @classmethod
    def enumerate_devices(cls):
        """
        This method returns the 'instance_id's of all connected devices for
        with the given product and vendor IDs.
        """
        cls._check_hid_import()

        devs = hid.HidDeviceFilter(
            porduct_id=cls.product_id,
            vendor_id=cls.vendor_id
        ).get_devices()

        return [dev.instance_id for dev in devs]
