"""
A mixin module for USB Human Interface Device instruments
"""
import os
import time
import struct
from typing import Optional, List

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
        instance_id: The id of the instrument we want to connect to. If
            there is only one instrument, then this argument is optional.
            If more than one instrument happen to be connected, use
            `enumerate_devices` method to query their IDs
        timeout: Specify a timeout for this instrument in seconds
    """
    # The following class attributes should be set by subclasses
    vendor_id = 0x0000
    product_id = 0x0000

    @staticmethod
    def _check_hid_import():
        if os.name != 'nt':
            raise ImportError("This driver only works on Windows.")

        if hid is None:
            raise ImportError(
                "pywinusb is not installed. Please install it by typing "
                "'pip install pywinusb' in a qcodes environment terminal"
            )

    def __init__(self, name: str, instance_id: str=None, timeout: float=2,
                 **kwargs):
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

        self._data_buffer: Optional[bytes] = None
        self._device.set_raw_data_handler(self._handler)

        self._timeout = timeout
        self._tries_per_second = 5

        super().__init__(name, **kwargs)

    def _handler(self, data: bytes) -> None:
        self._data_buffer = data

    def _get_data_buffer(self)->Optional[bytes]:
        data = self._data_buffer
        self._data_buffer = None
        return data

    def _pack_string(self, cmd: str) -> bytes:
        raise NotImplementedError("Please subclass")

    def _unpack_string(self, response: bytes) -> str:
        raise NotImplementedError("Please subclass")

    def write_raw(self, cmd: str) -> None:
        """
        Send a string command to the human interface device

        The given command is processed by `_pack_string` method to return a
        byte sequence that is going to be actually sent to the device.
        Subclasses must implement `_pack_string` method.

        Args:
           cmd: a command to send in a form of a string
        """
        data = self._pack_string(cmd)

        result = self._device.send_output_report(data)
        if not result:
            raise RuntimeError(f"Communication with device failed for command "
                               f"{cmd}")

    def ask_raw(self, cmd: str) -> str:
        """
        Send a string command to the human interface device and wait for a reply

        The given command is processed by `_pack_string` method to return a
        byte sequence that is going to be actually sent to the device.
        Subclasses must implement `_pack_string` method.

        The  byte sequence of the reply is processed by `_unpack_string`
        method, and the resulting string is returned. Subclasses must
        implement `_unpack_string` method.

        Args:
            cmd: a command to send in a form of a string
        """
        self.write_raw(cmd)

        number_of_tries = int(self._tries_per_second * self._timeout)

        response = None
        for _ in range(number_of_tries):
            time.sleep(1 / self._tries_per_second)
            response = self._get_data_buffer()
            if response is not None:
                break

        if response is None:
            raise TimeoutError(f"Timed out for command {cmd}")

        return self._unpack_string(response)

    def close(self) -> None:
        self._device.close()

    @classmethod
    def enumerate_devices(cls) -> List[str]:
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


class MiniCircuitsHIDMixin(USBHIDMixin):
    """
    The specific implementation for mini circuit human interface devices.

    This implementation allows to use `write`/`ask` methods of the instrument
    instance to send SCPI commands to MiniCircuits instruments over USB HID
    connection.

    Args:
        name: instrument name
        instance_id: The id of the instrument we want to connect. If there is
            only one instrument then this is an optional argument. If we have
            more then one instrument, use the class method
            `enumerate_devices` to query their IDs
        timeout: Specify a timeout for this instrument in seconds
    """

    def __init__(self, name: str, instance_id: str=None, timeout: float=2,
                 **kwargs):
        # USB interrupt code for sending SCPI commands
        self._sending_scpi_cmds_code = 1
        self._usb_endpoint = 0
        self._end_of_message = b"\x00"
        self.packet_size = 64

        super().__init__(name, instance_id, timeout, **kwargs)

    def _pack_string(self, cmd: str) -> bytes:
        """
        Pack a string to a binary format such that it can be sent to the HID.

        Args:
            cmd: a SCPI command to send
        """
        str_len = len(cmd)

        # "-1" is here because we need to compensate for the first byte in
        # the packet which is always the usb interrupt code of the command
        # (in this case the command tell the device that we are querying a
        # SCPI command)
        pad_len = self.packet_size - str_len - 1

        if pad_len < 0:
            raise ValueError(f"Length of data exceeds {self.packet_size} B")

        packed_data = struct.pack(
            f"BB{str_len}s{pad_len}x",
            self._usb_endpoint,
            self._sending_scpi_cmds_code,
            cmd.encode("ascii")
        )

        return packed_data

    def _unpack_string(self, response: bytes) ->str:
        """
        Unpack data received from the instrument into a string

        Note that this method is not specific to SCPI-only responses.

        Args:
            response: a raw byte sequence response from the instrument
        """
        _, _, reply_data = struct.unpack(
            f"BB{self.packet_size - 1}s",
            bytes(response)
        )
        span = reply_data.find(self._end_of_message)
        return reply_data[:span].decode("ascii")
