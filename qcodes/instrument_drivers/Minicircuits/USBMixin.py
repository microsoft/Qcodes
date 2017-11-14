import pywinusb.hid as hid
from functools import partial
from threading import Event
from typing import Optional

import logging
log = logging.getLogger(__name__)


class USBException(Exception):
    """Base class for exceptions in this module."""
    pass


class USBMixin:
    """
    Mixin class to use simple raw commands.

    This mixin uses the pywinusb package to enable communication with a HID-USB
    device.

    It should be used as the first superclass of an instrument, e.g.
    ```
    class MyInstrument(USBMixin, Instrument):
       ...
       vendor_id = 0x1234
       product_id = 0x5678
       package_size = 64

    ```
    so that the constructor and destructor can take care of establishing and
    releasing the connection to the device

    """

    # max size of usb package in bytes
    package_size = 0  # type: int

    # These class variables have to be overridden by the subclass
    vendor_id = 0x0000  # type: int
    product_id = 0x0000  # type: int

    def __init__(
            self,
            instance_id: Optional[str] = None,
            *args,
            **kwargs) -> None:
        # as this is a mixin class it is important to call the constructor of
        # the next in line class with the super() function.
        # So here super() refers to the second class from which the subclass
        # is inherited.
        super().__init__(*args, **kwargs)
        self.dev = None
        self._open_USB_device(porduct_id=self.product_id,
                              vendor_id=self.vendor_id,
                              instance_id=instance_id)

    def __del__(self) -> None:
        if self.dev is not None:
            self._close_USB_device
        super().__del__()

    def _open_USB_device(self, **kwargs) -> None:
        """
        establishes the connection to an USB device.

        """
        allDevices = hid.HidDeviceFilter(**kwargs).get_devices()

        if len(allDevices) == 0:
            raise USBException("Device not found.")
        elif len(allDevices) > 1:
            raise USBException("Device not unique. Supply extra keywords")

        if self.dev:
            raise USBException("Device already open")

        self.dev = allDevices[0]
        # may raise HIDError if resource already opened
        self.dev.open()

    def _close_USB_device(self):
        if not self.dev:
            raise USBException("trying to close device that is not open.")
        self.dev.close()
        self.dev = None

    def write_USB_data(self, feature_id: int, data: bytes) -> None:
        """
        writes raw data to the usb device.

        Args:
           feature_id: specifies the data endpoint of the device. A HID can
              offer different 'feature_id's for different purposes, similar to
              different ports in tcp/ip
           data: the raw data to be transferred. Has to be compatible with
              'package_size'
        """
        if feature_id > 255 or feature_id < 0:
            raise USBException("a feature ID has to be a single byte")

        if len(data) > self.package_size:
            raise USBException("trying to send a package bigger than {} bytes"
                               .format(self.package_size))
        b = bytearray(self.package_size + 1)
        b[0] = feature_id
        for i, v in enumerate(data):
            b[i + 1] = v

        # log.debug("sending {}\n".format(b))
        feat = self.dev.send_output_report(b)
        if not feat:
            raise USBException

    @staticmethod
    def _usb_handler(event: Event, receiver: bytearray, data: bytes):
        # log.debug("received {}\n".format(data))
        receiver[:] = data
        # break the 'wait()'-call in the mainloop by notifying that we have
        # received
        # data
        event.set()

    def ask_USB_data(
            self,
            feature_id: int,
            data: bytes,
            timeout: float=1) -> bytearray:
        """
        Queries data from the device, by registering a listener for a feature
        and subsequently sending data.

        Args:
           feature_id: the common feature ID for both sending the data and
              retrieving data. Using different IDs is not yet implemented.
           data: The raw data to be sent. Has to be compatible with
              'package_size'
           timeout: timeout in seconds for the query.
        """
        # We need an 'Event'-object from the threading library, because
        # 'pywinusb' is implented with a multi threaded response queue. One has
        # to provide a handler for the response. Here this behaviour is forced
        # to be linear again by envoking 'Event.wait(timeout)' right after the
        # query which waits until the timeout is reached or 'Event.set()' is
        # called from the handler
        event = Event()
        # A local buffer to receive the response data
        receiver = bytearray(USBMixin.package_size)
        self.dev.set_raw_data_handler(
            partial(USBMixin._usb_handler, event, receiver))

        # make the query
        self.write_USB_data(feature_id, data)
        # wait until response from device or timeout
        timedout = not event.wait(timeout)
        # this is how the handler is supposed to be deactivated in this lib
        self.dev.set_raw_data_handler(None)
        if timedout:
            raise USBException("request timed out {}".format(receiver))
        # remove first byte as it indicates the feature id, which is already
        # known
        return receiver[1:]

    @classmethod
    def enumerate_devices(cls):
        """
        This method returns the 'instance_id's of all connected decives for with
        the given product and vendor IDs.
        """
        allDevices = hid.HidDeviceFilter(
            porduct_id=cls.product_id,
            vendor_id=cls.vendor_id,).get_devices()
        instance_ids = [dev.instance_id for dev in allDevices]
        return instance_ids
