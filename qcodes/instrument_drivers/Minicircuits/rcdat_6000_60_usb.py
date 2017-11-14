import logging
from qcodes.instrument_drivers.Minicircuits.USBMixin import USBMixin


from qcodes.instrument.base import Instrument
from typing import Dict, Optional

log = logging.getLogger(__name__)


class RCDAT_6000_60_usb(USBMixin, Instrument):
    """
    Driver for the Minicircuits RCDAT-6000-60 programmable attenuator.

    This driver is implements the attenuator for a connection through raw USB.
    No drivers are needed but only the basic functionality of setting the
    attenuation is provided. For using the sweep and fast table hopping mode
    another driver has to be implemented.
    For implementing this other functionality there have been the following
    difficulties:
     * using the device in VISA USB-raw mode:
       To do this one can use the NI-VISA Driver Wizard with the PID and VID
       as provided in this file, to generate an .inf file. The trouble for
       installing the .inf file is that it is not signed. To install it anyways
       driver signatures have to be temporily disabled and the computer rebooted
       To avoid this, this reduced driver uses the windows inbuildt USB
       functions
     * An alternative approach is to use the dlls provided by Minicircuits.
       The functions prodided can be called from python using the ctypes library
       This requires to install the dll files on every computer to use the
       device
     * Yet another option is to extend this driver by reverse engineering the
       raw and binary USB commands for this functionality by analyzing the usb
       packages send by the supplied Minicircuits GUI.

    For the ethernet version use the driver RCDAT_6000_60_telnet.

    Depednecies:
     - pywinusb
    """

    # To identify the USB device. If there are multiple connected the serial
    # number has to be supplied as well
    vendor_id = 0x20ce  # type: int
    product_id = 0x0023  # type: int

    # the feature_id identifies the usb enpoint to and from which the data is
    # sent. There is only one endpoint for this simple device
    feature_id = 0  # type: int

    # a list of commands taken from the manual for programmable attenuators
    # p.104. Every command consists of 64 bytes. The first byte given here
    # specifies the type of command followed by 63 bytes payload.
    commands: Dict[str, bytes] = {'read_attenuation': bytes([18]),
                                  'set_attenuation': bytes([19]),
                                  'send_visa': bytes([1]),
                                  'get_device_model_name': bytes([40]),
                                  'get_device_serial_number': bytes([41]),
                                  'get_firmware': bytes([99]), }

    # size of usb package in bytes
    package_size = 64

    def __init__(
            self,
            name: str,
            instance_id: Optional[str] = None,
            **kwargs) -> None:
        """
        Instantiates the instrument.

        Args:
            name: The instrument name used by qcodes
            instanc_id: The identification string for the instrument. If there
               are multiple instruments of the same type connected, an
               additional 'instance_id' has to be provided to identify the
               desired instance. Use the enumerate_devices() function to get a
               list of all 'instance_id's of the connected devices.
        """

        # super refers to mixin, mixin calls super().__init__() to
        # refer to Instrument
        super().__init__(instance_id=instance_id, name=name, **kwargs)

        self.connect_message()

        self.add_parameter(name='attenuation',
                           label='attenuation',
                           unit='dB',
                           get_cmd=self.get_attenuation,
                           set_cmd=self.set_attenuation,)
        # TODO: add validator

    @staticmethod
    def _prepare_package(cmd: str, data: bytes = None) -> bytearray:
        bCmd = RCDAT_6000_60_usb.commands[cmd]
        if data:
            package = bCmd + data
        else:
            package = bCmd
        return package

    def write_cmd(self, cmd: str, data: bytes = None) -> None:
        self.write_USB_data(self.feature_id, self._prepare_package(cmd, data))

    def ask_cmd(self, cmd: str, data: bytes = None) -> bytearray:
        # clip the first byte off because it reflects the command
        ret = self.ask_USB_data(
            self.feature_id,
            self._prepare_package(
                cmd,
                data))
        return ret[1:]

    def get_attenuation(self) -> float:
        ret = self.ask_cmd('get_attenuation')
        return ret[2] + float(ret[3]) / 4

    def set_attenuation(self, attenuation: float) -> None:
        data = bytearray(2)
        data[0] = int(attenuation)
        data[1] = int((attenuation - data[0]) * 4)
        self.write_cmd('set_attenuation', data)

    def get_idn(self) -> Dict[str, str]:
        """
        overides get_idn from 'Instrument'
        """
        model = self._bytearray_to_string(
            self.ask_cmd('get_device_model_name'))
        serial = self._bytearray_to_string(
            self.ask_cmd('get_device_serial_number'))
        firmware = self._bytearray_to_string(self.ask_cmd('get_firmware'))
        return {'vendor': 'Mini-Circuits',
                'model': model,
                'serial': serial,
                'firmware': firmware}

    # NOTE: the following command silently failed when tested with firmware C9-2
    # according to the manual, it should work exactly as the other commands.
    def ask_visa(self, cmd: str) -> str:
        log.error(
            'Trying to call unimplemented function: ask_visa for RCDAT-6000-60')
        data = cmd.encode('ascii')
        return self.ask_cmd('send_visa', data)

    @staticmethod
    def _bytearray_to_string(b: bytes) -> bytearray:
        i = 0
        while i < len(b) and b[i] != 0:
            i += 1
        ret = b[0:i]
        return ret.decode('ascii')

    @staticmethod
    def round_to_nearest_attenuation_value(val: float) -> float:
        return float(round(val * 4) / 4)

    @staticmethod
    def round_to_nearest_attenuation_value(val: float) -> float:
        return float(round(val * 4) / 4)
