import os
from typing import TYPE_CHECKING

import usb1
from libusb1 import libusb_error
from usb1 import USBContext, USBDevice, USBDeviceHandle, USBError

# QCoDeS imports
from qcodes.instrument_drivers.Minicircuits.Base_SPDT import (
    MiniCircuitsSPDTBase,
    MiniCircuitsSPDTSwitchChannelBase,
)

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.instrument import InstrumentBaseKWArgs

MINICIRCUITS_VENDOR_ID = 0x20CE
RF_SWITCH_PRODUCT_ID = 0x0022


def open_switch_with_sn(serial_number: str | None) -> USBDeviceHandle | None:
    usb_context = USBContext()
    if serial_number is not None:
        device_iterator = usb_context.getDeviceIterator(
            skip_on_error=True,
        )
        try:
            for device in device_iterator:
                if (
                    device.getVendorID() == MINICIRCUITS_VENDOR_ID
                    and device.getProductID() == RF_SWITCH_PRODUCT_ID
                ):
                    handle = device.open()
                    if get_serial_number(handle) == serial_number:
                        return handle
                    device.close()  # Unsure what missing arguments are needed or what they do
        finally:
            device_iterator.close()
    else:  # If no SN is provided, we can use the built-in function to return the first Minicircuits switch
        return usb_context.openByVendorIDAndProductID(
            MINICIRCUITS_VENDOR_ID, RF_SWITCH_PRODUCT_ID
        )
    return None


def get_serial_number(handle: USBDeviceHandle) -> str:
    handle.resetDevice()
    handle.claimInterface(0)
    cmd = [
        41,
    ]
    cmd_array = bytearray([0] * 64)
    cmd_array[0 : len(cmd)] = cmd
    handle.interruptWrite(endpoint=1, data=cmd_array, timeout=50)
    response = handle.interruptRead(endpoint=1, length=64, timeout=1000)
    resp_length = response.index(bytearray([0]))
    trimmed_response = response[1:resp_length]
    return trimmed_response.decode("ascii")


class MiniCircuitsUsbSPDTSwitchChannel(
    MiniCircuitsSPDTSwitchChannelBase["MiniCircuitsUsbSPDT"]
):
    def _set_switch(self, switch: int) -> None:
        self.parent._query_scpi(f"set{self.channel_letter}={switch - 1}")

    def _get_switch(self) -> int:
        all_ports_state = int(self.parent._query_scpi("SWPORT?"))
        bitmask = 2**self.channel_number
        return int((all_ports_state & bitmask) >= 1) + 1


class MiniCircuitsUsbSPDT(MiniCircuitsSPDTBase):
    CHANNEL_CLASS = MiniCircuitsUsbSPDTSwitchChannel

    def __init__(
        self,
        name: str,
        serial_number: str | None = None,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ):
        """
        Mini-Circuits SPDT RF switch

        Args:
            name: the name of the instrument
            serial_number: the serial number of the device
                (printed on the sticker on the back side, without s/n)
            kwargs: kwargs to be passed to Instrument class.

        """
        super().__init__(name, **kwargs)
        self._handle: USBDeviceHandle | None = open_switch_with_sn(serial_number)

        self.add_channels()
        self.connect_message()

    @property
    def handle(self) -> USBDeviceHandle:
        if self._handle is None:
            raise Exception  # TODO Make better
        return self._handle

    def _query_scpi(self, command: str) -> str:
        cmd_bytes = bytearray([42, 58])  # Interrupt code for Send SCPI Command
        cmd_bytes.extend(bytearray(command, "ascii"))
        cmd_bytes.extend(bytearray(64 - len(cmd_bytes)))

        self.handle.interruptWrite(endpoint=1, data=cmd_bytes, timeout=50)

        response = self.handle.interruptRead(endpoint=1, length=64, timeout=1000)
        resp_length = response.index(bytearray([0]))
        trimmed_response = response[1:resp_length]

        return trimmed_response.decode("ascii")

    def get_idn(self) -> dict[str, str | None]:
        # the arguments in those functions is the serial number or none if
        # there is only one switch.
        fw = self._query_scpi("FIRMWARE?")
        MN = self._query_scpi("MN?")
        SN = self._query_scpi("SN?")

        id_dict = {"firmware": fw, "model": MN, "serial": SN, "vendor": "Mini-Circuits"}
        return id_dict


# Notes:
# https://www.minicircuits.com/softwaredownload/Prog_Manual-2-Switch.pdf
# Section 3 of the Minicircuits Programming Manual for RF switches includes additional
# SCPI commands which we may find useful. These are not currently implemented
# For example: `SETP=[states]` allows multiple switch states to be set with a single command
#
# We may also eventually be able to unify the Ethernet interface used by the RC-SPDT and RC-SP4T drivers
# with the USB interface and have both rely on the SCPI commands
#
# Finally, the commands for SP4T and SPDT are not so different that we couldn't have a generic driver
# That works for all versions
