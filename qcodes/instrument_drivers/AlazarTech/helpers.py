"""
This module provides helper objects for Alazar driver class. The objects allow
to hide realted pieces of logic into a form of "submodule" (analogious to
:meth:`.InstrumentBase.add_submodule`) that can be included into the driver
class in some way.
"""


from .ats_api import AlazarATSAPI
from .constants import Capability


class CapabilityHelper():
    """
    A helper class providing convenient methods for various useful
    'query capability' (``AlazarQueryCapability``) calls for a given
    Alazar board.

    Most common capabilities are enumerated in :attr:`.CAPABILITIES`.

    For frequently used capabilities, dedicated convenience ``query_<...>()``
    methods are available.

    Args:
        api: Instance of Alazar ATS API class
        handle: Handle of a specific board (from ``AlazarGetBoardBySystemId``)
    """

    CAPABILITIES = Capability

    def __init__(self, api: AlazarATSAPI, handle: int):
        self._api = api
        self._handle = handle

    def query(self, capability: int) -> int:
        """Query the given 'capability' of the board"""
        return self._api.query_capability_(self._handle, capability)

    # Convenience and specific methods

    def query_serial(self) -> str:
        return str(self.query(self.CAPABILITIES.GET_SERIAL_NUMBER))

    def query_latest_calibration(self) -> str:
        """Query latest calibration date in '12-34-56' format"""
        # ``date_int`` is a decimal number with the format DDMMYY where
        # DD is 1-31, MM is 1-12, and YY is 00-99 from 2000.
        date_int = self.query(self.CAPABILITIES.GET_LATEST_CAL_DATE)
        date_str = str(date_int)
        date = date_str[0:2] + "-" + date_str[2:4] + "-" + date_str[4:6]
        return date

    def query_memory_size(self) -> int:
        """Query board memory size in samples"""
        return self.query(self.CAPABILITIES.MEMORY_SIZE)

    def query_asopc_type(self) -> int:
        return self.query(self.CAPABILITIES.ASOPC_TYPE)

    def query_pcie_link_speed(self) -> float:
        """Query PCIE link speed in GB/s"""
        # See the ATS-SDK programmer's guide about the encoding
        # of the PCIE link speed.
        link_speed_int = self.query(self.CAPABILITIES.GET_PCIE_LINK_SPEED)
        link_speed = link_speed_int * 2.5 / 10
        return link_speed

    def query_pcie_link_width(self) -> int:
        """Query PCIE link width"""
        return self.query(self.CAPABILITIES.GET_PCIE_LINK_WIDTH)

    def query_firmware_version(self) -> str:
        """
        Query firmware version in "<major>.<minor>" format

        The firmware version reported should match the version number of
        downloadable fw files from AlazarTech. But note that the firmware
        version has often been found to be incorrect for several firmware
        versions. At the time of writing it is known to be correct for the
        9360 (v 21.07) and 9373 (v 30.04) but incorrect for several earlier
        versions. In Alazar DSO this is reported as FPGA Version.
        """
        asopc_type = self.query_asopc_type()
        # AlazarTech has confirmed in a support mail that this
        # is the way to get the firmware version
        firmware_major = (asopc_type >> 16) & 0xff
        firmware_minor = (asopc_type >> 24) & 0xf
        # firmware_minor above does not contain any prefixed zeros
        # but the minor version is always 2 digits.
        firmware_version = f'{firmware_major}.{firmware_minor:02d}'
        return firmware_version
