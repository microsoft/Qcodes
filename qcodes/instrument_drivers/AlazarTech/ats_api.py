from typing import TypeVar, Type, Dict, Tuple, Callable, NamedTuple, \
    Sequence, NewType, List, Any
from threading import Lock
import concurrent
import logging
import ctypes

from qcodes.instrument.parameter import Parameter
from .dll_wrapper import WrappedDll, DllWrapperMeta, Signature
from .constants import BOARD_NAMES, REGISTER_READING_PWD, Capability
from .utils import TraceParameter


# Define aliases for ctypes that match Alazar's notation.
U8 = ctypes.c_uint8
U32 = ctypes.c_uint32
HANDLE = U32

logger = logging.getLogger(__name__)

TApi = TypeVar("TApi", bound="AlazarATSAPI")


class AlazarATSAPI(WrappedDll):
    """
    Provides a thread-safe wrapper for the ATS API by
    isolating all calls to the API in a single thread.
    """

    ## CONSTANTS ##

    BOARD_NAMES = BOARD_NAMES
    Capability = Capability

    ## SIGNATURES ##

    signature_prefix: str = 'Alazar'
    signatures: Dict[str, Signature] = {
        'NumOfSystems': Signature(
            return_type=U32
        ),

        'BoardsInSystemBySystemID': Signature(
            return_type=U32,
            argument_types=[U32]
        ),

        'GetBoardKind': Signature(
            return_type=U32,
            argument_types=[HANDLE]
        ),

        'GetBoardBySystemID': Signature(
            return_type=HANDLE,
            argument_types=[U32, U32]
        ),

        'GetDriverVersion': Signature(
            argument_types=[
                ctypes.POINTER(U8),
                ctypes.POINTER(U8),
                ctypes.POINTER(U8)
            ]
        ),

        'GetSDKVersion': Signature(
            argument_types=[
                ctypes.POINTER(U8),
                ctypes.POINTER(U8),
                ctypes.POINTER(U8)
            ]
        ),

        'GetChannelInfo': Signature(
            argument_types=[HANDLE, ctypes.POINTER(U32), ctypes.POINTER(U8)]
        ),

        'QueryCapability': Signature(
            argument_types=[HANDLE, U32, U32, ctypes.POINTER(U32)]
        ),

        "WaitAsyncBufferComplete": Signature(
            argument_types=[U32, ctypes.c_void_p, U32]
        ),

        "ReadRegister": Signature(
            argument_types=[U32, U32, ctypes.POINTER(U32), U32]
        ),

        "WriteRegister": Signature(
            argument_types=[U32, U32, U32, U32]
        ),

        "BeforeAsyncRead": Signature(
            argument_types=[U32, U32, ctypes.c_long, U32, U32, U32, U32]
        ),

        "SetCaptureClock": Signature(
            argument_types=[U32, U32, U32, U32, U32]
        ),

        "SetRecordSize": Signature(
            argument_types=[HANDLE, U32, U32]
        ),

        "PostAsyncBuffer": Signature(
            argument_types=[U32, ctypes.c_void_p, U32]
        ),

        "AbortAsyncRead": Signature(
            argument_types=[HANDLE]
        ),

        "GetCPLDVersion": Signature(
            argument_types=[HANDLE, ctypes.POINTER(U8), ctypes.POINTER(U8)]
        ),

        "InputControl": Signature(
            argument_types=[HANDLE, U8, U32, U32, U32]
        ),

        "SetBWLimit": Signature(
            argument_types=[HANDLE, U32, U32]
        ),

        "SetTriggerOperation": Signature(
            argument_types=[
                HANDLE, U32, U32, U32, U32, U32, U32, U32, U32, U32
                ]
        ),

        "SetExternalTrigger": Signature(
            argument_types=[HANDLE, U32, U32]
        ),

        "SetTriggerDelay": Signature(
            argument_types=[HANDLE, U32]
        ),

        "SetTriggerTimeOut": Signature(
            argument_types=[HANDLE, U32]
        ),

        "ConfigureAuxIO": Signature(
            argument_types=[HANDLE, U32, U32]
        ),

        "ErrorToText": Signature(
            return_type=ctypes.c_char_p,
            argument_types=[U32]
        ),

        "StartCapture": Signature(
            argument_types=[HANDLE]
        )
    }

    ## CLASS MEMBERS ##

    # Only allow a single instance per DLL path.
    __instances: Dict[str, TApi] = {}

    def __new__(cls: Type[TApi], dll_path: str) -> TApi:
        if dll_path in cls.__instances:
            logger.debug(
                f"Found existing ATS API instance for DLL path {dll_path}.")
            return cls.__instances[dll_path]
        else:
            logger.debug(
                f"Loading new ATS API instance for DLL path {dll_path}.")
            new_api = super().__new__(cls)
            new_api.__init__(dll_path)
            cls.__instances[dll_path] = new_api
            return new_api

    ## INSTANCE MEMBERS ##

    def get_board_model(self, handle: int) -> str:
        return self.BOARD_NAMES[self.get_board_kind(handle)]

    def get_channel_info_(self, handle: int) -> Tuple[int, int]:
        """
        A more convenient version of `get_channel_info` method 
        (`AlazarGetChannelInfo`).
        
        This method hides the fact that the output values in the original
        function are written to the values of the provided pointers.

        Args:
            Handle: handle of the board of interest
        
        Returns:
            Tuple of bits per sample and maximum board memory in samples
        """
        bps = ctypes.c_uint8(0)  # bps bits per sample
        max_s = ctypes.c_uint32(0)  # max_s memory size in samples
        self.get_channel_info(
            handle,
            ctypes.byref(max_s),
            ctypes.byref(bps)
        )
        return max_s.value, bps.value
    
    def get_cpld_version_(self, handle: int) -> str:
        """
        A more convenient version of `get_cpld_version` method 
        (`AlazarGetCPLDVersion`).
        
        This method hides the fact that the output values in the original
        function are written to the values of the provided pointers.

        Args:
            handle: handle of the board of interest
        
        Returns:
            Version string in the format "<major>.<minor>"
        """
        major = ctypes.c_uint8(0)
        minor = ctypes.c_uint8(0)
        self.get_cpld_version(
            handle,
            ctypes.byref(major),
            ctypes.byref(minor)
        )
        cpld_ver = str(major.value) + "." + str(minor.value)
        return cpld_ver

    def get_driver_version_(self) -> str:
        """
        A more convenient version of `get_driver_version` method 
        (`AlazarGetDriverVersion`).
        
        This method hides the fact that the output values in the original
        function are written to the values of the provided pointers.

        Returns:
            Version string in the format "<major>.<minor>.<revision>"
        """
        major = ctypes.c_uint8(0)
        minor = ctypes.c_uint8(0)
        revision = ctypes.c_uint8(0)
        self.get_driver_version(
            ctypes.byref(major),
            ctypes.byref(minor),
            ctypes.byref(revision)
        )
        driver_ver = (str(major.value) + "." 
                      + str(minor.value) + "."
                      + str(revision.value))
        return driver_ver

    def get_sdk_version_(self) -> str:   
        """
        A more convenient version of `get_sdk_version` method 
        (`AlazarGetSDKVersion`).
        
        This method hides the fact that the output values in the original
        function are written to the values of the provided pointers.

        Returns:
            Version string in the format "<major>.<minor>.<revision>"
        """
        major = ctypes.c_uint8(0)
        minor = ctypes.c_uint8(0)
        revision = ctypes.c_uint8(0)
        self.get_sdk_version(
            ctypes.byref(major),
            ctypes.byref(minor),
            ctypes.byref(revision)
        )
        sdk_ver = (str(major.value) + "." 
                   + str(minor.value) + "."
                   + str(revision.value))
        return sdk_ver

    def query_capability_(self, handle: int, capability: int) -> int:
        """
        A more convenient version of `query_capability` method 
        (`AlazarQueryCapability`).
        
        This method hides the fact that the output values in the original
        function are written to the values of the provided pointers.

        Args:
            handle: Handle of the board of interest
            capability: An integer identifier of a capability parameter 
                (refer to constants defined in ATS API for more info)

        Returns:
            Value of the requested capability
        """
        value = ctypes.c_uint32(0)
        reserved = 0
        self.query_capability(
            handle, capability, reserved, ctypes.byref(value))
        return value.value
    
    def read_register_(self, handle: int, offset: int) -> int:
        """
        Read a value from a given register in the Alazars memory.

        A more convenient version of `read_register` method 
        (`AlazarReadRegister`).

        Args:
            handle: Handle of the board of interest
            offset: Offset into the memory to read from

        Returns:
            The value read as an integer
        """
        output = ctypes.c_uint32(0)
        self.read_register(
            handle,
            offset,
            ctypes.byref(output),
            ctypes.c_uint32(REGISTER_READING_PWD)
        )
        return output.value

    def write_register_(self, handle: int, offset: int, value: int) -> None:
        """
        Write a value to a given offset in the Alazars memory.

        A more convenient version of `write_register` method 
        (`AlazarWriteRegister`).

        Args:
            handle: Handle of the board of interest
            offset: The offset in memory to write to
            value: The value to write
        """
        self.write_register(
            handle, offset, value, ctypes.c_uint32(REGISTER_READING_PWD))


class CapabilityHelper():
    """
    A helper class providing convenient methods for various useful 
    'query capability' calls for a given Alazar board.

    Most common capabilities are enumerated in `.CAPABILITIES`.

    Frequently used capabilities even have their `query_<...>()` convenience
    methods.

    Args:
        api: Instance of Alazar ATS API class
        handle: Handle of a specific board (from `AlazarGetBoardBySystemId`)
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
        # `date_int` is a decimal number with the format DDMMYY where 
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

    def query_pcie_link_speed(self) -> int:
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
        # Alazartech has confirmed in a support mail that this
        # is the way to get the firmware version
        firmware_major = (asopc_type >> 16) & 0xff
        firmware_minor = (asopc_type >> 24) & 0xf
        # firmware_minor above does not contain any prefixed zeros
        # but the minor version is always 2 digits.
        firmware_version = f'{firmware_major}.{firmware_minor:02d}'
        return firmware_version
