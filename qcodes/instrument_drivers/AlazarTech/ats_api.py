from typing import TypeVar, Type, Dict, Tuple, Callable, NamedTuple, \
    Sequence, NewType, List, Any
from threading import Lock
import concurrent
import logging
import ctypes

from qcodes.instrument.parameter import Parameter
from .dll_wrapper import WrappedDll, DllWrapperMeta, Signature
from .constants import BOARD_NAMES, REGISTER_READING_PWD
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
