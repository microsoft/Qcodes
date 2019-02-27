from typing import TypeVar, Type, Dict, Tuple, Callable, NamedTuple, \
    Sequence, NewType, List, Any
from threading import Lock
import concurrent
import logging
import ctypes

from qcodes.instrument.parameter import Parameter
from .dll_wrapper import DllWrapperMeta, Signature
from .constants import API_SUCCESS, ERROR_CODES, ReturnCode, BOARD_NAMES
from .utils import TraceParameter


# Define aliases for ctypes that match Alazar's notation.
U8 = ctypes.c_uint8
U32 = ctypes.c_uint32
HANDLE = U32

logger = logging.getLogger(__name__)

TApi = TypeVar("TApi", bound="AlazarATSAPI")


def check_error_code(return_code_c: ctypes.c_uint, func, arguments
                     ) -> None:
    return_code = int(return_code_c.value)

    if (return_code != API_SUCCESS) and (return_code != 518):
        # TODO(damazter) (C) log error
        argrepr = repr(arguments)
        if len(argrepr) > 100:
            argrepr = argrepr[:96] + '...]'

        if return_code not in ERROR_CODES:
            raise RuntimeError(
                'unknown error {} from function {} with args: {}'.format(
                    return_code, func.__name__, argrepr))
        raise RuntimeError(
            'error {}: {} from function {} with args: {}'.format(
                return_code, ERROR_CODES[ReturnCode(
                    return_code)], func.__name__,
                argrepr))


def convert_bytes_to_str(output: bytes, func, arguments) -> str:
    return output.decode()


class AlazarATSAPI(object, metaclass=DllWrapperMeta):
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
            argument_types=[
                U32
            ]
        ),

        'GetBoardKind': Signature(
            return_type=U32,
            argument_types=[
                HANDLE
            ]
        ),

        'GetBoardBySystemID': Signature(
            return_type=HANDLE,
            argument_types=[
                U32, U32
            ]
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
            argument_types=[
                HANDLE,
                ctypes.POINTER(U32),
                ctypes.POINTER(U8)
            ]
        ),

        'QueryCapability': Signature(
            argument_types=[
                HANDLE, U32, U32,
                ctypes.POINTER(U32)
            ]
        ),

        "WaitAsyncBufferComplete": Signature(
            argument_types=[
                U32, ctypes.c_void_p, U32
            ]
        ),

        "ReadRegister": Signature(
            argument_types=[
                U32,
                U32,
                ctypes.POINTER(U32),
                U32
            ]
        ),

        "WriteRegister": Signature(
            argument_types=[
                U32,
                U32,
                U32,
                U32
            ]
        ),

        "BeforeAsyncRead": Signature(
            argument_types=[
                U32,
                U32,
                ctypes.c_long,
                U32,
                U32,
                U32,
                U32
            ]
        ),

        "SetCaptureClock": Signature(
            argument_types=[
                U32,
                U32,
                U32,
                U32,
                U32
            ]
        ),

        "SetRecordSize": Signature(
            argument_types=[
                HANDLE, U32, U32
            ]
        ),

        "PostAsyncBuffer": Signature(
            argument_types=[
                U32,
                ctypes.c_void_p,
                U32
            ]
        ),

        "AbortAsyncRead": Signature(
            argument_types=[
                HANDLE
            ]
        ),

        "GetCPLDVersion": Signature(
            argument_types=[
                HANDLE, ctypes.POINTER(U8), ctypes.POINTER(U8)
            ]
        ),

        "InputControl": Signature(
            argument_types=[
                HANDLE, U8, U32, U32, U32
            ]
        ),

        "SetBWLimit": Signature(
            argument_types=[
                HANDLE, U32, U32
            ]
        ),

        "SetTriggerOperation": Signature(
            argument_types=[
                HANDLE, U32, U32, U32, U32, U32, U32, U32, U32, U32
            ]
        ),

        "SetExternalTrigger": Signature(
            argument_types=[
                HANDLE, U32, U32
            ]
        ),

        "SetTriggerDelay": Signature(
            argument_types=[
                HANDLE, U32
            ]
        ),

        "SetTriggerTimeOut": Signature(
            argument_types=[
                HANDLE, U32
            ]
        ),

        "ConfigureAuxIO": Signature(
            argument_types=[
                HANDLE, U32, U32
            ]
        ),

        "ErrorToText": Signature(
            return_type=ctypes.c_char_p,
            argument_types=[
                U32
            ]
        ),

        "StartCapture": Signature(
            argument_types=[
                HANDLE
            ]
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

    _dll: ctypes.CDLL
    _executor: concurrent.futures.Executor

    #: The ATS API DLL is not guaranteed to be thread-safe
    #: This lock guards API calls.
    _lock: Lock

    def __init__(self, dll_path: str):
        self._dll = ctypes.cdll.LoadLibrary(dll_path)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._lock = Lock()
        self.__apply_signatures()

    def __apply_signatures(self):
        """
        Adds ctypes signatures to all of the functions exposed by the ATS API
        that we use in this class.
        """
        for name, signature in self.signatures.items():
            c_func = getattr(self._dll, f"{self.signature_prefix}{name}")
            c_func.argtypes = signature.argument_types
            ret_type = signature.return_type
            if ret_type is ReturnCode:
                ret_type = ret_type.__supertype__
                c_func.errcheck = check_error_code
            elif ret_type in (ctypes.c_char_p, ctypes.c_char, 
                              ctypes.c_wchar, ctypes.c_wchar_p):
                c_func.errcheck = convert_bytes_to_str

            c_func.restype = ret_type
