from .dll_wrapper import DllWrapperMeta, Signature
from .utils import TraceParameter
from qcodes.instrument.parameter import Parameter
from typing import TypeVar, Type, Dict, Tuple, Callable, NamedTuple, \
    Sequence, NewType, List, Any
from functools import partial
from threading import Lock
import re
import sys
import concurrent
import asyncio
import logging
import ctypes


# Define aliases for ctypes that match Alazar's notation.
U8 = ctypes.c_uint8
U32 = ctypes.c_uint32
HANDLE = U32

logger = logging.getLogger(__name__)

TApi = TypeVar("TApi", bound="AlazarATSAPI")
ReturnCode = NewType('ReturnCode', int)


## CONSTANTS ##

API_SUCCESS = 512

ERROR_CODES: Dict[ReturnCode, str] = {ReturnCode(code): msg for code, msg in {
    513: 'ApiFailed',
    514: 'ApiAccessDenied',
    515: 'ApiDmaChannelUnavailable',
    516: 'ApiDmaChannelInvalid',
    517: 'ApiDmaChannelTypeError',
    518: 'ApiDmaInProgress',
    519: 'ApiDmaDone',
    520: 'ApiDmaPaused',
    521: 'ApiDmaNotPaused',
    522: 'ApiDmaCommandInvalid',
    523: 'ApiDmaManReady',
    524: 'ApiDmaManNotReady',
    525: 'ApiDmaInvalidChannelPriority',
    526: 'ApiDmaManCorrupted',
    527: 'ApiDmaInvalidElementIndex',
    528: 'ApiDmaNoMoreElements',
    529: 'ApiDmaSglInvalid',
    530: 'ApiDmaSglQueueFull',
    531: 'ApiNullParam',
    532: 'ApiInvalidBusIndex',
    533: 'ApiUnsupportedFunction',
    534: 'ApiInvalidPciSpace',
    535: 'ApiInvalidIopSpace',
    536: 'ApiInvalidSize',
    537: 'ApiInvalidAddress',
    538: 'ApiInvalidAccessType',
    539: 'ApiInvalidIndex',
    540: 'ApiMuNotReady',
    541: 'ApiMuFifoEmpty',
    542: 'ApiMuFifoFull',
    543: 'ApiInvalidRegister',
    544: 'ApiDoorbellClearFailed',
    545: 'ApiInvalidUserPin',
    546: 'ApiInvalidUserState',
    547: 'ApiEepromNotPresent',
    548: 'ApiEepromTypeNotSupported',
    549: 'ApiEepromBlank',
    550: 'ApiConfigAccessFailed',
    551: 'ApiInvalidDeviceInfo',
    552: 'ApiNoActiveDriver',
    553: 'ApiInsufficientResources',
    554: 'ApiObjectAlreadyAllocated',
    555: 'ApiAlreadyInitialized',
    556: 'ApiNotInitialized',
    557: 'ApiBadConfigRegEndianMode',
    558: 'ApiInvalidPowerState',
    559: 'ApiPowerDown',
    560: 'ApiFlybyNotSupported',
    561: 'ApiNotSupportThisChannel',
    562: 'ApiNoAction',
    563: 'ApiHSNotSupported',
    564: 'ApiVPDNotSupported',
    565: 'ApiVpdNotEnabled',
    566: 'ApiNoMoreCap',
    567: 'ApiInvalidOffset',
    568: 'ApiBadPinDirection',
    569: 'ApiPciTimeout',
    570: 'ApiDmaChannelClosed',
    571: 'ApiDmaChannelError',
    572: 'ApiInvalidHandle',
    573: 'ApiBufferNotReady',
    574: 'ApiInvalidData',
    575: 'ApiDoNothing',
    576: 'ApiDmaSglBuildFailed',
    577: 'ApiPMNotSupported',
    578: 'ApiInvalidDriverVersion',
    579: ('ApiWaitTimeout: operation did not finish during '
          'timeout interval. Check your trigger.'),
    580: 'ApiWaitCanceled',
    581: 'ApiBufferTooSmall',
    582: ('ApiBufferOverflow: rate of acquiring data > rate of '
          'transferring data to local memory. Try reducing sample rate, '
          'reducing number of enabled channels, increasing size of each '
          'DMA buffer or increase number of DMA buffers.'),
    583: 'ApiInvalidBuffer',
    584: 'ApiInvalidRecordsPerBuffer',
    585: ('ApiDmaPending: Async I/O operation was successfully started, '
          'it will be completed when sufficient trigger events are '
          'supplied to fill the buffer.'),
    586: ('ApiLockAndProbePagesFailed:Driver or operating system was '
          'unable to prepare the specified buffer for DMA transfer. '
          'Try reducing buffer size or total number of buffers.'),
    587: 'ApiWaitAbandoned',
    588: 'ApiWaitFailed',
    589: ('ApiTransferComplete: This buffer is last in the current '
          'acquisition.'),
    590: 'ApiPllNotLocked: hardware error, contact AlazarTech',
    591: ('ApiNotSupportedInDualChannelMode:Requested number of samples '
          'per channel is too large to fit in on-board memory. Try '
          'reducing number of samples per channel, or switch to '
          'single channel mode.'),
    592: ('ApiNotSupportedInQuadChannelMode: The requested number of '
          'samples per channel is too large to fit in onboard memory. '
          'Try reducing the num'),
    593: 'ApiFileIoError: A file read or write error occurred.',
    594: ('ApiInvalidClockFrequency: The requested ADC clock frequency is '
          'not supported.'),
    595: 'ApiInvalidSkipTable',
    596: 'ApiInvalidDspModule',
    597: 'ApiDESOnlySupportedInSingleChannelMode',
    598: 'ApiInconsistentChannel',
    599: 'ApiDspFiniteRecordsPerAcquisition',
    600: 'ApiNotEnoughNptFooters',
    601: 'ApiInvalidNptFooter',
    602: ('ApiOCTIgnoreBadClockNotSupported: OCT ignore bad clock is not '
          'supported.'),
    603: ('ApiError: The requested number of records in a single-port '
          'acquisition exceeds the maximum supported by the digitizer. '
          'Use dual-ported AutoDMA to acquire more records per acquisition.'),
    604: ('ApiError: The requested number of records in a single-port '
          'acquisition exceeds the maximum supported by the digitizer.'),
    605: ('ApiOCTNoTriggerDetected: No trigger is detected for OCT ignore '
          'bad clock feature.'),
    606: ('ApiOCTTriggerTooFast: Trigger detected is too fast for OCT ignore '
          'bad clock feature.'),
    607: ('ApiNetworkError: There was an issue related to network. Make sure '
          'that the network connection and settings are correct.'),
    608: ('ApiFftSizeTooLarge: The on-FPGA FFT cannot support FFT that '
          'large. Try reducing the FFT size.'),
}.items()}

BOARD_NAMES = {
    0: 'ATS_NONE',
    1: 'ATS850',
    2: 'ATS310',
    3: 'ATS330',
    4: 'ATS855',
    5: 'ATS315',
    6: 'ATS335',
    7: 'ATS460',
    8: 'ATS860',
    9: 'ATS660',
    10: 'ATS665',
    11: 'ATS9462',
    12: 'ATS9434',
    13: 'ATS9870',
    14: 'ATS9350',
    15: 'ATS9325',
    16: 'ATS9440',
    17: 'ATS9410',
    18: 'ATS9351',
    19: 'ATS9310',
    20: 'ATS9461',
    21: 'ATS9850',
    22: 'ATS9625',
    23: 'ATG6500',
    24: 'ATS9626',
    25: 'ATS9360',
    26: 'AXI9870',
    27: 'ATS9370',
    28: 'ATU7825',
    29: 'ATS9373',
    30: 'ATS9416',
    31: 'ATS9637',
    32: 'ATS9120',
    33: 'ATS9371',
    34: 'ATS9130',
    35: 'ATS9352',
    36: 'ATS9453',
}


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

        if sys.platform == 'win32':
            ctypes.windll.kernel32.VirtualAlloc.argtypes = [
                ctypes.c_void_p,
                ctypes.c_long,
                ctypes.c_long,
                ctypes.c_long
            ]
            ctypes.windll.kernel32.VirtualFree.argtypes = [
                ctypes.c_void_p,
                ctypes.c_long,
                ctypes.c_long
            ]
