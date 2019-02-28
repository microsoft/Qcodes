import ctypes
import logging
import re
from typing import TypeVar, Type, Dict, Callable, NamedTuple, Sequence, NewType, List, Any
from threading import Lock
import concurrent
from functools import partial

from qcodes.instrument.parameter import Parameter
from qcodes.instrument_drivers.AlazarTech.utils import TraceParameter
from .constants import API_SUCCESS, ERROR_CODES, ReturnCode


logger = logging.getLogger(__name__)

TApi = TypeVar("TApi", bound="AlazarATSAPI")
CReturnCode = NewType('CReturnCode', ctypes.c_uint)


## FUNCTIONS ##


def api_call_task(lock, c_func, callback, *args):
    with lock:
        retval = c_func(*args)
    callback()
    return retval


def convert_to_camel_case(name):
    # https://stackoverflow.com/a/1176023
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def normalize_params(*args) -> List[Any]:
    args_out: List[int] = []
    for arg in args:
        if isinstance(arg, Parameter):
            args_out.append(arg.raw_value)
        else:
            args_out.append(arg)
    return args_out


def mark_params_as_updated(*args) -> None:
    for arg in args:
        if isinstance(arg, TraceParameter):
            arg._set_updated()


def api_calls(full_name: str, signature: "Signature") -> Callable:
    def sync_call(self: TApi, *args) -> signature.return_type:
        c_func = getattr(self._dll, full_name)
        future = self._executor.submit(
            api_call_task,
            self._lock, 
            c_func,
            partial(mark_params_as_updated, *args),
            *normalize_params(*args)
        )
        return future.result()

    return sync_call


def check_error_code(return_code: int, func, arguments
                     ) -> None:
    if (return_code != API_SUCCESS) and (return_code != 518):
        argrepr = repr(arguments)
        if len(argrepr) > 100:
            argrepr = argrepr[:96] + '...]'

        logger.error(f'Alazar API returned code {return_code} from function '
                     f'{func.__name__} with args {argrepr}')

        if return_code not in ERROR_CODES:
            raise RuntimeError(
                'unknown error {} from function {} with args: {}'.format(
                    return_code, func.__name__, argrepr))
        raise RuntimeError(
            'error {}: {} from function {} with args: {}'.format(
                return_code, 
                ERROR_CODES[ReturnCode(return_code)],
                func.__name__,
                argrepr))

    return arguments


def convert_bytes_to_str(output: bytes, func, arguments) -> str:
    return output.decode()


## CLASSES ##


class Signature(NamedTuple):
    return_type: Type = CReturnCode
    argument_types: Sequence[Type] = ()


class DllWrapperMeta(type):
    """
    This metaclass uses `signatures` dictionary defined in a class attribute
    in order to generate methods for functions of a DLL library
    (from the `._dll` attribute of the class).

    The `signatures` dictionary supplies the information on the names of 
    the functions and types of their arguments and return values.

    Each function is potentially executed in a different thread (depending on
    what the class sets `_executor` attribute to), hence the class has to
    provide a `_lock` instance that is used to wrap around the calls to
    the DLL library.

    The original names of the DLL functions are not preserved. Instead, their 
    pythonic equivalents in snake_case are generated. If provided in 
    `signature_prefix` attribute of the class, a common prefix is removed from
    the names of the DLL functions.
    """
    def __new__(mcls, name, bases, dct):
        mcls.add_api_calls(dct)
        cls = super().__new__(mcls, name, bases, dct)
        return cls

    @classmethod
    def add_api_calls(mcls, dct: Dict[str, Any]):
        prefix = dct.get('signature_prefix', '')

        for call_name, signature in dct['signatures'].items():
            c_name = prefix + call_name
            
            api_call = api_calls(c_name, signature)

            py_name = convert_to_camel_case(call_name)
            logger.debug(f"Adding method {py_name} for C func {c_name}.")
            dct[py_name] = api_call


class WrappedDll(metaclass=DllWrapperMeta):
    """
    A base class for wrapping DLL libraries in a certain way that is defined
    by `DllWrapperMeta` metaclass. This class contains attributes that 
    subclasses have to define and/or initialize.

    Note that the use of lock, processing of the return codes (if any), etc.
    are specific to Alazar ATS DLL library.

    Args:
        dll_path: path to the DLL library to load and wrap
    """

    # These attributes define the generated class methods for DLL calls.
    signature_prefix: str = ''
    signatures: Dict[str, Signature] = {}

    # This is the DLL library instance
    _dll: ctypes.CDLL

    # This lock guards DLL calls.
    _lock: Lock

    # This executor is used to execute DLL calls.
    _executor: concurrent.futures.Executor
 
    def __init__(self, dll_path: str):
        super().__init__()
        self._dll = ctypes.cdll.LoadLibrary(dll_path)
        self.__apply_signatures()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._lock = Lock()  # ATS API DLL is not guaranteed to be thread-safe

    def __apply_signatures(self):
        """
        Adds ctypes signatures to all of the functions exposed by the ATS API
        that we use in this class.
        """
        for name, signature in self.signatures.items():
            c_func = getattr(self._dll, f"{self.signature_prefix}{name}")

            c_func.argtypes = signature.argument_types

            ret_type = signature.return_type
            if ret_type is CReturnCode:
                ret_type = ret_type.__supertype__
                c_func.errcheck = check_error_code
            elif ret_type in (ctypes.c_char_p, ctypes.c_char, 
                              ctypes.c_wchar, ctypes.c_wchar_p):
                c_func.errcheck = convert_bytes_to_str
            c_func.restype = ret_type
