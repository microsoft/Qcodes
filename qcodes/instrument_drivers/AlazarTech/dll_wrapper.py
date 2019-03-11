import ctypes
import logging
from typing import Type, Dict, NamedTuple, Sequence, NewType, List, Any
from threading import Lock
import concurrent
from functools import partial
from weakref import WeakValueDictionary

from qcodes.instrument.parameter import _BaseParameter
from .constants import API_SUCCESS, API_DMA_IN_PROGRESS, ERROR_CODES, \
    ReturnCode


logger = logging.getLogger(__name__)

# Define aliases for ctypes that match Alazar's notation.
RETURN_CODE = NewType('RETURN_CODE', ctypes.c_uint)


## FUNCTIONS ##


def api_call_task(lock, c_func, callback, *args):
    with lock:
        retval = c_func(*args)
    callback()
    return retval


def normalize_params(*args: Any) -> List[Any]:
    args_out: List[Any] = []
    for arg in args:
        if isinstance(arg, _BaseParameter):
            args_out.append(arg.raw_value)
        else:
            args_out.append(arg)
    return args_out


def mark_params_as_updated(*args) -> None:
    for arg in args:
        if isinstance(arg, TraceParameter):
            arg._set_updated()


def check_error_code(return_code: int, func, arguments
                     ) -> None:
    if (return_code != API_SUCCESS) and (return_code != API_DMA_IN_PROGRESS):
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
    return_type: Type = RETURN_CODE
    argument_types: Sequence[Type] = ()


class DllWrapperMeta(type):
    """DLL-path-based 'singleton' metaclass for DLL wrapper classes"""

    # Only allow a single instance per DLL path.
    _instances: WeakValueDictionary = WeakValueDictionary()  # of [str, Any]

    # Note: without the 'type: ignore' for the `__call__` method below, mypy
    # generates 'Signature of "__call__" incompatible with supertype "type"'
    # error, which is an indicator of Liskov principle violation - subtypes
    # should not change the method signatures, but we need it here in order to
    # use the `dll_path` argument which `type` superclass obviously does not
    # have in its `__call__` method.
    def __call__(  # type: ignore
            cls, dll_path: int):
        api = cls._instances.get(dll_path, None)
        if api is not None:
            logger.debug(
                f"Using existing instance for DLL path {dll_path}.")
            return api
        else:
            logger.debug(
                f"Creating new instance for DLL path {dll_path}.")
            new_api = super().__call__(dll_path)  # <- strong reference
            cls._instances[dll_path] = new_api
            return new_api


class WrappedDll(metaclass=DllWrapperMeta):
    """
    A base class for wrapping DLL libraries. This class contains attributes
    that subclasses have to define and/or initialize.

    This class uses `signatures` dictionary defined in a class attribute
    in order to assign `argtypes` and `restype` atttributes for functions of
    a loaded DLL library (from the `._dll` attribute of the class).

    Each function is potentially executed in a different thread (depending on
    what the class sets `_executor` attribute to), hence the class has to
    provide a `_lock` instance that is used to wrap around the calls to
    the DLL library.

    Note that the use of lock, processing of the return codes (if any), etc.
    are specific to Alazar ATS DLL library.

    Args:
        dll_path: path to the DLL library to load and wrap
    """

    # This defines ctypes signatures for loaded DLL functions.
    signatures: Dict[str, Signature] = {}

    # This is the DLL library instance.
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
            c_func = getattr(self._dll, name)

            c_func.argtypes = signature.argument_types

            ret_type = signature.return_type
            if ret_type is RETURN_CODE:
                ret_type = ret_type.__supertype__
                c_func.errcheck = check_error_code
            elif ret_type in (ctypes.c_char_p, ctypes.c_char, 
                              ctypes.c_wchar, ctypes.c_wchar_p):
                c_func.errcheck = convert_bytes_to_str
            c_func.restype = ret_type

    def _sync_dll_call(self, c_name: str, *args: Any) -> Any:
        c_func = getattr(self._dll, c_name)
        future = self._executor.submit(
            api_call_task,
            self._lock,
            c_func,
            partial(mark_params_as_updated, *args), #TODO make optional
            *normalize_params(*args) #TODO make optional
        )
        return future.result()
