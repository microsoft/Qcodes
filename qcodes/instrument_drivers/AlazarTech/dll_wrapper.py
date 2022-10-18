"""
This module provides infrastructure for wrapping DLL libraries, loaded using
:mod:`ctypes`. With this infrastructure, one just needs to subclass
:class:`.WrappedDll` and define class methods for the functions of interest
from the DLL library with mostly python types in mind, and conveniently
specify their signatures in terms of :mod:`ctypes` types.
"""
from __future__ import annotations

import concurrent
import concurrent.futures
import ctypes
import logging
from collections.abc import Callable, Sequence
from functools import partial
from threading import Lock
from typing import Any, NamedTuple, NewType, TypeVar
from weakref import WeakValueDictionary

from qcodes.parameters import ParameterBase

from .constants import API_DMA_IN_PROGRESS, API_SUCCESS, ERROR_CODES, ReturnCode
from .utils import TraceParameter

logger = logging.getLogger(__name__)

# Define aliases for ctypes that match Alazar's notation.
RETURN_CODE = NewType('RETURN_CODE', ctypes.c_uint)


# FUNCTIONS #
T = TypeVar('T')


def _api_call_task(
        lock: Lock,
        c_func: Callable[..., int],
        callback: Callable[[], None],
        *args: Any) -> int:
    with lock:
        retval = c_func(*args)
    callback()
    return retval


def _normalize_params(*args: T) -> list[T]:
    args_out: list[T] = []
    for arg in args:
        if isinstance(arg, ParameterBase):
            args_out.append(arg.raw_value)
        else:
            args_out.append(arg)
    return args_out


def _mark_params_as_updated(*args: Any) -> None:
    for arg in args:
        if isinstance(arg, TraceParameter):
            arg._set_updated()


def _check_error_code(
    return_code: int, func: Callable[..., Any], arguments: tuple[Any, ...]
) -> tuple[Any, ...]:
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


def _convert_bytes_to_str(
    output: bytes, func: Callable[..., Any], arguments: tuple[Any, ...]
) -> str:
    return output.decode()


# CLASSES #


class Signature(NamedTuple):
    return_type: type[Any] = RETURN_CODE
    argument_types: Sequence[type[Any]] = ()


class DllWrapperMeta(type):
    """DLL-path-based 'singleton' metaclass for DLL wrapper classes"""

    # Only allow a single instance per DLL path.
    _instances: WeakValueDictionary[str, Any] = WeakValueDictionary()

    def __call__(cls, dll_path: str, *args: Any, **kwargs: Any) -> Any:
        api = cls._instances.get(dll_path, None)
        if api is not None:
            logger.debug(
                f"Using existing instance for DLL path {dll_path}.")
            return api
        else:
            logger.debug(
                f"Creating new instance for DLL path {dll_path}.")
            # strong reference:
            new_api = super().__call__(dll_path, *args, **kwargs)
            cls._instances[dll_path] = new_api
            return new_api


class WrappedDll(metaclass=DllWrapperMeta):
    """
    A base class for wrapping DLL libraries.

    Note that this class is still quite specific to Alazar ATS DLL library.

    This class uses dictionary of the :attr:`signatures` attribute in order
    to assign ``argtypes`` and ``restype`` attributes for functions of
    a loaded DLL library (from the ``_dll`` attribute of the class).
    If ``restype`` is of type ``RETURN_CODE``, then an exception is
    raised in case the return code is an Alazar error code. For string-alike
    ``restype`` s, the returned value is converted to a python string.

    Functions are executed in a single separate thread (see what the
    ``_executor`` gets initialize to), hence the class also has a lock
    instance in the ``_lock`` attribute that is used to wrap around the
    actual calls to the DLL library.

    Method ``_sync_dll_call`` is supposed to be called when a subclass
    implements calls to functions of the loaded DLL.

    Args:
        dll_path: Path to the DLL library to load and wrap
    """

    signatures: dict[str, Signature] = {}
    """
    Signatures for loaded DLL functions;
    It is to be filled with :class:`Signature` instances for the DLL
    functions of interest in a subclass.
    """

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

    def __apply_signatures(self) -> None:
        """
        Apply :mod:`ctypes` signatures for all of the C library functions
        specified in :attr:`signatures` attribute.
        """
        for name, signature in self.signatures.items():
            c_func = getattr(self._dll, name)

            c_func.argtypes = signature.argument_types

            ret_type = signature.return_type
            if ret_type is RETURN_CODE:
                ret_type = ret_type.__supertype__
                c_func.errcheck = _check_error_code
            elif ret_type in (ctypes.c_char_p, ctypes.c_char,
                              ctypes.c_wchar, ctypes.c_wchar_p):
                c_func.errcheck = _convert_bytes_to_str
            c_func.restype = ret_type

    def _sync_dll_call(self, c_name: str, *args: Any) -> Any:
        """Call given function from the DLL library on the given arguments"""
        c_func = getattr(self._dll, c_name)
        future = self._executor.submit(
            _api_call_task,
            self._lock,
            c_func,
            partial(_mark_params_as_updated, *args),
            *_normalize_params(*args)
        )
        return future.result()
