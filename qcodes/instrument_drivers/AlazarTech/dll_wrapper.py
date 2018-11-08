import ctypes
# Define aliases for ctypes that match Alazar's notation.
U8 = ctypes.c_uint8
U32 = ctypes.c_uint32
HANDLE = U32

import logging
import asyncio
import concurrent
import sys
import re

from threading import Lock
from functools import partial

logger = logging.getLogger(__name__)

from typing import TypeVar, Type, Dict, Tuple, Callable, NamedTuple, Sequence, NewType, List, Any
TApi = TypeVar("TApi", bound="AlazarATSAPI")
ReturnCode = NewType('ReturnCode', ctypes.c_uint)

from qcodes.instrument.parameter import Parameter
from qcodes.instrument_drivers.AlazarTech.utils import TraceParameter

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

def api_calls(full_name : str, signature : "Signature") -> Tuple[Callable, Callable]:
    def sync_call(self : TApi, *args) -> signature.return_type:
        c_func = getattr(self._dll, full_name)
        future = self._executor.submit(
            api_call_task,
            self._lock, c_func,
            partial(mark_params_as_updated, *args),
            *normalize_params(*args)
        )
        return future.result()

    def async_call(self : TApi, *args) -> signature.return_type:
        c_func = getattr(self._dll, full_name)
        task = asyncio.get_event_loop().run_in_executor(
            self._executor,
            partial(
                api_call_task,
                self._lock, c_func,
                partial(mark_params_as_updated, *args),
                *normalize_params(*args)
            )
        )
        return task

    return sync_call, async_call

## CLASSES ##

class Signature(NamedTuple):
    return_type : Type = ReturnCode
    argument_types : Sequence[Type] = ()

class DllWrapperMeta(type):
    def __new__(mcls, name, bases, dct):
        mcls.add_api_calls(dct)
        cls = super().__new__(mcls, name, bases, dct)
        return cls

    @classmethod
    def add_api_calls(mcls, dct : Dict[str, Any]):
        prefix = dct.get('signature_prefix', '')
        for call_name, signature in dct['signatures'].items():
            full_name = prefix + call_name
            
            sync_call, async_call = api_calls(full_name, signature)

            py_name = convert_to_camel_case(call_name)
            logger.debug(f"Adding method {py_name} for C func {full_name}.")
            dct[py_name] = sync_call
            logger.debug(f"Adding async method {py_name}_async for C func {full_name}.")
            dct[py_name + "_async"] = async_call
