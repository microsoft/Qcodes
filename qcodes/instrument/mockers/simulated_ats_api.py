"""
This module provides an api that can be passed instead for the
:class:`qcodes.instrument_drivers.AlazarTech.ats_api.AlazarATSAPI`
, which does not require a real alazar card present in the system.
This mock does currently represent a ATS9360 and does not provide
any functionality whatsoever.
"""


from typing import Tuple, Callable, Optional, Dict, Any
import numpy as np
import ctypes

from qcodes.instrument_drivers.AlazarTech.dll_wrapper import (
    _mark_params_as_updated, ReturnCode)
from qcodes.instrument_drivers.AlazarTech.constants import (
    Capability, API_SUCCESS)
from qcodes.instrument_drivers.AlazarTech.ats_api import AlazarATSAPI


class SimulatedATS9360API(AlazarATSAPI):

    registers = {
        8: 70254688,
        58: int(np.uint32(1 << 26))  # Trigger hold off
    }

    def __init__(
            self,
            dll_path: str = None,  # Need this for meta super class
            buffer_generator: Optional[Callable[[np.ndarray], None]] = None,
            dtype: type = np.uint16
    ):
        def _default_buffer_generator(buffer: np.ndarray) -> None:
            upper = buffer.size // 2
            buffer[0:upper] = 30000 * np.ones(upper)

        # don't call `super().__init__` here to avoid dependence on the
        # alazar driver, when loading the dll
        self._buffer_generator = (
            buffer_generator or _default_buffer_generator)
        self.dtype = dtype
        self.buffers: Dict[int, np.ndarray] = {}

    def _sync_dll_call(self, c_name: str, *args: Any) -> None:
        _mark_params_as_updated(*args)

    def get_board_by_system_id(self, system_id: int, board_id: int) -> int:
        # This is the handle....
        return 12335

    def busy(self, handle: int) -> int:
        return 0

    # OTHER API-RELATED METHODS

    def get_board_model(self, handle: int) -> str:
        return 'ATS9360'

    def get_channel_info_(self, handle: int) -> Tuple[int, int]:
        return 4294967294, 12

    def get_cpld_version_(self, handle: int) -> str:
        return "25.16"

    def get_driver_version_(self) -> str:
        return '6.5.1'

    def get_sdk_version_(self) -> str:
        return '6.5.1'

    def query_capability_(self, handle: int, capability: int) -> int:
        capabilities = {
            Capability.ASOPC_TYPE: 1763017568,
            Capability.GET_SERIAL_NUMBER: 970396,
            Capability.MEMORY_SIZE: 4294967294,
            Capability.GET_PCIE_LINK_WIDTH: 8,
            Capability.GET_PCIE_LINK_SPEED: 2,
            Capability.GET_LATEST_CAL_DATE: 250117}
        return capabilities[Capability(capability)]

    def read_register_(self, handle: int, offset: int) -> int:
        return self.registers[offset]

    def write_register_(self, handle: int, offset: int, value: int) -> None:
        self.registers[offset] = value

    def post_async_buffer(
            self,
            handle: int,
            buffer: ctypes.c_void_p,
            buffer_length: int
    ) -> ReturnCode:
        if buffer.value is None:
            raise RuntimeError(
                '`post_async_buffer` received buffer with invalid address.')
        ctypes_array = (ctypes.c_uint16 *
                        (buffer_length // 2)).from_address(buffer.value)
        self.buffers[buffer.value] = np.frombuffer(
            ctypes_array, dtype=self.dtype)
        self._sync_dll_call(
            'AlazarPostAsyncBuffer', handle, buffer, buffer_length)
        return API_SUCCESS

    def wait_async_buffer_complete(
            self,
            handle: int,
            buffer: ctypes.c_void_p,
            timeout_in_ms: int
    ) -> ReturnCode:
        if buffer.value is None:
            raise RuntimeError(
                '`wait_async_buffer` received buffer with invalid address.')
        b = self.buffers.get(buffer.value)
        if b is None:
            raise RuntimeError("received an empty buffer")
        self._buffer_generator(b)
        self._sync_dll_call(
            'AlazarWaitAsyncBufferComplete', handle, buffer, timeout_in_ms)
        return API_SUCCESS
