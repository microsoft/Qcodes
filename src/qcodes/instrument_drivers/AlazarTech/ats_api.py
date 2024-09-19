"""
This module provides a class that encapsulates Alazar ATS API,
:class:`AlazarATSAPI`. The class is used to expose Alazar API functions
of its C library in a python-friendly way.
"""

import ctypes
from ctypes import POINTER
from typing import TYPE_CHECKING, Any, ClassVar, Union

# `ParameterBase` is needed because users may pass instrument parameters
# that originate from `Instrument.parameters` dictionary which is typed
# with `ParameterBase`, not `Parameter`.
from .constants import BOARD_NAMES, REGISTER_ACCESS_PASSWORD, ReturnCode
from .dll_wrapper import Signature, WrappedDll

if TYPE_CHECKING:
    from qcodes.parameters import ParameterBase as Parameter

# Define aliases for ctypes that match Alazar's notation.
U8 = ctypes.c_uint8
U16 = ctypes.c_uint16
U32 = ctypes.c_uint32
C_LONG = ctypes.c_long
HANDLE = ctypes.c_void_p

POINTER_c_uint8 = Any
POINTER_c_uint16 = Any
POINTER_c_uint32 = Any
POINTER_c_long = Any


class AlazarATSAPI(WrappedDll):
    """
    A thread-safe wrapper for the ATS API library.

    The exposed ATS API functions have snake_case names, may accept python
    types and :class:`qcodes.instrument.parameter.Parameter` s as input.
    The API calls are thread-safe, and are executed in a separate thread.
    Moreover, behind the scenes it is ensured that only single instance of
    this class exists per unique ATS API DLL file.

    Some of the ATS API functions are also exposed with more convenient
    signatures. These usually have the same name but with an additional
    underscore at the end.
    """

    ## CONSTANTS ##

    BOARD_NAMES = BOARD_NAMES

    ## ACTUAL DLL API FUNCTIONS ##

    signatures: ClassVar[dict[str, Signature]] = {}

    def set_trigger_time_out(
        self, handle: int, timeout_ticks: Union[int, "Parameter"]
    ) -> ReturnCode:
        return self._sync_dll_call("AlazarSetTriggerTimeOut", handle, timeout_ticks)

    signatures.update(
        {"AlazarSetTriggerTimeOut": Signature(argument_types=[HANDLE, U32])}
    )

    def get_board_kind(self, handle: int) -> int:
        return self._sync_dll_call("AlazarGetBoardKind", handle)

    signatures.update(
        {"AlazarGetBoardKind": Signature(argument_types=[HANDLE], return_type=U32)}
    )

    def get_channel_info(
        self,
        handle: int,
        memory_size_in_samples: POINTER_c_uint32,
        bits_per_sample: POINTER_c_uint8,
    ) -> ReturnCode:
        return self._sync_dll_call(
            "AlazarGetChannelInfo", handle, memory_size_in_samples, bits_per_sample
        )

    signatures.update(
        {
            "AlazarGetChannelInfo": Signature(
                argument_types=[HANDLE, POINTER(U32), POINTER(U8)]
            )
        }
    )

    def get_cpld_version(
        self, handle: int, major: POINTER_c_uint8, minor: POINTER_c_uint8
    ) -> ReturnCode:
        return self._sync_dll_call("AlazarGetCPLDVersion", handle, major, minor)

    signatures.update(
        {
            "AlazarGetCPLDVersion": Signature(
                argument_types=[HANDLE, POINTER(U8), POINTER(U8)]
            )
        }
    )

    def get_driver_version(
        self, major: POINTER_c_uint8, minor: POINTER_c_uint8, revision: POINTER_c_uint8
    ) -> ReturnCode:
        return self._sync_dll_call("AlazarGetDriverVersion", major, minor, revision)

    signatures.update(
        {
            "AlazarGetDriverVersion": Signature(
                argument_types=[POINTER(U8), POINTER(U8), POINTER(U8)]
            )
        }
    )

    def get_sdk_version(
        self, major: POINTER_c_uint8, minor: POINTER_c_uint8, revision: POINTER_c_uint8
    ) -> ReturnCode:
        return self._sync_dll_call("AlazarGetSDKVersion", major, minor, revision)

    signatures.update(
        {
            "AlazarGetSDKVersion": Signature(
                argument_types=[POINTER(U8), POINTER(U8), POINTER(U8)]
            )
        }
    )

    def query_capability(
        self, handle: int, capability: int, reserved: int, value: POINTER_c_uint32
    ) -> ReturnCode:
        return self._sync_dll_call(
            "AlazarQueryCapability", handle, capability, reserved, value
        )

    signatures.update(
        {
            "AlazarQueryCapability": Signature(
                argument_types=[HANDLE, U32, U32, POINTER(U32)]
            )
        }
    )

    def read_register(
        self,
        handle: int,
        offset: int,
        output: POINTER_c_uint32,
        password: int,
    ) -> ReturnCode:
        return self._sync_dll_call(
            "AlazarReadRegister", handle, offset, output, password
        )

    signatures.update(
        {
            "AlazarReadRegister": Signature(
                argument_types=[HANDLE, U32, POINTER(U32), U32]
            )
        }
    )

    def write_register(
        self, handle: int, offset: int, value: int, password: int
    ) -> ReturnCode:
        return self._sync_dll_call(
            "AlazarWriteRegister", handle, offset, value, password
        )

    signatures.update(
        {"AlazarWriteRegister": Signature(argument_types=[HANDLE, U32, U32, U32])}
    )

    def num_of_systems(self) -> int:
        return self._sync_dll_call("AlazarNumOfSystems")

    signatures.update({"AlazarNumOfSystems": Signature(return_type=U32)})

    def boards_in_system_by_system_id(self, system_id: int) -> int:
        return self._sync_dll_call("AlazarBoardsInSystemBySystemID", system_id)

    signatures.update(
        {
            "AlazarBoardsInSystemBySystemID": Signature(
                argument_types=[U32], return_type=U32
            )
        }
    )

    def get_board_by_system_id(self, system_id: int, board_id: int) -> int:
        return self._sync_dll_call("AlazarGetBoardBySystemID", system_id, board_id)

    signatures.update(
        {
            "AlazarGetBoardBySystemID": Signature(
                argument_types=[U32, U32], return_type=HANDLE
            )
        }
    )

    def get_parameter(
        self, handle: int, channel: int, parameter: int, ret_value: POINTER_c_long
    ) -> ReturnCode:
        return self._sync_dll_call(
            "AlazarGetParameter", handle, channel, parameter, ret_value
        )

    signatures.update(
        {
            "AlazarGetParameter": Signature(
                argument_types=[HANDLE, U8, U32, POINTER(C_LONG)]
            )
        }
    )

    def set_parameter(
        self, handle: int, channel: int, parameter: int, value: int
    ) -> ReturnCode:
        return self._sync_dll_call(
            "AlazarSetParameter", handle, channel, parameter, value
        )

    signatures.update(
        {"AlazarSetParameter": Signature(argument_types=[HANDLE, U8, U32, C_LONG])}
    )

    def set_capture_clock(
        self,
        handle: int,
        source_id: Union[int, "Parameter"],
        sample_rate_id_or_value: Union[int, "Parameter"],
        edge_id: Union[int, "Parameter"],
        decimation: Union[int, "Parameter"],
    ) -> ReturnCode:
        return self._sync_dll_call(
            "AlazarSetCaptureClock",
            handle,
            source_id,
            sample_rate_id_or_value,
            edge_id,
            decimation,
        )

    signatures.update(
        {
            "AlazarSetCaptureClock": Signature(
                argument_types=[HANDLE, U32, U32, U32, U32]
            )
        }
    )

    def input_control(
        self,
        handle: int,
        channel_id: Union[int, "Parameter"],
        coupling_id: Union[int, "Parameter"],
        range_id: Union[int, "Parameter"],
        impedance_id: Union[int, "Parameter"],
    ) -> ReturnCode:
        return self._sync_dll_call(
            "AlazarInputControl",
            handle,
            channel_id,
            coupling_id,
            range_id,
            impedance_id,
        )

    signatures.update(
        {"AlazarInputControl": Signature(argument_types=[HANDLE, U8, U32, U32, U32])}
    )

    def set_bw_limit(
        self,
        handle: int,
        channel_id: Union[int, "Parameter"],
        flag: Union[int, "Parameter"],
    ) -> ReturnCode:
        return self._sync_dll_call("AlazarSetBWLimit", handle, channel_id, flag)

    signatures.update(
        {"AlazarSetBWLimit": Signature(argument_types=[HANDLE, U32, U32])}
    )

    def set_trigger_operation(
        self,
        handle: int,
        trigger_operation: Union[int, "Parameter"],
        trigger_engine_id_1: Union[int, "Parameter"],
        source_id_1: Union[int, "Parameter"],
        slope_id_1: Union[int, "Parameter"],
        level_1: Union[int, "Parameter"],
        trigger_engine_id_2: Union[int, "Parameter"],
        source_id_2: Union[int, "Parameter"],
        slope_id_2: Union[int, "Parameter"],
        level_2: Union[int, "Parameter"],
    ) -> ReturnCode:
        return self._sync_dll_call(
            "AlazarSetTriggerOperation",
            handle,
            trigger_operation,
            trigger_engine_id_1,
            source_id_1,
            slope_id_1,
            level_1,
            trigger_engine_id_2,
            source_id_2,
            slope_id_2,
            level_2,
        )

    signatures.update(
        {
            "AlazarSetTriggerOperation": Signature(
                argument_types=[HANDLE, U32, U32, U32, U32, U32, U32, U32, U32, U32]
            )
        }
    )

    def set_external_trigger(
        self,
        handle: int,
        coupling_id: Union[int, "Parameter"],
        range_id: Union[int, "Parameter"],
    ) -> ReturnCode:
        return self._sync_dll_call(
            "AlazarSetExternalTrigger", handle, coupling_id, range_id
        )

    signatures.update(
        {"AlazarSetExternalTrigger": Signature(argument_types=[HANDLE, U32, U32])}
    )

    def set_trigger_delay(
        self, handle: int, value: Union[int, "Parameter"]
    ) -> ReturnCode:
        return self._sync_dll_call("AlazarSetTriggerDelay", handle, value)

    signatures.update(
        {"AlazarSetTriggerDelay": Signature(argument_types=[HANDLE, U32])}
    )

    def configure_aux_io(
        self,
        handle: int,
        mode_id: Union[int, "Parameter"],
        mode_parameter_value: Union[int, "Parameter"],
    ) -> ReturnCode:
        return self._sync_dll_call(
            "AlazarConfigureAuxIO", handle, mode_id, mode_parameter_value
        )

    signatures.update(
        {"AlazarConfigureAuxIO": Signature(argument_types=[HANDLE, U32, U32])}
    )

    def set_record_size(
        self,
        handle: int,
        pre_trigger_samples: Union[int, "Parameter"],
        post_trigger_samples: Union[int, "Parameter"],
    ) -> ReturnCode:
        return self._sync_dll_call(
            "AlazarSetRecordSize", handle, pre_trigger_samples, post_trigger_samples
        )

    signatures.update(
        {"AlazarSetRecordSize": Signature(argument_types=[HANDLE, U32, U32])}
    )

    def before_async_read(
        self,
        handle: int,
        channel_select: int,
        transfer_offset: int,
        samples_per_record: int,
        records_per_buffer: int,
        records_per_acquisition: int,
        flags: int,
    ) -> ReturnCode:
        return self._sync_dll_call(
            "AlazarBeforeAsyncRead",
            handle,
            channel_select,
            transfer_offset,
            samples_per_record,
            records_per_buffer,
            records_per_acquisition,
            flags,
        )

    signatures.update(
        {
            "AlazarBeforeAsyncRead": Signature(
                argument_types=[HANDLE, U32, C_LONG, U32, U32, U32, U32]
            )
        }
    )

    def post_async_buffer(
        self, handle: int, buffer: ctypes.c_void_p, buffer_length: int
    ) -> ReturnCode:
        return self._sync_dll_call(
            "AlazarPostAsyncBuffer", handle, buffer, buffer_length
        )

    signatures.update(
        {
            "AlazarPostAsyncBuffer": Signature(
                argument_types=[HANDLE, ctypes.c_void_p, U32]
            )
        }
    )

    def wait_async_buffer_complete(
        self, handle: int, buffer: ctypes.c_void_p, timeout_in_ms: int
    ) -> ReturnCode:
        return self._sync_dll_call(
            "AlazarWaitAsyncBufferComplete", handle, buffer, timeout_in_ms
        )

    signatures.update(
        {
            "AlazarWaitAsyncBufferComplete": Signature(
                argument_types=[HANDLE, ctypes.c_void_p, U32]
            )
        }
    )

    def start_capture(self, handle: int) -> ReturnCode:
        return self._sync_dll_call("AlazarStartCapture", handle)

    signatures.update({"AlazarStartCapture": Signature(argument_types=[HANDLE])})

    def abort_async_read(self, handle: int) -> ReturnCode:
        return self._sync_dll_call("AlazarAbortAsyncRead", handle)

    signatures.update({"AlazarAbortAsyncRead": Signature(argument_types=[HANDLE])})

    def error_to_text(self, return_code: ReturnCode) -> str:
        return self._sync_dll_call("AlazarErrorToText", return_code)

    signatures.update(
        {
            "AlazarErrorToText": Signature(
                argument_types=[U32], return_type=ctypes.c_char_p
            )
        }
    )

    def force_trigger(self, handle: int) -> ReturnCode:
        return self._sync_dll_call("AlazarForceTrigger", handle)

    signatures.update({"AlazarForceTrigger": Signature(argument_types=[HANDLE])})

    def force_trigger_enable(self, handle: int) -> ReturnCode:
        return self._sync_dll_call("AlazarForceTriggerEnable", handle)

    signatures.update({"AlazarForceTriggerEnable": Signature(argument_types=[HANDLE])})

    def busy(self, handle: int) -> int:
        return self._sync_dll_call("AlazarBusy", handle)

    signatures.update(
        {"AlazarBusy": Signature(argument_types=[HANDLE], return_type=U32)}
    )

    def configure_record_average(
        self,
        handle: int,
        mode: int,
        samples_per_record: int,
        records_per_average: int,
        options: int,
    ) -> ReturnCode:
        return self._sync_dll_call(
            "AlazarConfigureRecordAverage",
            handle,
            mode,
            samples_per_record,
            records_per_average,
            options,
        )

    signatures.update(
        {
            "AlazarConfigureRecordAverage": Signature(
                argument_types=[HANDLE, U32, U32, U32, U32]
            )
        }
    )

    def free_buffer_u16(self, handle: int, buffer: POINTER_c_uint16) -> ReturnCode:
        return self._sync_dll_call("AlazarFreeBufferU16", handle, buffer)

    signatures.update(
        {"AlazarFreeBufferU16": Signature(argument_types=[HANDLE, POINTER(U16)])}
    )

    def free_buffer_u8(self, handle: int, buffer: POINTER_c_uint8) -> ReturnCode:
        return self._sync_dll_call("AlazarFreeBufferU8", handle, buffer)

    signatures.update(
        {"AlazarFreeBufferU8": Signature(argument_types=[HANDLE, POINTER(U8)])}
    )

    def set_led(self, handle: int, led_on: bool) -> ReturnCode:
        return self._sync_dll_call("AlazarSetLED", handle, led_on)

    signatures.update({"AlazarSetLED": Signature(argument_types=[HANDLE, U32])})

    def triggered(self, handle: int) -> int:
        return self._sync_dll_call("AlazarTriggered", handle)

    signatures.update(
        {"AlazarTriggered": Signature(argument_types=[HANDLE], return_type=U32)}
    )

    ## OTHER API-RELATED METHODS ##

    def get_board_model(self, handle: int) -> str:
        return self.BOARD_NAMES[self.get_board_kind(handle)]

    def get_channel_info_(self, handle: int) -> tuple[int, int]:
        """
        A more convenient version of :meth:`get_channel_info` method
        (``AlazarGetChannelInfo``).

        This method hides the fact that the output values in the original
        function are written to the provided pointers.

        Args:
            handle: Handle of the board of interest

        Returns:
            Tuple of bits per sample and maximum board memory in samples
        """
        bps = ctypes.c_uint8(0)  # bps bits per sample
        max_s = ctypes.c_uint32(0)  # max_s memory size in samples
        self.get_channel_info(handle, ctypes.byref(max_s), ctypes.byref(bps))
        return max_s.value, bps.value

    def get_cpld_version_(self, handle: int) -> str:
        """
        A more convenient version of :meth:`get_cpld_version` method
        (``AlazarGetCPLDVersion``).

        This method hides the fact that the output values in the original
        function are written to the provided pointers.

        Args:
            handle: Handle of the board of interest

        Returns:
            Version string in the format "<major>.<minor>"
        """
        major = ctypes.c_uint8(0)
        minor = ctypes.c_uint8(0)
        self.get_cpld_version(handle, ctypes.byref(major), ctypes.byref(minor))
        cpld_ver = str(major.value) + "." + str(minor.value)
        return cpld_ver

    def get_driver_version_(self) -> str:
        """
        A more convenient version of :meth:`get_driver_version` method
        (``AlazarGetDriverVersion``).

        This method hides the fact that the output values in the original
        function are written to the provided pointers.

        Returns:
            Version string in the format "<major>.<minor>.<revision>"
        """
        major = ctypes.c_uint8(0)
        minor = ctypes.c_uint8(0)
        revision = ctypes.c_uint8(0)
        self.get_driver_version(
            ctypes.byref(major), ctypes.byref(minor), ctypes.byref(revision)
        )
        driver_ver = (
            str(major.value) + "." + str(minor.value) + "." + str(revision.value)
        )
        return driver_ver

    def get_sdk_version_(self) -> str:
        """
        A more convenient version of :meth:`get_sdk_version` method
        (``AlazarGetSDKVersion``).

        This method hides the fact that the output values in the original
        function are written to the provided pointers.

        Returns:
            Version string in the format "<major>.<minor>.<revision>"
        """
        major = ctypes.c_uint8(0)
        minor = ctypes.c_uint8(0)
        revision = ctypes.c_uint8(0)
        self.get_sdk_version(
            ctypes.byref(major), ctypes.byref(minor), ctypes.byref(revision)
        )
        sdk_ver = str(major.value) + "." + str(minor.value) + "." + str(revision.value)
        return sdk_ver

    def query_capability_(self, handle: int, capability: int) -> int:
        """
        A more convenient version of :meth:`query_capability` method
        (``AlazarQueryCapability``).

        This method hides the fact that the output values in the original
        function are written to the provided pointers.

        Args:
            handle: Handle of the board of interest
            capability: An integer identifier of a capability parameter
                (:class:`.constants.Capability` enumeration encapsulates
                the available identifiers)

        Returns:
            Value of the requested capability
        """
        value = ctypes.c_uint32(0)
        reserved = 0
        self.query_capability(handle, capability, reserved, ctypes.byref(value))
        return value.value

    def read_register_(self, handle: int, offset: int) -> int:
        """
        Read a value from a given offset in the Alazar card's register.

        A more convenient version of :meth:`read_register` method
        (``AlazarReadRegister``).

        Args:
            handle: Handle of the board of interest
            offset: Offset into the memory to read from

        Returns:
            The value read as an integer
        """
        output = ctypes.c_uint32(0)
        self.read_register(
            handle, offset, ctypes.byref(output), REGISTER_ACCESS_PASSWORD
        )
        return output.value

    def write_register_(self, handle: int, offset: int, value: int) -> None:
        """
        Write a value to a given offset in the Alazar card's register.

        A more convenient version of :meth:`write_register` method
        (``AlazarWriteRegister``).

        Args:
            handle: Handle of the board of interest
            offset: The offset in memory to write to
            value: The value to write
        """
        self.write_register(handle, offset, value, REGISTER_ACCESS_PASSWORD)
