from .KtMAwg_Defs import *
from qcodes import Instrument, validators as vals
import ctypes
from functools import partial
from typing import (Dict, Optional)


class KtMAwg(Instrument):
    _default_buf_size = 256
    _dll_loc = r"C:\Program Files\IVI Foundation\IVI\Bin\KtMAwg_64.dll"

    def __init__(self, name: str, address: str, options: bytes = b"", **kwargs) -> None:
        super().__init__(name, **kwargs)

        if not isinstance(address, bytes):
            address = bytes(address, "ascii")

        self._address = address
        self._session = ctypes.c_int(0)
        self._dll = ctypes.windll.LoadLibrary(self._dll_loc)


        self.get_driver_desc = partial(
            self.get_vi_string, KTMAWG_ATTR_SPECIFIC_DRIVER_DESCRIPTION)
        self.get_driver_prefix = partial(
            self.get_vi_string, KTMAWG_ATTR_SPECIFIC_DRIVER_PREFIX)
        self.get_driver_revision = partial(
            self.get_vi_string, KTMAWG_ATTR_SPECIFIC_DRIVER_REVISION)
        self.get_firmware_revision = partial(
            self.get_vi_string, KTMAWG_ATTR_INSTRUMENT_FIRMWARE_REVISION)
        self.get_model = partial(
            self.get_vi_string, KTMAWG_ATTR_INSTRUMENT_MODEL)
        self.get_serial_number = partial(
            self.get_vi_string, KTMAWG_ATTR_MODULE_SERIAL_NUMBER)
        self.get_manufactorer = partial(
            self.get_vi_string, KTMAWG_ATTR_INSTRUMENT_MANUFACTURER)

        self._connect(options)

        self.connect_message()

    def _connect(self, options):
        if not isinstance(options, bytes):
            options = bytes(options, "ascii")
        status = self._dll.KtMAwg_InitWithOptions(self._address, 1, 1,
                                                   options, ctypes.byref(self._session))
        if status:
            print("connection to device failed! error: ", status)
            raise SystemError

    def get_idn(self) -> Dict[str, Optional[str]]:
        """generates the ``*IDN`` dictionary for qcodes"""

        id_dict: Dict[str, Optional[str]] = {'firmware': self.get_firmware_revision(),
                                             'model': self.get_model(),
                                             'serial': self.get_serial_number(),
                                             'vendor': self.get_manufactorer()}
        return id_dict

    # Query the driver for errors
    def _get_errors(self):
        error_code = ctypes.c_int(-1)
        error_message = ctypes.create_string_buffer(256)
        while error_code.value != 0:
            status = self._dll.KTMAWG_error_query(
                self._session, ctypes.byref(error_code), error_message)
            assert(status == 0)
            print(
                f"error_query: {error_code.value}, {error_message.value.decode('utf-8')}")

    # Generic functions for reading/writing different attributes
    def get_vi_string(self, attr: int) -> str:
        s = ctypes.create_string_buffer(self._default_buf_size)
        status = self._dll.KtMAwg_GetAttributeViString(self._session, b"",
                                                        attr, self._default_buf_size, s)
        if status:
            print("Driver error! ", status)
            raise ValueError
        return s.value.decode('utf-8')

    def get_vi_bool(self, attr: int) -> bool:
        s = ctypes.c_uint16(0)
        status = self._dll.KtMAwg_GetAttributeViBoolean(self._session, b"",
                                                         attr, ctypes.byref(s))
        if status:
            print("Driver error! ", status)
            raise ValueError
        return True if s else False

    def set_vi_bool(self, attr: int, value: bool) -> bool:
        v = ctypes.c_uint16(1) if value else ctypes.c_uint16(0)
        status = self._dll.KtMAwg_SetAttributeViBoolean(self._session, b"",
                                                         attr, v)
        if status:
            print("Driver error! ", status)
            raise ValueError
        return True

    def get_vi_real64(self, attr: int) -> float:
        s = ctypes.c_double(0)
        status = self._dll.KtMAwg_GetAttributeViReal64(self._session, b"",
                                                        attr, ctypes.byref(s))

        if status:
            print("Driver error! ", status)
            raise ValueError
        return float(s.value)

    def set_vi_real64(self, attr: int, value: float) -> bool:
        v = ctypes.c_double(value)
        status = self._dll.KtMAwg_SetAttributeViReal64(self._session, b"",
                                                        attr, v)
        if status:
            print("Driver error! ", status)
            raise ValueError
        return True

    def set_vi_int(self, attr: int, value: int) -> bool:
        v = ctypes.c_int32(value)
        status = self._dll.KtMAwg_SetAttributeViInt32(self._session, b"",
                                                       attr, v)
        if status:
            print("Driver error! ", status)
            raise ValueError
        return True

    def get_vi_int(self, attr: int) -> int:
        v = ctypes.c_int32(0)
        status = self._dll.KtMAwg_GetAttributeViInt32(self._session, b"",
                                                       attr, ctypes.byref(v))
        if status:
            print("Driver error! ", status)
            raise ValueError
        return int(v.value)

    def close(self):
        self._dll.KtMAwg_close(self._session)
        super().close()
