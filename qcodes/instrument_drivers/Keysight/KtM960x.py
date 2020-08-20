from qcodes import Instrument, validators as vals
import ctypes
from functools import partial
from .KtM960x_Defs import *
from typing import (Dict, Optional)


class KtM960x(Instrument):
    _default_buf_size = 256
    _dll_loc = r"C:\Program Files\IVI Foundation\IVI\Bin\KtM960x_64.dll"

    def __init__(self, name: str, address: str, options: bytes = b"", **kwargs) -> None:
        super().__init__(name, **kwargs)

        if not isinstance(address, bytes):
            address = bytes(address, "ascii")

        self._address = address
        self._session = ctypes.c_int(0)
        self._dll = ctypes.windll.LoadLibrary(self._dll_loc)

        self.add_parameter('output',
                           label="Source Output Enable",
                           get_cmd=partial(self.get_vi_bool,
                                           KTM960X_ATTR_OUTPUT_ENABLED),
                           set_cmd=partial(self.set_vi_bool,
                                           KTM960X_ATTR_OUTPUT_ENABLED),
                           val_mapping={'on': True, 'off': False})

        self.add_parameter('voltage_level',
                           label="Source Voltage Level",
                           unit="Volt",
                           get_cmd=partial(self.get_vi_real64,
                                           KTM960X_ATTR_OUTPUT_VOLTAGE_LEVEL),
                           set_cmd=partial(self.set_vi_real64,
                                           KTM960X_ATTR_OUTPUT_VOLTAGE_LEVEL),
                           vals=vals.Numbers(-210, 210))

        self.get_driver_desc = partial(
            self.get_vi_string, KTM960X_ATTR_SPECIFIC_DRIVER_DESCRIPTION)
        self.get_driver_prefix = partial(
            self.get_vi_string, KTM960X_ATTR_SPECIFIC_DRIVER_PREFIX)
        self.get_driver_revision = partial(
            self.get_vi_string, KTM960X_ATTR_SPECIFIC_DRIVER_REVISION)
        self.get_firmware_revision = partial(
            self.get_vi_string, KTM960X_ATTR_INSTRUMENT_FIRMWARE_REVISION)
        self.get_model = partial(
            self.get_vi_string, KTM960X_ATTR_INSTRUMENT_MODEL)
        self.get_serial_number = partial(
            self.get_vi_string, KTM960X_ATTR_MODULE_SERIAL_NUMBER)
        self.get_manufactorer = partial(
            self.get_vi_string, KTM960X_ATTR_INSTRUMENT_MANUFACTURER)

        self._connect(options)

        self.connect_message()

    def _connect(self, options):
        if not isinstance(options, bytes):
            options = bytes(options, "ascii")
        status = self._dll.KtM960x_InitWithOptions(self._address, 1, 1,
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

    def get_vi_string(self, attr: int) -> str:
        s = ctypes.create_string_buffer(self._default_buf_size)
        status = self._dll.KtM960x_GetAttributeViString(self._session, b"",
                                                        attr, self._default_buf_size, s)
        if status:
            print("Driver error! ", status)
            return None
        else:
            return s.value.decode('utf-8')

    def get_vi_bool(self, attr: int) -> bool:
        s = ctypes.c_uint16(0)
        status = self._dll.KtM960x_GetAttributeViBoolean(self._session, b"",
                                                         attr, ctypes.byref(s))
        if status:
            print("Driver error! ", status)
            return None
        else:
            return True if s else False

    def set_vi_bool(self, attr: int, value: bool) -> bool:
        v = ctypes.c_uint16(1) if value else ctypes.c_uint16(0)
        status = self._dll.KtM960x_SetAttributeViBoolean(self._session, b"",
                                                         attr, v)
        if status:
            print("Driver error! ", status)
            return False
        else:
            return True

    def get_vi_real64(self, attr: int) -> float:
        s = ctypes.c_double(0)
        status = self._dll.KtM960x_GetAttributeViReal64(self._session, b"",
                                                         attr, ctypes.byref(s))

        if status:
            print("Driver error! ", status)
            return None
        else:
            return float(s.value)

    def set_vi_real64(self, attr: int, value: float) -> bool:
        v = ctypes.c_double(value)
        status = self._dll.KtM960x_SetAttributeViReal64(self._session, b"",
                                                         attr, v)
        if status:
            print("Driver error! ", status)
            return False
        else:
            return True

    # def get_driver_description(self) -> str:
    #     s = ctypes.create_string_buffer(self._default_buf_size)
    #     status = self._dll.KtM960x_GetAttributeViString(self._session, b"",
    #     KTM960X_ATTR_SPECIFIC_DRIVER_DESCRIPTION, self._default_buf_size, s)
    #     if status:
    #         _err(status)
    #         return None
    #     else:
    #         return s.value.decode('utf-8')

    def close(self):
        self._dll.KtM960x_close(self._session)
        super().close()
