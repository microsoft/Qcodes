from .KtM960xDefs import *

import ctypes
from functools import partial
from typing import (Dict, Optional, Any)

from qcodes import Instrument, validators as vals
from qcodes.utils.helpers import create_on_off_val_mapping


class KtM960x(Instrument):
    """
    Provide a wrapper for the Keysight KtM960x DAC. This driver provides an
    interface into the IVI-C driver provided by Keysight. The .dll is installed
    by default into C:\\Program Files\\IVI Foundation\\IVI\\Bin\\KtM960x_64.dll
    but a different path can be supplied to the constructor
    """

    _default_buf_size = 256

    def __init__(self,
                 name: str,
                 address: str,
                 options: bytes = b"",
                 dll_path: str = r"C:\Program Files\IVI "
                                 r"Foundation\IVI\Bin\KtM960x_64.dll",
                 **kwargs: Any) -> None:
        super().__init__(name, **kwargs)

        if not isinstance(address, bytes):
            address = bytes(address, "ascii")

        self._address = address
        self._session = ctypes.c_int(0)
        self._dll_loc = dll_path
        self._dll = ctypes.windll.LoadLibrary(self._dll_loc)

        self.add_parameter('output',
                           label="Source Output Enable",
                           get_cmd=partial(self.get_vi_bool,
                                           KTM960X_ATTR_OUTPUT_ENABLED),
                           set_cmd=partial(self.set_vi_bool,
                                           KTM960X_ATTR_OUTPUT_ENABLED),
                           val_mapping=create_on_off_val_mapping(on_val=True,
                                                                 off_val=False))

        self.add_parameter('voltage_level',
                           label="Source Voltage Level",
                           unit="Volt",
                           get_cmd=partial(self.get_vi_real64,
                                           KTM960X_ATTR_OUTPUT_VOLTAGE_LEVEL),
                           set_cmd=partial(self.set_vi_real64,
                                           KTM960X_ATTR_OUTPUT_VOLTAGE_LEVEL),
                           vals=vals.Numbers(-210, 210))

        self.add_parameter("current_range",
                           label="Output Current Range",
                           unit="Amp",
                           vals=vals.Numbers(1e-9, 300e-3),
                           get_cmd=partial(self.get_vi_real64,
                                           KTM960X_ATTR_OUTPUT_CURRENT_RANGE),
                           set_cmd=partial(self.set_vi_real64,
                                           KTM960X_ATTR_OUTPUT_CURRENT_RANGE)
                           )

        self.add_parameter("measure_current_range",
                           label="Current Measurement Range",
                           unit="Amp",
                           get_cmd=partial(
                               self.get_vi_real64,
                               KTM960X_ATTR_MEASUREMENT_CURRENT_RANGE),
                           set_cmd=partial(
                               self.set_vi_real64,
                               KTM960X_ATTR_MEASUREMENT_CURRENT_RANGE)
                           )

        self.add_parameter("measure_current_time",
                           label="Current Measurement Integration Time",
                           unit="Seconds",
                           get_cmd=partial(
                               self.get_vi_real64,
                               KTM960X_ATTR_MEASUREMENT_CURRENT_APERTURE),
                           set_cmd=partial(
                               self.set_vi_real64,
                               KTM960X_ATTR_MEASUREMENT_CURRENT_APERTURE)
                           )

        self.add_parameter("measure_current",
                           label="Measured Current",
                           unit="Amp",
                           get_cmd=partial(self._measure, key="current"),
                           set_cmd=None)

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

    def _connect(self, options) -> None:
        if not isinstance(options, bytes):
            options = bytes(options, "ascii")
        status = self._dll.KtM960x_InitWithOptions(self._address,
                                                   1,
                                                   1,
                                                   options,
                                                   ctypes.byref(self._session))
        if status:
            raise SystemError(f"connection to device failed! error: {status}")

    def get_idn(self) -> Dict[str, Optional[str]]:
        """generates the ``*IDN`` dictionary for qcodes"""

        id_dict: Dict[str, Optional[str]] = {
            'firmware': self.get_firmware_revision(),
            'model': self.get_model(),
            'serial': self.get_serial_number(),
            'vendor': self.get_manufactorer()
        }
        return id_dict

    def _measure(self, key: str) -> float:

        # Setup the output
        self.set_vi_int(KTM960X_ATTR_OUTPUT_PRIORITY_MODE,
                        KTM960X_VAL_PRIORITY_MODE_VOLTAGE)
        self.set_vi_int(KTM960X_ATTR_OUTPUT_OPERATION_MODE,
                        KTM960X_VAL_OUTPUT_OPERATION_MODE_STANDARD)
        self.set_vi_int(KTM960X_ATTR_MEASUREMENT_ACQUISITION_MODE,
                        KTM960X_VAL_ACQUISITION_MODE_NORMAL)

        ch_num_buf = (ctypes.c_int32 * 1)()
        val_buf = (ctypes.c_double * 1024)()
        actual_size = ctypes.c_int32(0)
        ch_num_buf[0] = 1
        status = self._dll.KtM960x_MeasurementMeasure(
            self._session,
            KTM960X_VAL_MEASUREMENT_TYPE_ALL,
            1,
            ch_num_buf,
            1024,
            val_buf,
            ctypes.byref(actual_size)
        )

        if status:
            raise ValueError(f"Driver error: {status}")
        # This might be a bit slow?
        # Returned as [voltage, current, resistance, status,
        #                                               timestamp, and source]
        v = list(val_buf)[0:actual_size.value]
        val_map = {'voltage': v[0],
                   'current': v[1],
                   'resistance': v[2],
                   'status': v[3],
                   'timestamp': v[4],
                   'source': v[5]}

        return val_map[key]

    # Query the driver for errors
    def get_errors(self):
        error_code = ctypes.c_int(-1)
        error_message = ctypes.create_string_buffer(256)
        while error_code.value != 0:
            status = self._dll.KtM960x_error_query(
                self._session, ctypes.byref(error_code), error_message)
            assert(status == 0)
            print(
                f"error_query: {error_code.value}, "
                f"{error_message.value.decode('utf-8')}")

    # Generic functions for reading/writing different attributes
    def get_vi_string(self, attr: int) -> str:
        s = ctypes.create_string_buffer(self._default_buf_size)
        status = self._dll.KtM960x_GetAttributeViString(self._session,
                                                        b"",
                                                        attr,
                                                        self._default_buf_size,
                                                        s)
        if status:
            raise ValueError(f"Driver error: {status}")
        return s.value.decode('utf-8')

    def get_vi_bool(self, attr: int) -> bool:
        s = ctypes.c_uint16(0)
        status = self._dll.KtM960x_GetAttributeViBoolean(self._session, b"",
                                                         attr, ctypes.byref(s))
        if status:
            raise ValueError(f"Driver error: {status}")
        return bool(s)

    def set_vi_bool(self, attr: int, value: bool) -> bool:
        v = ctypes.c_uint16(1) if value else ctypes.c_uint16(0)
        status = self._dll.KtM960x_SetAttributeViBoolean(self._session, b"",
                                                         attr, v)
        if status:
            raise ValueError(f"Driver error: {status}")
        return True

    def get_vi_real64(self, attr: int) -> float:
        s = ctypes.c_double(0)
        status = self._dll.KtM960x_GetAttributeViReal64(self._session, b"",
                                                        attr, ctypes.byref(s))

        if status:
            raise ValueError(f"Driver error: {status}")
        return float(s.value)

    def set_vi_real64(self, attr: int, value: float) -> bool:
        v = ctypes.c_double(value)
        status = self._dll.KtM960x_SetAttributeViReal64(self._session, b"",
                                                        attr, v)
        if status:
            raise ValueError(f"Driver error: {status}")
        return True

    def set_vi_int(self, attr: int, value: int) -> bool:
        v = ctypes.c_int32(value)
        status = self._dll.KtM960x_SetAttributeViInt32(self._session, b"",
                                                       attr, v)
        if status:
            raise ValueError(f"Driver error: {status}")
        return True

    def get_vi_int(self, attr: int) -> int:
        v = ctypes.c_int32(0)
        status = self._dll.KtM960x_GetAttributeViInt32(self._session, b"",
                                                       attr, ctypes.byref(v))
        if status:
            raise ValueError(f"Driver error: {status}")
        return int(v.value)

    def close(self) -> None:
        self._dll.KtM960x_close(self._session)
        super().close()
