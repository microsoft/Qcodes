import ctypes
from functools import partial
from typing import TYPE_CHECKING

import qcodes.validators as vals
from qcodes.instrument import Instrument, InstrumentBaseKWArgs
from qcodes.parameters import (
    MultiParameter,
    Parameter,
    ParamRawDataType,
    create_on_off_val_mapping,
)

from .KtM960xDefs import *  # noqa F403

if TYPE_CHECKING:
    from typing_extensions import Unpack


class Measure(MultiParameter):
    def __init__(self, name: str, instrument: "KeysightM960x") -> None:
        super().__init__(
            name=name,
            names=("voltage", "current", "resistance", "status", "timestamp", "source"),
            shapes=((), (), (), (), (), ()),
            units=("V", "A", "Ohm", "", "", ""),
            instrument=instrument,
            labels="Measurement Data",
            docstring="param that returns measurement values",
        )
        self.instrument: KeysightM960x

    def get_raw(self) -> tuple[ParamRawDataType, ...]:
        return self.instrument._measure()


class KeysightM960x(Instrument):
    """
    Provide a wrapper for the Keysight KtM960x DAC. This driver provides an
    interface into the IVI-C driver provided by Keysight. The .dll is installed
    by default into C:\\Program Files\\IVI Foundation\\IVI\\Bin\\KtM960x_64.dll
    but a different path can be supplied to the constructor
    """

    _default_buf_size = 256

    def __init__(
        self,
        name: str,
        address: str,
        options: str = "",
        dll_path: str = r"C:\Program Files\IVI Foundation\IVI\Bin\KtM960x_64.dll",
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        super().__init__(name, **kwargs)

        self._address = bytes(address, "ascii")
        self._options = bytes(options, "ascii")
        self._session = ctypes.c_int(0)
        self._dll_loc = dll_path
        self._dll = ctypes.cdll.LoadLibrary(self._dll_loc)

        self.output: Parameter = self.add_parameter(
            "output",
            label="Source Output Enable",
            get_cmd=partial(self._get_vi_bool, KTM960X_ATTR_OUTPUT_ENABLED),
            set_cmd=partial(self._set_vi_bool, KTM960X_ATTR_OUTPUT_ENABLED),
            val_mapping=create_on_off_val_mapping(on_val=True, off_val=False),
        )
        """Parameter output"""

        self.voltage_level: Parameter = self.add_parameter(
            "voltage_level",
            label="Source Voltage Level",
            unit="Volt",
            get_cmd=partial(self._get_vi_real64, KTM960X_ATTR_OUTPUT_VOLTAGE_LEVEL),
            set_cmd=partial(self._set_vi_real64, KTM960X_ATTR_OUTPUT_VOLTAGE_LEVEL),
            vals=vals.Numbers(-210, 210),
        )
        """Parameter voltage_level"""

        self.current_range: Parameter = self.add_parameter(
            "current_range",
            label="Output Current Range",
            unit="Amp",
            vals=vals.Numbers(1e-9, 300e-3),
            get_cmd=partial(self._get_vi_real64, KTM960X_ATTR_OUTPUT_CURRENT_RANGE),
            set_cmd=partial(self._set_vi_real64, KTM960X_ATTR_OUTPUT_CURRENT_RANGE),
        )
        """Parameter current_range"""

        self.measure_current_range: Parameter = self.add_parameter(
            "measure_current_range",
            label="Current Measurement Range",
            unit="Amp",
            get_cmd=partial(
                self._get_vi_real64, KTM960X_ATTR_MEASUREMENT_CURRENT_RANGE
            ),
            set_cmd=partial(
                self._set_vi_real64, KTM960X_ATTR_MEASUREMENT_CURRENT_RANGE
            ),
            vals=vals.Numbers(1e-9, 300e-3),
        )
        """Parameter measure_current_range"""

        self.measure_current_time: Parameter = self.add_parameter(
            "measure_current_time",
            label="Current Measurement Integration Time",
            unit="Seconds",
            get_cmd=partial(
                self._get_vi_real64, KTM960X_ATTR_MEASUREMENT_CURRENT_APERTURE
            ),
            set_cmd=partial(
                self._set_vi_real64, KTM960X_ATTR_MEASUREMENT_CURRENT_APERTURE
            ),
            vals=vals.Numbers(800e-9, 2),
        )
        """Parameter measure_current_time"""

        self.measure_data: Measure = self.add_parameter(
            "measure_data", parameter_class=Measure
        )
        """Parameter measure_data"""

        self._get_driver_desc = partial(
            self._get_vi_string, KTM960X_ATTR_SPECIFIC_DRIVER_DESCRIPTION
        )
        self._get_driver_prefix = partial(
            self._get_vi_string, KTM960X_ATTR_SPECIFIC_DRIVER_PREFIX
        )
        self._get_driver_revision = partial(
            self._get_vi_string, KTM960X_ATTR_SPECIFIC_DRIVER_REVISION
        )
        self._get_firmware_revision = partial(
            self._get_vi_string, KTM960X_ATTR_INSTRUMENT_FIRMWARE_REVISION
        )
        self._get_model = partial(self._get_vi_string, KTM960X_ATTR_INSTRUMENT_MODEL)
        self._get_serial_number = partial(
            self._get_vi_string, KTM960X_ATTR_MODULE_SERIAL_NUMBER
        )
        self._get_manufacturer = partial(
            self._get_vi_string, KTM960X_ATTR_INSTRUMENT_MANUFACTURER
        )

        self._connect()

        self.connect_message()

    def _connect(self) -> None:
        status = self._dll.KtM960x_InitWithOptions(
            self._address, 1, 1, self._options, ctypes.byref(self._session)
        )
        if status:
            raise SystemError(f"connection to device failed! error: {status}")

    def get_idn(self) -> dict[str, str | None]:
        """generates the ``*IDN`` dictionary for qcodes"""

        id_dict: dict[str, str | None] = {
            "firmware": self._get_firmware_revision(),
            "model": self._get_model(),
            "serial": self._get_serial_number(),
            "vendor": self._get_manufacturer(),
            "driver desc": self._get_driver_desc(),
            "driver prefix": self._get_driver_prefix(),
            "driver revision": self._get_driver_revision(),
        }
        return id_dict

    def _measure(self) -> tuple[ParamRawDataType, ...]:
        # Setup the output
        self._set_vi_int(
            KTM960X_ATTR_OUTPUT_PRIORITY_MODE, KTM960X_VAL_PRIORITY_MODE_VOLTAGE
        )
        self._set_vi_int(
            KTM960X_ATTR_OUTPUT_OPERATION_MODE,
            KTM960X_VAL_OUTPUT_OPERATION_MODE_STANDARD,
        )
        self._set_vi_int(
            KTM960X_ATTR_MEASUREMENT_ACQUISITION_MODE,
            KTM960X_VAL_ACQUISITION_MODE_NORMAL,
        )

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
            ctypes.byref(actual_size),
        )

        if status:
            raise ValueError(f"Driver error: {status}")
        # This might be a bit slow?
        # Returned as [voltage, current, resistance, status,
        #                                               timestamp, and source]
        v = list(val_buf)[0 : actual_size.value]

        # 'voltage': v[0], 'current': v[1], 'resistance': v[2], 'status': v[3],
        # 'timestamp': v[4], 'source': v[5]
        return v[0], v[1], v[2], v[3], v[4], v[5]

    # Query the driver for errors
    def get_errors(self) -> dict[int, str]:
        error_code = ctypes.c_int(-1)
        error_message = ctypes.create_string_buffer(256)
        error_dict = dict()
        while error_code.value != 0:
            status = self._dll.KtM960x_error_query(
                self._session, ctypes.byref(error_code), error_message
            )
            assert status == 0
            error_dict[error_code.value] = error_message.value.decode("utf-8")

        return error_dict

    # Generic functions for reading/writing different attributes
    def _get_vi_string(self, attr: int) -> str:
        s = ctypes.create_string_buffer(self._default_buf_size)
        status = self._dll.KtM960x_GetAttributeViString(
            self._session, b"", attr, self._default_buf_size, s
        )
        if status:
            raise ValueError(f"Driver error: {status}")
        return s.value.decode("utf-8")

    def _get_vi_bool(self, attr: int) -> bool:
        s = ctypes.c_uint16(0)
        status = self._dll.KtM960x_GetAttributeViBoolean(
            self._session, b"", attr, ctypes.byref(s)
        )
        if status:
            raise ValueError(f"Driver error: {status}")
        return bool(s)

    def _set_vi_bool(self, attr: int, value: bool) -> None:
        v = ctypes.c_uint16(1) if value else ctypes.c_uint16(0)
        status = self._dll.KtM960x_SetAttributeViBoolean(self._session, b"", attr, v)
        if status:
            raise ValueError(f"Driver error: {status}")

    def _get_vi_real64(self, attr: int) -> float:
        s = ctypes.c_double(0)
        status = self._dll.KtM960x_GetAttributeViReal64(
            self._session, b"", attr, ctypes.byref(s)
        )

        if status:
            raise ValueError(f"Driver error: {status}")
        return float(s.value)

    def _set_vi_real64(self, attr: int, value: float) -> None:
        v = ctypes.c_double(value)
        status = self._dll.KtM960x_SetAttributeViReal64(self._session, b"", attr, v)
        if status:
            raise ValueError(f"Driver error: {status}")

    def _set_vi_int(self, attr: int, value: int) -> None:
        v = ctypes.c_int32(value)
        status = self._dll.KtM960x_SetAttributeViInt32(self._session, b"", attr, v)
        if status:
            raise ValueError(f"Driver error: {status}")

    def _get_vi_int(self, attr: int) -> int:
        v = ctypes.c_int32(0)
        status = self._dll.KtM960x_GetAttributeViInt32(
            self._session, b"", attr, ctypes.byref(v)
        )
        if status:
            raise ValueError(f"Driver error: {status}")
        return int(v.value)

    def close(self) -> None:
        self._dll.KtM960x_close(self._session)
        super().close()


KtM960x = KeysightM960x
"Alias for backwards compatibility"
