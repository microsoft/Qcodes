from typing import Any, Optional

from pyvisa.constants import Parity, StopBits
from pyvisa.resources.serial import SerialInstrument
from pyvisa.resources.tcpip import TCPIPSocket

import qcodes.validators as vals
from qcodes.instrument.visa import VisaInstrument
from qcodes.parameters import ParameterBase
from qcodes.parameters.val_mapping import create_on_off_val_mapping


class SR570(VisaInstrument):
    """
    QCoDeS driver for the Stanford Research Systems SR570 Voltage-preamplifier.

    This is a real driver and it will talk to your instrument.

    It can't listen to it, so make sure that you either reset it or set parameters before reading them.
    (Resetting the device will update the parameters with their reset value.)
    """

    def __init__(self, name: str, address: str, **kwargs: Any):
        super().__init__(name, address, **kwargs)
        if isinstance(self.visa_handle, TCPIPSocket):
            pass  # allow connection to remote serial device
        else:
            assert isinstance(self.visa_handle, SerialInstrument)
            serial: SerialInstrument = self.visa_handle
            # 9600 Baud DCE, 8 bit, no parity, 2 stop bits
            serial.baud_rate = 9600
            serial.parity = Parity.none
            serial.stop_bits = StopBits.two
            serial.write_termination = "\r\n"
            serial.read_termination = ""  # but there's no read

        self.connect_message()

        # fmt:off
        sensitivity = [
            1e-12,   2e-12,   5e-12,
            10e-12,  20e-12,  50e-12,
            100e-12, 200e-12, 500e-12,
            1e-9,    2e-9,    5e-9,
            10e-9,   20e-9,   50e-9,
            100e-9,  200e-9,  500e-9,
            1e-6,    2e-6,    5e-6,
            10e-6,   20e-6,   50e-6,
            100e-6,  200e-6,  500e-6,
            1e-3
        ]

        input_offset_current = [
            1e-12,   2e-12,   5e-12,
            10e-12,  20e-12,  50e-12,
            100e-12, 200e-12, 500e-12,
            1e-9,    2e-9,    5e-9,
            10e-9,   20e-9,   50e-9,
            100e-9,  200e-9,  500e-9,
            1e-6,    2e-6,    5e-6,
            10e-6,   20e-6,   50e-6,
            100e-6,  200e-6,  500e-6,
            1e-3,    2e-3,    5e-3
        ]

        filter_type = [
            "6db_highpass",
            "12db_highpass",
            "6db_bandpass",
            "6db_lowpass",
            "12db_lowpass",
            "none",
        ]

        filter_freq = [
                   0.03,
            0.1,    0.3,
              1,      3,
              10,     30,
              100,    300,
              1e+3,   3e+3,
              10e+3,  30e+3,
              100e+3, 300e+3,
              1e+6,
        ]

        gain_modes = [
            "low_noise",
            "high_bandwidth",
            "low_drift"
        ]

        # fmt:on
        on_off_val_mapping = create_on_off_val_mapping(on_val=1, off_val=0)

        self._reset_defaults: dict[str, Any] = {}

        self.add_parameter(
            "sensitivity",
            get_cmd=None,
            set_cmd="SENS {}",
            val_mapping={scale: n for n, scale in enumerate(sensitivity)},
            unit="A/V",
        )

        self._reset_defaults["sensitivity"] = 1e-6

        self.add_parameter(
            "sensitivity_uncalibrated_mode",
            get_cmd=None,
            set_cmd="SUCM {}",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
        )

        self._reset_defaults["sensitivity_uncalibrated_mode"] = False

        self.add_parameter(
            "sensitivity_uncalibrated_vernier",
            get_cmd=None,
            set_cmd="SUCV {}",
            vals=vals.Ints(0, 100),
            unit="%",
        )

        self.add_parameter(
            "input_offset_current_status",
            get_cmd=None,
            set_cmd="IOON {}",
            val_mapping=on_off_val_mapping,
        )

        self._reset_defaults["input_offset_current_status"] = False

        self.add_parameter(
            "input_offset_current_level",
            get_cmd=None,
            set_cmd="IOLV {}",
            val_mapping={scale: n for n, scale in enumerate(input_offset_current)},
            unit="A",
        )

        self._reset_defaults["input_offset_current_level"] = 1e-12

        self.add_parameter(
            "input_offset_current_sign",
            get_cmd=None,
            set_cmd="IOSN {}",
            val_mapping={+1: 0, -1: 1},
        )

        self._reset_defaults["input_offset_current_sign"] = +1

        self.add_parameter(
            "input_offset_uncalibrated_mode",
            get_cmd=None,
            set_cmd="IOUC {}",
            val_mapping=on_off_val_mapping,
        )

        self._reset_defaults["input_offset_uncalibrated_mode"] = False

        self.add_parameter(
            "input_offset_uncalibrated_vernier",
            get_cmd=None,
            set_cmd="IOUV {}",
            vals=vals.MultiTypeAnd(
                vals.Numbers(-100, 100), vals.PermissiveMultiples(0.1)
            ),
            set_parser=lambda v: int(v * 10),
            unit="%",
        )

        self.add_parameter(
            "bias_voltage_status",
            get_cmd=None,
            set_cmd="BSON {}",
            val_mapping=on_off_val_mapping,
        )

        self._reset_defaults["bias_voltage_status"] = False

        self.add_parameter(
            "bias_voltage",
            get_cmd=None,
            set_cmd="BSLV {}",
            vals=vals.Numbers(-5.0, +5.0),
            set_parser=lambda v: int(v * 1000),
            unit="V",
        )

        self._reset_defaults["bias_voltage"] = 0.0

        self.add_parameter(
            "filter_type",
            get_cmd=None,
            set_cmd="FLTT {}",
            val_mapping={fltt: n for n, fltt in enumerate(filter_type)},
        )

        self._reset_defaults["filter_type"] = "none"

        self.add_parameter(
            "filter_lowpass_frequency",
            get_cmd=None,
            set_cmd="LFRQ {}",
            val_mapping={f: n for n, f in enumerate(filter_freq)},
            unit="Hz",
        )

        self._reset_defaults["filter_lowpass_frequency"] = 1e6

        self.add_parameter(
            "filter_highpass_frequency",
            get_cmd=None,
            set_cmd="HFRQ {}",
            val_mapping={f: n for n, f in enumerate(filter_freq[:12])},
            unit="Hz",
        )

        self._reset_defaults["filter_highpass_frequency"] = 0.03

        self.add_function("reset_overload_condition", call_cmd="ROLD")

        self.add_parameter(
            "gain_mode",
            get_cmd=None,
            set_cmd="GNMD {}",
            val_mapping={mode: n for n, mode in enumerate(gain_modes)},
        )

        self._reset_defaults["gain_mode"] = gain_modes[0]

        self.add_parameter(
            "invert", get_cmd=None, set_cmd="INVT {}", val_mapping=on_off_val_mapping
        )

        self._reset_defaults["invert"] = False

        self.add_parameter(
            "blank", get_cmd=None, set_cmd="BLNK {}", val_mapping=on_off_val_mapping
        )

        self._reset_defaults["blank"] = False

    def reset(self) -> None:
        self.write("*RST")
        p: ParameterBase
        for p in self.parameters.values():
            if not hasattr(p, "cache"):
                continue
            p.cache.invalidate()

        for name, value in self._reset_defaults.items():
            p = self.parameters[name]
            if not hasattr(p, "cache"):
                continue
            p.cache.set(value)

    def get_idn(self) -> dict[str, Optional[str]]:
        vendor = "Stanford Research Systems"
        model = "SR570"
        serial = None
        firmware = None

        return {
            "vendor": vendor,
            "model": model,
            "serial": serial,
            "firmware": firmware,
        }
