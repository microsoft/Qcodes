"""Tests for the SignalHound USB SA124B driver.

The driver requires the vendor DLL and a connected device, so the driver itself
is tested with a minimal fake of the DLL, which implements the functions that
the driver calls, based on how the driver uses them. The parameter classes guard
against being attached to a different instrument and that guard can be
exercised with any instrument.
"""

import ctypes
import logging
from typing import TYPE_CHECKING, Any
from unittest import mock

import numpy as np
import pytest

from qcodes.instrument import Instrument
from qcodes.instrument_drivers.mock_instruments import DummyInstrument
from qcodes.instrument_drivers.signal_hound.SignalHound_USB_SA124B import (
    Constants,
    FrequencySweep,
    ScaleParameter,
    SignalHoundUSBSA124B,
    SweepTraceParameter,
    TraceParameter,
    saStatus,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

_DEVICE_HANDLE = 7
_SWEEP_LENGTH = 5


class _DllFunction:
    """
    A function of the fake DLL, which returns the status configured on the
    fake for it, and on which the driver can set ctypes argtypes.
    """

    def __init__(
        self, dll: "_FakeSignalHoundDll", name: str, function: "Callable[..., None]"
    ) -> None:
        self._dll = dll
        self._name = name
        self._function = function
        self.argtypes: list[Any] | None = None

    def __call__(self, *args: Any) -> int:
        self._dll.calls.append(self._name)
        self._function(*args)
        return self._dll.statuses.get(self._name, saStatus.saNoError)


class _FakeSignalHoundDll:
    """
    A fake of the sa_api DLL. Output arguments are passed as ctypes pointers or
    arrays, which the fake writes to, like the DLL does.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        # statuses to return from functions other than saNoError
        self.statuses: dict[str, int] = {}
        self.closed_handles: list[int] = []
        self._center = 0.0
        self._span = 0.0
        no_op = lambda *args: None  # noqa: E731
        functions: dict[str, Callable[..., None]] = {
            "saOpenDevice": self._open_device,
            "saCloseDevice": self._close_device,
            "saGetDeviceType": self._get_device_type,
            "saGetSerialNumber": self._get_serial_number,
            "saGetFirmwareString": self._get_firmware_string,
            "saConfigCenterSpan": self._config_center_span,
            "saQuerySweepInfo": self._query_sweep_info,
            "saGetSweep_32f": self._get_sweep,
            "saConfigAcquisition": no_op,
            "saConfigLevel": no_op,
            "saSetTimebase": no_op,
            "saConfigSweepCoupling": no_op,
            "saInitiate": no_op,
            "saAbort": no_op,
            "saPreset": no_op,
        }
        for name, function in functions.items():
            setattr(self, name, _DllFunction(self, name, function))

    def _open_device(self, handle: Any) -> None:
        handle.contents.value = _DEVICE_HANDLE

    def _close_device(self, handle: ctypes.c_int) -> None:
        self.closed_handles.append(handle.value)

    def _get_device_type(self, handle: ctypes.c_int, device_type: Any) -> None:
        device_type.contents.value = Constants.saDeviceTypeSA124B

    def _get_serial_number(self, handle: ctypes.c_int, serial_number: Any) -> None:
        serial_number.contents.value = 1234

    def _get_firmware_string(self, handle: ctypes.c_int, firmware: Any) -> None:
        firmware.value = b"Version 3.0.0"

    def _config_center_span(
        self, handle: ctypes.c_int, center: ctypes.c_double, span: ctypes.c_double
    ) -> None:
        self._center = center.value
        self._span = span.value

    def _query_sweep_info(
        self, handle: ctypes.c_int, sweep_length: Any, start: Any, step: Any
    ) -> None:
        sweep_length.contents.value = _SWEEP_LENGTH
        start.contents.value = self._center - self._span / 2
        step.contents.value = self._span / (_SWEEP_LENGTH - 1)

    def _get_sweep(self, handle: ctypes.c_int, minimum: Any, maximum: Any) -> None:
        for index in range(_SWEEP_LENGTH):
            minimum[index] = -50.0 + index
            maximum[index] = -40.0 + index


@pytest.fixture(name="fake_dll")
def _make_fake_dll() -> _FakeSignalHoundDll:
    return _FakeSignalHoundDll()


@pytest.fixture(name="signal_hound")
def _make_signal_hound(
    fake_dll: _FakeSignalHoundDll,
) -> "Generator[SignalHoundUSBSA124B, None, None]":
    with mock.patch.object(ctypes, "CDLL", return_value=fake_dll):
        instrument = SignalHoundUSBSA124B("signal_hound")
    instrument.sleep_time(0)
    try:
        yield instrument
    finally:
        instrument.close()


def test_initialization_opens_and_configures_device(
    signal_hound: SignalHoundUSBSA124B, fake_dll: _FakeSignalHoundDll
) -> None:
    assert signal_hound.deviceHandle.value == _DEVICE_HANDLE
    assert fake_dll.calls[0] == "saOpenDevice"
    assert "saConfigCenterSpan" in fake_dll.calls
    assert "saInitiate" in fake_dll.calls
    # the external reference is not used by default
    assert "saSetTimebase" not in fake_dll.calls
    assert signal_hound.get_idn() == {
        "vendor": "Signal Hound",
        "model": "sa124B",
        "serial": "1234",
        "firmware": "Version 3.0.0",
    }


def test_frequency_sweep(signal_hound: SignalHoundUSBSA124B) -> None:
    signal_hound.frequency(1e9)
    signal_hound.span(1e6)

    frequencies = signal_hound.frequency_axis()
    power = signal_hound.freq_sweep()

    np.testing.assert_allclose(frequencies, np.linspace(1e9 - 0.5e6, 1e9 + 0.5e6, 5))
    np.testing.assert_allclose(power, [-50, -49, -48, -47, -46])


def test_configure_with_external_reference(
    signal_hound: SignalHoundUSBSA124B, fake_dll: _FakeSignalHoundDll
) -> None:
    signal_hound.external_reference(True)

    signal_hound.configure()

    assert "saSetTimebase" in fake_dll.calls


def test_logs_through_instrument_logger(
    signal_hound: SignalHoundUSBSA124B,
    fake_dll: _FakeSignalHoundDll,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_dll.statuses["saConfigLevel"] = saStatus.saNoCorrections

    with caplog.at_level(logging.INFO):
        signal_hound.configure()

    records = [
        (record.name, record.levelno, record.getMessage()) for record in caplog.records
    ]
    prefix = "[signal_hound(SignalHoundUSBSA124B)] "
    instrument_logger = "qcodes.instrument.instrument_base"
    assert (
        instrument_logger,
        logging.INFO,
        prefix + "Setting device CenterSpan configuration.",
    ) in records
    assert (
        instrument_logger,
        logging.INFO,
        prefix + "Call to saConfigCenterSpan was successful",
    ) in records
    assert (
        instrument_logger,
        logging.WARNING,
        prefix + "During call of saConfigLevel the followingWarning: "
        "saNoCorrections was raised",
    ) in records


def test_error_status_raises(
    signal_hound: SignalHoundUSBSA124B, fake_dll: _FakeSignalHoundDll
) -> None:
    fake_dll.statuses["saQuerySweepInfo"] = saStatus.saInvalidDeviceErr

    with pytest.raises(OSError, match=r"saQuerySweepInfo .* saInvalidDeviceErr"):
        signal_hound.QuerySweep()


def test_close_aborts_and_closes_device(
    signal_hound: SignalHoundUSBSA124B, fake_dll: _FakeSignalHoundDll
) -> None:
    signal_hound.close()
    signal_hound.close()

    assert fake_dll.calls[-2:] == ["saAbort", "saCloseDevice"]
    assert fake_dll.closed_handles == [_DEVICE_HANDLE]
    assert not Instrument.exist("signal_hound")


def test_close_closes_device_if_abort_fails(
    signal_hound: SignalHoundUSBSA124B,
    fake_dll: _FakeSignalHoundDll,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_dll.statuses["saAbort"] = saStatus.saInvalidDeviceErr

    with caplog.at_level(logging.ERROR):
        signal_hound.close()

    assert fake_dll.closed_handles == [_DEVICE_HANDLE]
    assert not Instrument.exist("signal_hound")
    assert "Could not abort acquisition" in caplog.text


def test_preset_resets_device_and_closes_instrument(
    signal_hound: SignalHoundUSBSA124B, fake_dll: _FakeSignalHoundDll
) -> None:
    signal_hound.preset()

    assert fake_dll.calls[-1] == "saPreset"
    assert not Instrument.exist("signal_hound")


def test_device_that_fails_to_open_is_not_closed(
    fake_dll: _FakeSignalHoundDll,
) -> None:
    fake_dll.statuses["saOpenDevice"] = saStatus.saDeviceNotFoundErr

    with (
        mock.patch.object(ctypes, "CDLL", return_value=fake_dll),
        pytest.raises(OSError, match="saDeviceNotFoundErr") as exc_info,
    ):
        SignalHoundUSBSA124B("signal_hound")

    assert fake_dll.closed_handles == []
    assert not hasattr(exc_info.value, "__notes__")
    assert not Instrument.exist("signal_hound")


@pytest.fixture(name="not_a_signal_hound")
def _make_not_a_signal_hound() -> "Generator[DummyInstrument, None, None]":
    instrument = DummyInstrument("not_a_signal_hound", gates=[])
    try:
        yield instrument
    finally:
        instrument.close()


def test_trace_parameter_requires_signal_hound(
    not_a_signal_hound: DummyInstrument,
) -> None:
    param = TraceParameter("trace_param", instrument=not_a_signal_hound, set_cmd=None)

    with pytest.raises(
        RuntimeError, match="TraceParameter only works with 'SignalHound_USB_SA124B'"
    ):
        param.set(1)


def test_scale_parameter_requires_signal_hound(
    not_a_signal_hound: DummyInstrument,
) -> None:
    param = ScaleParameter("scale_param", instrument=not_a_signal_hound, set_cmd=None)

    with pytest.raises(
        RuntimeError, match="ScaleParameter only works with 'SignalHound_USB_SA124B'"
    ):
        param.set("log-scale")


def test_sweep_trace_parameter_requires_signal_hound(
    not_a_signal_hound: DummyInstrument,
) -> None:
    param = SweepTraceParameter(
        "sweep_trace_param", instrument=not_a_signal_hound, set_cmd=None
    )

    with pytest.raises(
        RuntimeError,
        match="SweepTraceParameter only works with 'SignalHound_USB_SA124B'",
    ):
        param.set(1)


def test_frequency_sweep_requires_signal_hound(
    not_a_signal_hound: DummyInstrument,
) -> None:
    with pytest.raises(
        RuntimeError,
        match="'FrequencySweep' is only implemented for 'SignalHound_USB_SA124B'",
    ):
        FrequencySweep(
            "frequency_sweep",
            instrument=not_a_signal_hound,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            sweep_len=10,
            start_freq=1e9,
            stepsize=1e6,
        )
