"""
Closing a driver must also work if the instrument has already been closed, in
which case its attributes have been stripped, or if its initialization failed
before all its attributes were set.

The drivers tested here need hardware, vendor libraries or servers to be
initialized, so they are closed without having been initialized, which leaves
them without any instance attributes, like a closed instrument, or initialized
with fake vendor libraries that fail to open the device.
"""

import ctypes
from unittest import mock

import pytest
from pyvisa.errors import InvalidSession

from qcodes.instrument import Instrument, IPInstrument
from qcodes.instrument_drivers.Keysight.KtM960x import KeysightM960x
from qcodes.instrument_drivers.Keysight.KtMAwg import KeysightM9336A
from qcodes.instrument_drivers.QuantumDesign import DynaCool
from qcodes.instrument_drivers.signal_hound.SignalHound_USB_SA124B import (
    SignalHoundUSBSA124B,
    saStatus,
)


@pytest.mark.parametrize(
    "instrument_class",
    [IPInstrument, KeysightM960x, KeysightM9336A, DynaCool, SignalHoundUSBSA124B],
)
def test_close_instrument_without_attributes(
    instrument_class: type[Instrument],
) -> None:
    instrument = instrument_class.__new__(instrument_class)

    instrument.close()
    instrument.close()


def test_dynacool_closes_visa_handle_if_session_is_already_closed() -> None:
    """
    DynaCool closes its VISA handle also if saying goodbye to the server fails
    because the VISA session has already been closed, e.g. by its finalizer
    when the garbage collector closes the instrument.
    """
    instrument = DynaCool.__new__(DynaCool)
    visa_handle = mock.MagicMock()
    instrument._visa_handle = visa_handle
    instrument.log = mock.MagicMock()

    with mock.patch.object(DynaCool, "write", side_effect=InvalidSession()):
        instrument.close()

    visa_handle.close.assert_called_once()


@pytest.mark.parametrize(
    ("instrument_class", "function_prefix"),
    [
        pytest.param(
            KeysightM960x,
            "KtM960x",
            # unrelated to closing: its Measure parameter does not pass on kwargs
            marks=pytest.mark.filterwarnings(
                "ignore:Parameter measure_data on instrument keysight does not "
                "correctly pass kwargs"
            ),
        ),
        (KeysightM9336A, "KtMAwg"),
    ],
)
def test_keysight_does_not_close_session_that_failed_to_open(
    instrument_class: type[KeysightM960x | KeysightM9336A], function_prefix: str
) -> None:
    dll = mock.MagicMock()
    getattr(dll, f"{function_prefix}_InitWithOptions").return_value = 1

    with (
        mock.patch.object(ctypes.cdll, "LoadLibrary", return_value=dll),
        pytest.raises(SystemError, match="connection to device failed") as exc_info,
    ):
        instrument_class("keysight", address="PXI0::1::INSTR")

    getattr(dll, f"{function_prefix}_close").assert_not_called()
    assert not hasattr(exc_info.value, "__notes__")
    assert not Instrument.exist("keysight")


def test_signal_hound_does_not_close_device_that_failed_to_open() -> None:
    dll = mock.MagicMock()
    dll.saOpenDevice.return_value = saStatus.saDeviceNotFoundErr

    with (
        mock.patch.object(ctypes, "CDLL", return_value=dll),
        pytest.raises(OSError, match="saDeviceNotFoundErr") as exc_info,
    ):
        SignalHoundUSBSA124B("signal_hound")

    dll.saCloseDevice.assert_not_called()
    assert not hasattr(exc_info.value, "__notes__")
    assert not Instrument.exist("signal_hound")
