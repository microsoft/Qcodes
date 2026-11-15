"""
Closing a driver must also work if the instrument has already been closed, in
which case its attributes have been stripped, or if its initialization failed
before all its attributes were set.

The drivers tested here need hardware, vendor libraries or servers to be
initialized, so they are closed without having been initialized, which leaves
them without any instance attributes, like a closed instrument, or initialized
with fake vendor libraries. The SignalHound driver is tested with a fake of its
library in test_signal_hound_usb_sa124b.py.
"""

import ctypes
from typing import Any
from unittest import mock

import pytest
from pyvisa import VisaIOError
from pyvisa.constants import StatusCode
from pyvisa.errors import InvalidSession

from qcodes.instrument import Instrument, IPInstrument
from qcodes.instrument_drivers.Keysight.KtM960x import KeysightM960x
from qcodes.instrument_drivers.Keysight.KtMAwg import KeysightM9336A
from qcodes.instrument_drivers.QuantumDesign import DynaCool
from qcodes.instrument_drivers.signal_hound.SignalHound_USB_SA124B import (
    SignalHoundUSBSA124B,
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


@pytest.mark.parametrize(
    "error",
    [
        # e.g. closed by its finalizer when the garbage collector closes it
        pytest.param(InvalidSession(), id="session already closed"),
        pytest.param(VisaIOError(StatusCode.error_timeout), id="server down"),
        pytest.param(ValueError("invalid literal"), id="malformed reply"),
    ],
)
def test_dynacool_closes_visa_handle_if_closing_server_connection_fails(
    error: Exception,
) -> None:
    """
    DynaCool closes its VISA handle also if telling the server that it closes
    the connection fails.
    """
    instrument = DynaCool.__new__(DynaCool)
    visa_handle = mock.MagicMock()
    instrument._visa_handle = visa_handle
    instrument.log = mock.MagicMock()

    with mock.patch.object(DynaCool, "write", side_effect=error):
        instrument.close()

    visa_handle.close.assert_called_once()


_KEYSIGHT_DRIVERS = pytest.mark.parametrize(
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


@_KEYSIGHT_DRIVERS
def test_keysight_does_not_close_session_that_failed_to_open(
    instrument_class: type[KeysightM960x | KeysightM9336A], function_prefix: str
) -> None:
    dll = mock.MagicMock()
    # a negative status is an error, with which no session is opened
    getattr(dll, f"{function_prefix}_InitWithOptions").return_value = -1

    with (
        mock.patch.object(ctypes.cdll, "LoadLibrary", return_value=dll),
        pytest.raises(SystemError, match="connection to device failed") as exc_info,
    ):
        instrument_class("keysight", address="PXI0::1::INSTR")

    getattr(dll, f"{function_prefix}_close").assert_not_called()
    assert not hasattr(exc_info.value, "__notes__")
    assert not Instrument.exist("keysight")


@_KEYSIGHT_DRIVERS
def test_keysight_closes_session_opened_with_warning(
    instrument_class: type[KeysightM960x | KeysightM9336A], function_prefix: str
) -> None:
    """
    A positive status is a warning, with which the session is opened anyway.
    The driver still treats it as a failure, so it must close the session.
    """

    def open_session_with_warning(
        address: bytes, id_query: int, reset: int, options: bytes, session: Any
    ) -> int:
        session._obj.value = 42
        return 1

    dll = mock.MagicMock()
    getattr(
        dll, f"{function_prefix}_InitWithOptions"
    ).side_effect = open_session_with_warning

    with (
        mock.patch.object(ctypes.cdll, "LoadLibrary", return_value=dll),
        pytest.raises(SystemError, match="connection to device failed"),
    ):
        instrument_class("keysight", address="PXI0::1::INSTR")

    close = getattr(dll, f"{function_prefix}_close")
    close.assert_called_once()
    (session,) = close.call_args.args
    assert session.value == 42
    assert not Instrument.exist("keysight")


class _SucceedingIviDll:
    """A fake IVI DLL whose functions all succeed without returning any data."""

    def __init__(self) -> None:
        self._functions: dict[str, mock.MagicMock] = {}

    def __getattr__(self, name: str) -> mock.MagicMock:
        return self._functions.setdefault(name, mock.MagicMock(return_value=0))


@_KEYSIGHT_DRIVERS
def test_keysight_closes_open_session_once(
    instrument_class: type[KeysightM960x | KeysightM9336A], function_prefix: str
) -> None:
    dll = _SucceedingIviDll()
    with mock.patch.object(ctypes.cdll, "LoadLibrary", return_value=dll):
        instrument = instrument_class("keysight", address="PXI0::1::INSTR")

    instrument.close()
    instrument.close()

    close = getattr(dll, f"{function_prefix}_close")
    close.assert_called_once()
    (session,) = close.call_args.args
    assert isinstance(session, ctypes.c_int)
    assert not Instrument.exist("keysight")
