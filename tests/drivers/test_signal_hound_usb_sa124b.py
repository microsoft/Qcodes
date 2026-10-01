"""Tests for the parameter classes of the SignalHound USB SA124B driver.

The driver itself requires the vendor DLL and a connected device, but the
parameter classes guard against being attached to a different instrument and
that guard can be exercised with any instrument.
"""

from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.mock_instruments import DummyInstrument
from qcodes.instrument_drivers.signal_hound.SignalHound_USB_SA124B import (
    FrequencySweep,
    ScaleParameter,
    SweepTraceParameter,
    TraceParameter,
)

if TYPE_CHECKING:
    from collections.abc import Generator


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
