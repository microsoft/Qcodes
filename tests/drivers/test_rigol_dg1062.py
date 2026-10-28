from typing import TYPE_CHECKING

import numpy as np
import pytest

from qcodes.instrument_drivers.rigol import RigolDG1062

if TYPE_CHECKING:
    from collections.abc import Generator

    from pytest_mock import MockerFixture


@pytest.fixture(scope="function", name="dg1062")
def _make_dg1062() -> "Generator[RigolDG1062, None, None]":
    driver = RigolDG1062(
        "dg1062", address="GPIB::1::INSTR", pyvisa_sim_file="Rigol_DG1062.yaml"
    )
    try:
        yield driver
    finally:
        driver.close()


def test_burst_ncycles_is_int(dg1062: RigolDG1062) -> None:
    burst = dg1062.ch1.burst
    assert burst.ncycles() == 1

    burst.ncycles(10)
    ncycles = burst.ncycles()
    assert isinstance(ncycles, int)
    assert ncycles == 10


@pytest.mark.parametrize("value", [10.0, np.float64(10)], ids=["float", "numpy_float"])
def test_burst_ncycles_accepts_whole_floats(
    dg1062: RigolDG1062, mocker: "MockerFixture", value: float
) -> None:
    # whole-number floats, e.g. from numpy, are accepted and sent as int
    burst = dg1062.ch1.burst
    write_raw = mocker.spy(dg1062, "write_raw")

    burst.ncycles(value)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]

    write_raw.assert_called_once_with(":SOUR1:BURS:NCYC 10")
    assert burst.ncycles() == 10


def test_burst_ncycles_rejects_non_integer(
    dg1062: RigolDG1062, mocker: "MockerFixture"
) -> None:
    write_raw = mocker.spy(dg1062, "write_raw")

    with pytest.raises(TypeError):
        dg1062.ch1.burst.ncycles(2.5)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
    write_raw.assert_not_called()


def test_burst_phase_is_float(dg1062: RigolDG1062) -> None:
    burst = dg1062.ch1.burst
    assert burst.phase() == 0.0

    burst.phase(10.5)
    phase = burst.phase()
    assert isinstance(phase, float)
    assert phase == 10.5


def test_burst_time_delay_is_float(dg1062: RigolDG1062) -> None:
    burst = dg1062.ch1.burst
    assert burst.time_delay() == 0.0

    burst.time_delay(0.1)
    time_delay = burst.time_delay()
    assert isinstance(time_delay, float)
    assert time_delay == 0.1


def test_waveform_is_str(dg1062: RigolDG1062) -> None:
    assert dg1062.ch1.waveform() == "SQU"
    assert dg1062.ch2.waveform() == "SIN"


def test_waveform_params_are_floats(dg1062: RigolDG1062) -> None:
    assert dg1062.ch1.freq() == 1000.0
    assert dg1062.ch1.ampl() == 5.0


def test_duty_cycle_is_float(dg1062: RigolDG1062) -> None:
    assert dg1062.ch1.waveform() == "SQU"
    assert dg1062.ch1.duty_cycle() == 50.0

    dg1062.ch1.duty_cycle(45)
    duty_cycle = dg1062.ch1.duty_cycle()
    assert isinstance(duty_cycle, float)
    assert duty_cycle == 45.0
