from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.Keysight.Keysight_N9030B import (
    KeysightN9030B,
    KeysightN9030BPhaseNoiseMode,
    KeysightN9030BSpectrumAnalyzerMode,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(name="driver")
def _make_driver() -> "Generator[KeysightN9030B, None, None]":
    driver = KeysightN9030B(
        "n9030B_sim", address="GPIB::1::INSTR", pyvisa_sim_file="Keysight_N9030B.yaml"
    )
    yield driver
    driver.close()


@pytest.fixture(name="sa")
def _activate_swept_sa_measurement(
    driver: KeysightN9030B,
) -> "Generator[KeysightN9030BSpectrumAnalyzerMode, None, None]":
    yield driver.sa


@pytest.fixture(name="pn")
def _activate_log_plot_measurement(
    driver: KeysightN9030B,
) -> "Generator[KeysightN9030BPhaseNoiseMode, None, None]":
    yield driver.pn


def test_idn(driver: KeysightN9030B) -> None:
    assert {
        "firmware": "0.1",
        "model": "N9030B",
        "serial": "1000",
        "vendor": "Keysight Technologies",
    } == driver.IDN()


def test_swept_sa_setup(sa: KeysightN9030BSpectrumAnalyzerMode) -> None:
    assert isinstance(sa, KeysightN9030BSpectrumAnalyzerMode)

    sa.setup_swept_sa_sweep(123, 11e3, 501)
    assert sa.root_instrument.mode() == "SA"
    assert sa.root_instrument.measurement() == "SAN"

    assert sa.start() == 123
    assert sa.stop() == 11e3
    assert sa.npts() == 501


def test_log_plot_setup(pn: KeysightN9030BPhaseNoiseMode) -> None:
    assert isinstance(pn, KeysightN9030BPhaseNoiseMode)

    pn.setup_log_plot_sweep(1000, 1e7, 10001)
    assert pn.root_instrument.mode() == "PNOISE"
    assert pn.root_instrument.measurement() == "LPL"

    assert pn.start_offset() == 1000
    assert pn.stop_offset() == 1e7
    assert pn.npts() == 10001
