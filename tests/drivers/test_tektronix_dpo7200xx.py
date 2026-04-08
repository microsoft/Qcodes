from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.tektronix.DPO7200xx import TektronixDPO7000xx

if TYPE_CHECKING:
    from collections.abc import Generator

    from pytest_mock import MockerFixture


@pytest.fixture(scope="function")
def tektronix_dpo() -> "Generator[TektronixDPO7000xx, None, None]":
    """
    A six channel-per-relay instrument
    """
    driver = TektronixDPO7000xx(
        "dpo",
        address="TCPIP0::0.0.0.0::inst0::INSTR",
        pyvisa_sim_file="Tektronix_DPO7200xx.yaml",
    )

    yield driver
    driver.close()


def test_adjust_timer(
    tektronix_dpo: TektronixDPO7000xx, mocker: "MockerFixture"
) -> None:
    """
    After adjusting the type of the measurement or the source of the
    measurement, we need wait at least 0.1 seconds
    ('minimum_adjustment_time') before a measurement value can be
    retrieved. Test this.
    """
    measurement = tektronix_dpo.measurement[0]
    min_time = measurement._minimum_adjustment_time

    mock_time = mocker.patch("qcodes.instrument_drivers.tektronix.DPO7200xx.time")

    # Simulate: source was set at t=1.0, measurement read at t=1.05
    # (only 0.05s elapsed, less than the 0.1s minimum)
    mock_time.perf_counter.return_value = 1.05
    measurement._adjustment_time = 1.0
    measurement.wait_adjustment_time()
    mock_time.sleep.assert_called_once_with(pytest.approx(min_time - 0.05))

    # Simulate: enough time has passed (0.2s > 0.1s minimum)
    mock_time.reset_mock()
    mock_time.perf_counter.return_value = 1.2
    measurement._adjustment_time = 1.0
    measurement.wait_adjustment_time()
    mock_time.sleep.assert_not_called()

    # Verify _set_source records the adjustment time
    mock_time.perf_counter.return_value = 5.0
    measurement._set_source(1, "CH1")
    assert measurement._adjustment_time == 5.0

    # Verify _set_measurement_type records the adjustment time
    mock_time.perf_counter.return_value = 6.0
    measurement._set_measurement_type("AMPlitude")
    assert measurement._adjustment_time == 6.0


def test_measurements_return_float(tektronix_dpo: TektronixDPO7000xx) -> None:
    amplitude = tektronix_dpo.measurement[0].amplitude()
    assert isinstance(amplitude, float)

    mean_amplitude = tektronix_dpo.measurement[0].amplitude.mean()
    assert isinstance(mean_amplitude, float)


def test_measurement_sets_state(tektronix_dpo: TektronixDPO7000xx) -> None:
    tektronix_dpo.measurement[1].frequency()
    assert tektronix_dpo.measurement[1].state() == 1
