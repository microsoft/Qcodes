from typing import TYPE_CHECKING

import numpy as np
import pytest

from qcodes.instrument_drivers.AimTTi import AimTTiPL601

if TYPE_CHECKING:
    from collections.abc import Generator

    from pytest_mock import MockerFixture


@pytest.fixture(scope="function", name="driver")
def _make_driver() -> "Generator[AimTTiPL601, None, None]":
    driver = AimTTiPL601(
        "AimTTi", address="GPIB::1::INSTR", pyvisa_sim_file="AimTTi_PL601P.yaml"
    )

    yield driver
    driver.close()


def test_idn(driver: AimTTiPL601) -> None:
    assert {
        "firmware": "3.05-4.06",
        "model": "PL601-P",
        "serial": "514710",
        "vendor": "THURLBY THANDAR",
    } == driver.IDN()


@pytest.mark.parametrize(
    "value", [2, 2.0, np.float64(2)], ids=["int", "float", "numpy_float"]
)
def test_curr_range_accepts_whole_numbers(
    driver: AimTTiPL601, mocker: "MockerFixture", value: float
) -> None:
    # the range is 1 (low) or 2 (high); whole-number floats are sent as int
    channel = driver.ch1
    # the output is switched off while changing the range; start from off so
    # that its state does not have to be queried
    channel.output.cache.set(False)
    write_raw = mocker.patch.object(driver, "write_raw")

    channel.curr_range(value)  # pyright: ignore[reportArgumentType]

    write_raw.assert_any_call("IRANGE1 2")


def test_curr_range_rejects_non_integer(
    driver: AimTTiPL601, mocker: "MockerFixture"
) -> None:
    write_raw = mocker.patch.object(driver, "write_raw")

    with pytest.raises(TypeError):
        driver.ch1.curr_range(1.5)  # pyright: ignore[reportArgumentType]
    write_raw.assert_not_called()
