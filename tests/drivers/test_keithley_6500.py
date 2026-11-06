from typing import TYPE_CHECKING

import numpy as np
import pytest

from qcodes.instrument_drivers.Keithley import Keithley6500

if TYPE_CHECKING:
    from collections.abc import Generator

    from pytest_mock import MockerFixture


@pytest.fixture(scope="function", name="dmm6500")
def _make_dmm6500() -> "Generator[Keithley6500, None, None]":
    driver = Keithley6500(
        "dmm6500", address="GPIB::1::INSTR", pyvisa_sim_file="Keithley_6500.yaml"
    )
    try:
        yield driver
    finally:
        driver.close()


def test_digits(dmm6500: Keithley6500) -> None:
    assert dmm6500.digits() == 6

    dmm6500.digits(5)
    assert dmm6500.digits() == 5


@pytest.mark.parametrize("count", [0, 1, 1440, 100_000_000])
def test_trigger_count(dmm6500: Keithley6500, count: int) -> None:
    # 0 repeats the scan until it is aborted
    dmm6500.trigger_count(count)
    value = dmm6500.trigger_count()
    assert isinstance(value, int)
    assert value == count


@pytest.mark.parametrize("value", [10.0, np.float64(10)], ids=["float", "numpy_float"])
def test_trigger_count_accepts_whole_floats(
    dmm6500: Keithley6500, mocker: "MockerFixture", value: float
) -> None:
    write_raw = mocker.spy(dmm6500, "write_raw")

    dmm6500.trigger_count(value)  # pyright: ignore[reportArgumentType]

    write_raw.assert_called_once_with("ROUT:SCAN:COUN:SCAN 10")
    assert dmm6500.trigger_count() == 10


@pytest.mark.parametrize(
    "value",
    [-1, 100_000_001, 2.5, "inf"],
    ids=["negative", "too_large", "fraction", "inf"],
)
def test_trigger_count_rejects_invalid(
    dmm6500: Keithley6500, mocker: "MockerFixture", value: float | str
) -> None:
    write_raw = mocker.spy(dmm6500, "write_raw")

    with pytest.raises((ValueError, TypeError)):
        dmm6500.trigger_count(value)  # pyright: ignore[reportArgumentType]
    write_raw.assert_not_called()
