from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.Keithley import Keithley6500

if TYPE_CHECKING:
    from collections.abc import Generator


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
