from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.stanford_research import SR860

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(scope="function", name="sr860")
def _make_sr860() -> "Generator[SR860, None, None]":
    driver = SR860("sr860", address="GPIB::1::INSTR", pyvisa_sim_file="SR860.yaml")
    try:
        yield driver
    finally:
        driver.close()


def test_count_capture_kilobytes_is_int(sr860: SR860) -> None:
    captured = sr860.buffer.count_capture_kilobytes()
    assert isinstance(captured, int)
    assert captured == 12
