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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("50", "50OHMS"),
        ("50OHMS", "50OHMS"),
        ("1M", "1MEG"),
        ("1MEG", "1MEG"),
    ],
)
def test_external_reference_trigger_input_resistance_returns_name(
    sr860: SR860, value: str, expected: str
) -> None:
    sr860.external_reference_trigger_input_resistance(value)
    assert sr860.external_reference_trigger_input_resistance() == expected


@pytest.mark.parametrize(("value", "expected"), [(0, "50OHMS"), (1, "1MEG")])
def test_external_reference_trigger_input_resistance_int_alias(
    sr860: SR860, value: int, expected: str
) -> None:
    # The instrument's integer values are still accepted for backwards
    # compatibility, but get always returns the name.
    sr860.external_reference_trigger_input_resistance.set(value)
    assert sr860.external_reference_trigger_input_resistance() == expected
