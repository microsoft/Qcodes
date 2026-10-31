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


@pytest.mark.parametrize("value", [1, 300e-3, 100e-3, 30e-3, 10e-3])
def test_input_range(sr860: SR860, value: float) -> None:
    sr860.input_range(value)
    assert sr860.input_range() == value


@pytest.mark.parametrize("value", [1e-6, 100e-3, 1, 1.0, 30e3])
def test_time_constant(sr860: SR860, value: float) -> None:
    sr860.time_constant(value)
    assert sr860.time_constant() == value


def test_complex_voltage(sr860: SR860) -> None:
    value = sr860.complex_voltage()
    assert isinstance(value, complex)
    assert value == complex(1.5e-3, -2.5e-4)
