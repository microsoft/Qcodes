from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.stanford_research import SR560

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(name="sr560")
def _make_sr560() -> "Generator[SR560, None, None]":
    driver = SR560("sr560")
    try:
        yield driver
    finally:
        driver.close()


def test_initial_values(sr560: SR560) -> None:
    assert sr560.cutoff_lo() == "DC"
    assert sr560.cutoff_hi() == 1e6
    assert sr560.gain() == 10
    assert sr560.invert() is True


@pytest.mark.parametrize("value", ["DC", 0.03, 1, 10e3])
def test_cutoff_lo_accepts_high_pass_settings(sr560: SR560, value: float) -> None:
    sr560.cutoff_lo(value)
    assert sr560.cutoff_lo() == value


@pytest.mark.parametrize("value", [30e3, 100e3, 300e3, 1e6])
def test_cutoff_lo_rejects_settings_without_high_pass(
    sr560: SR560, value: float
) -> None:
    with pytest.raises(ValueError, match="is not in"):
        sr560.cutoff_lo(value)


@pytest.mark.parametrize("value", [0.03, 1, 10e3, 1e6])
def test_cutoff_hi_accepts_low_pass_settings(sr560: SR560, value: float) -> None:
    sr560.cutoff_hi(value)
    assert sr560.cutoff_hi() == value


def test_cutoff_hi_rejects_dc(sr560: SR560) -> None:
    with pytest.raises(ValueError, match="is not in"):
        sr560.cutoff_hi("DC")  # type: ignore[call-overload]


@pytest.mark.parametrize("value", [1, 2, 5, 100, 50000])
def test_gain(sr560: SR560, value: int) -> None:
    sr560.gain(value)
    assert sr560.gain() == value


def test_gain_rejects_vernier_values(sr560: SR560) -> None:
    with pytest.raises(ValueError, match="is not in"):
        sr560.gain(3)
