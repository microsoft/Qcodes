from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.stanford_research import SR560, SR560VoltageParameter
from qcodes.parameters import Parameter

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
    assert sr560.invert() is True
    assert sr560.gain() == 10
    assert sr560.get_idn() == {
        "vendor": "Stanford Research Systems",
        "model": "SR560",
        "serial": None,
        "firmware": None,
    }


@pytest.mark.parametrize(
    ("name", "valid", "invalid"),
    [
        ("cutoff_lo", 0.03, 0.05),
        ("cutoff_hi", "DC", 2e6),
        ("gain", 50000, 3),
    ],
)
def test_parameters_validate(
    sr560: SR560, name: str, valid: float | str, invalid: float
) -> None:
    param = sr560.parameters[name]
    assert isinstance(param, Parameter)

    param(valid)
    assert param() == valid

    with pytest.raises(ValueError):
        param(invalid)


@pytest.mark.parametrize(("invert", "sign"), [(True, -1), (False, 1)])
def test_voltage_parameter(sr560: SR560, invert: bool, sign: int) -> None:
    measured = Parameter(
        "measured", set_cmd=None, initial_value=2.0, unit="V", label="Measured"
    )
    sr560.gain(100)
    sr560.invert(invert)

    volt = SR560VoltageParameter(measured, sr560)

    assert volt.names == ("measured_raw", "volt")
    assert volt.units == ("V", "V")
    assert volt.labels == ("Measured", "Voltage")
    assert volt() == pytest.approx((2.0, sign * 2.0 / 100))
