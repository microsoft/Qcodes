from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.ithaco import CurrentParameter, Ithaco1211
from qcodes.parameters import Parameter

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(name="ithaco")
def _make_ithaco() -> "Generator[Ithaco1211, None, None]":
    driver = Ithaco1211("ithaco")
    try:
        yield driver
    finally:
        driver.close()


def test_initial_values(ithaco: Ithaco1211) -> None:
    assert ithaco.sens() == 1e-8
    assert ithaco.sens_factor() == 1
    assert ithaco.suppression() == 1e-7
    assert ithaco.risetime() == 0.3
    assert ithaco.invert() is True


@pytest.mark.parametrize(
    ("name", "valid", "invalid"),
    [
        ("sens", 1e-4, 2e-4),
        ("sens_factor", 0.1, 5),
        ("suppression", 1e-3, 1e-2),
        ("risetime", 1000, 2000),
    ],
)
def test_parameters_validate(
    ithaco: Ithaco1211, name: str, valid: float, invalid: float
) -> None:
    param = ithaco.parameters[name]
    assert isinstance(param, Parameter)

    param(valid)
    assert param() == valid

    with pytest.raises(ValueError):
        param(invalid)


@pytest.mark.parametrize(("invert", "sign"), [(True, -1), (False, 1)])
def test_current_parameter(ithaco: Ithaco1211, invert: bool, sign: int) -> None:
    volt = Parameter("volt", set_cmd=None, initial_value=2.0, unit="V", label="Volt")
    ithaco.sens(1e-6)
    ithaco.sens_factor(10)
    ithaco.invert(invert)

    curr = CurrentParameter(volt, ithaco)

    assert curr.names == ("volt_raw", "curr")
    assert curr.units == ("V", "A")
    assert curr.labels == ("Volt", "Current")
    assert curr() == pytest.approx((2.0, sign * 2.0 * 1e-6 * 10))
