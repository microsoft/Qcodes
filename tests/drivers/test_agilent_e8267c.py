from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.agilent import AgilentE8267C

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(scope="function", name="e8267c")
def _make_e8267c() -> "Generator[AgilentE8267C, None, None]":
    driver = AgilentE8267C(
        "e8267c", address="GPIB::1::INSTR", pyvisa_sim_file="Agilent_E8267C.yaml"
    )
    try:
        yield driver
    finally:
        driver.close()


def test_pulse_width_is_float_in_seconds(e8267c: AgilentE8267C) -> None:
    assert e8267c.pulse_width.unit == "s"

    pulse_width = e8267c.pulse_width()
    assert isinstance(pulse_width, float)
    assert pulse_width == pytest.approx(1e-6)

    e8267c.pulse_width(15e-9)
    assert e8267c.pulse_width() == pytest.approx(15e-9)
