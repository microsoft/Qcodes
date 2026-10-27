from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.tektronix import TektronixTPS2012

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(scope="function", name="tps2012")
def _make_tps2012() -> "Generator[TektronixTPS2012, None, None]":
    driver = TektronixTPS2012(
        "tps2012",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="Tektronix_TPS2012.yaml",
        # the driver does not set a default terminator; pyvisa-sim needs one
        terminator="\n",
    )
    try:
        yield driver
    finally:
        driver.close()


def test_trigger_level_is_float(tps2012: TektronixTPS2012) -> None:
    trigger_level = tps2012.trigger_level()
    assert isinstance(trigger_level, float)
    assert trigger_level == 1.4

    tps2012.trigger_level(0.25)
    assert tps2012.trigger_level() == 0.25
