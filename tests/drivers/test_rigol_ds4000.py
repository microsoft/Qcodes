from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.rigol import RigolDS4000

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(scope="function", name="ds4000")
def _make_ds4000() -> "Generator[RigolDS4000, None, None]":
    driver = RigolDS4000(
        "rigol_ds4000",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="Rigol_DS4000.yaml",
        # the driver does not set a default terminator; pyvisa-sim needs one
        terminator="\n",
    )
    try:
        yield driver
    finally:
        driver.close()


def test_enable_auto_scale(ds4000: RigolDS4000) -> None:
    assert ds4000.enable_auto_scale() is True

    ds4000.enable_auto_scale(False)
    assert ds4000.enable_auto_scale() is False

    ds4000.enable_auto_scale(True)
    assert ds4000.enable_auto_scale() is True
