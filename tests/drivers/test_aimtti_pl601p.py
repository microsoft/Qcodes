from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.AimTTi import AimTTiPL601

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(scope="function", name="driver")
def _make_driver() -> "Generator[AimTTiPL601, None, None]":
    driver = AimTTiPL601(
        "AimTTi", address="GPIB::1::INSTR", pyvisa_sim_file="AimTTi_PL601P.yaml"
    )

    yield driver
    driver.close()


def test_idn(driver: AimTTiPL601) -> None:
    assert {
        "firmware": "3.05-4.06",
        "model": "PL601-P",
        "serial": "514710",
        "vendor": "THURLBY THANDAR",
    } == driver.IDN()
