import pytest

import qcodes.instrument_drivers.Keysight.Keysight_N6705B as N6705B


@pytest.fixture(scope="function", name="driver")
def _make_driver():
    driver = N6705B.N6705B(
        "N6705B", address="GPIB::1::INSTR", pyvisa_sim_file="Keysight_N6705B.yaml"
    )
    yield driver
    driver.close()


def test_idn(driver) -> None:
    assert {
        "firmware": "D.01.08",
        "model": "N6705B",
        "serial": "MY50001897",
        "vendor": "Agilent Technologies",
    } == driver.IDN()


def test_channels(driver) -> None:
    # Ensure each channel got instantiated
    assert len(driver.channels) == 4
    for i, ch in enumerate(driver.channels):
        assert ch.channel == i + 1
        assert ch.ch_name == f"ch{i+1}"
