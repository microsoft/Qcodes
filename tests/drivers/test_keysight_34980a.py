import logging

import pytest
from pytest import FixtureRequest, LogCaptureFixture

from qcodes.instrument_drivers.Keysight.keysight_34980a import Keysight34980A


@pytest.fixture(scope="function")
def switch_driver():
    inst = Keysight34980A(
        "keysight_34980A_sim",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="keysight_34980A.yaml",
    )

    try:
        yield inst
    finally:
        inst.close()


def test_safety_interlock_during_init(
    request: FixtureRequest, caplog: LogCaptureFixture
) -> None:
    """
    to check if a warning would show when initialize the instrument with a
    module in safety interlock state.
    """
    with caplog.at_level(logging.WARNING):
        inst = Keysight34980A(
            "keysight_34980A_sim",
            address="GPIB::1::INSTR",
            pyvisa_sim_file="keysight_34980A.yaml",
        )
    request.addfinalizer(inst.close)

    assert "safety interlock" in caplog.records[0].msg


def test_get_idn(switch_driver: Keysight34980A) -> None:
    """
    to check if the instrument attributes are set correctly after getting
    the IDN
    """
    assert switch_driver.IDN() == {
        "vendor": "Keysight",
        "model": "34980A",
        "serial": "1000",
        "firmware": "0.1"
    }


def test_scan_slots(switch_driver: Keysight34980A) -> None:
    """
    to check if the submodule attributes are set correctly after scanning
    every slot
    """
    assert len(switch_driver.system_slots_info) == 2

    assert switch_driver.system_slots_info[1] == {
        "vendor": "Agilent Technologies",
        "model": "34934A-8x64",
        "serial": "AB10000000",
        "firmware": "1.00"
    }

    assert switch_driver.system_slots_info[3] == {
        "vendor": "Agilent Technologies",
        "model": "34934A-4x32",
        "serial": "AB10000001",
        "firmware": "1.00"
    }


def test_safety_interlock(
    switch_driver: Keysight34980A, caplog: LogCaptureFixture
) -> None:
    """
    to check if a warning would show when talk to a module that is in safety
    interlock state
    """
    module = switch_driver.module[3]
    assert module is not None
    module.write("*CLS")
    with caplog.at_level(logging.DEBUG):
        assert "safety interlock" in caplog.text
