import logging
from typing import TYPE_CHECKING

import pytest
from pytest import FixtureRequest, LogCaptureFixture

from qcodes.instrument_drivers.Keysight import keysight_34980a
from qcodes.instrument_drivers.Keysight.keysight_34934a import Keysight34934A
from qcodes.instrument_drivers.Keysight.keysight_34980a import Keysight34980A
from qcodes.instrument_drivers.Keysight.keysight_34980a_submodules import (
    Keysight34980ASwitchMatrixSubModule,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(scope="function")
def switch_driver() -> "Generator[Keysight34980A, None, None]":
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
    To check if a warning would show when initialize the instrument with a
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
    To check if the instrument attributes are set correctly after getting
    the IDN
    """
    assert switch_driver.IDN() == {
        "vendor": "Keysight",
        "model": "34980A",
        "serial": "1000",
        "firmware": "0.1",
    }


def test_scan_slots(switch_driver: Keysight34980A) -> None:
    """
    To check if the submodule attributes are set correctly after scanning
    every slot
    """
    assert len(switch_driver.system_slots_info) == 2

    assert switch_driver.system_slots_info[1] == {
        "vendor": "Agilent Technologies",
        "model": "34934A-8x64",
        "serial": "AB10000000",
        "firmware": "1.00",
    }

    assert switch_driver.system_slots_info[3] == {
        "vendor": "Agilent Technologies",
        "model": "34934A-4x32",
        "serial": "AB10000001",
        "firmware": "1.00",
    }


def test_safety_interlock(
    switch_driver: Keysight34980A, caplog: LogCaptureFixture
) -> None:
    """
    To check if a warning would show when talk to a module that is in safety
    interlock state
    """
    module = switch_driver.module[3]
    module.write("*CLS")
    with caplog.at_level(logging.DEBUG):
        assert "safety interlock" in caplog.text


def test_modules_only_for_occupied_slots(switch_driver: Keysight34980A) -> None:
    # empty slots report model "0" and get no module
    assert set(switch_driver.module) == {1, 3}
    for slot, module in switch_driver.module.items():
        assert isinstance(module, Keysight34934A)
        assert switch_driver.submodules[f"slot_{slot}_34934A"] is module


def test_unknown_module_falls_back_to_generic_submodule(
    request: FixtureRequest,
    caplog: LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keysight_34980a, "KEYSIGHT_MODELS", {})
    # the model string contains a "-", which is not valid in a name
    with (
        caplog.at_level(logging.WARNING),
        pytest.warns(UserWarning, match="_no_driver") as warning_records,
    ):
        inst = Keysight34980A(
            "keysight_34980A_sim",
            address="GPIB::1::INSTR",
            pyvisa_sim_file="keysight_34980A.yaml",
        )
    request.addfinalizer(inst.close)

    warning_messages = {str(record.message) for record in warning_records}
    assert any("slot_1_34934A_8x64_no_driver" in msg for msg in warning_messages)
    assert any("slot_3_34934A_4x32_no_driver" in msg for msg in warning_messages)

    assert set(inst.module) == {1, 3}
    for slot, model in ((1, "34934A-8x64"), (3, "34934A-4x32")):
        module = inst.module[slot]
        assert type(module) is Keysight34980ASwitchMatrixSubModule
        assert inst.submodules[f"slot_{slot}_{model}_no_driver"] is module
        assert f"can not find driver for {model} in slot {slot}" in caplog.text


def test_rescan_slots_keeps_existing_modules(switch_driver: Keysight34980A) -> None:
    modules = dict(switch_driver.module)
    submodules = dict(switch_driver.submodules)

    switch_driver.scan_slots()

    assert switch_driver.module == modules
    assert switch_driver.submodules == submodules


def test_rescan_slots_keeps_existing_generic_modules(
    request: FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keysight_34980a, "KEYSIGHT_MODELS", {})
    with pytest.warns(UserWarning, match="_no_driver"):
        inst = Keysight34980A(
            "keysight_34980A_sim",
            address="GPIB::1::INSTR",
            pyvisa_sim_file="keysight_34980A.yaml",
        )
    request.addfinalizer(inst.close)
    modules = dict(inst.module)

    inst.scan_slots()

    assert inst.module == modules
    for slot, model in ((1, "34934A-8x64"), (3, "34934A-4x32")):
        assert inst.submodules[f"slot_{slot}_{model}_no_driver"] is inst.module[slot]
