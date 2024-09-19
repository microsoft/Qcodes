from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.yokogawa import YokogawaGS200

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="function", name="gs200")
def _make_gs200() -> "Iterator[YokogawaGS200]":
    gs200 = YokogawaGS200(
        "GS200", address="GPIB0::1::INSTR", pyvisa_sim_file="Yokogawa_GS200.yaml"
    )
    yield gs200

    gs200.close()


def test_basic_init(gs200: YokogawaGS200) -> None:
    idn = gs200.get_idn()
    assert idn["vendor"] == "QCoDeS Yokogawa Mock"


def test_current_raises_in_voltage_mode(gs200: YokogawaGS200) -> None:
    gs200.source_mode("VOLT")

    with pytest.raises(
        ValueError, match="Cannot get/set CURR settings while in VOLT mode"
    ):
        gs200.current_range()

    with pytest.raises(
        ValueError, match="Cannot get/set CURR settings while in VOLT mode"
    ):
        gs200.current(1)


def test_voltage_raises_in_current_mode(gs200: YokogawaGS200) -> None:
    gs200.source_mode("CURR")

    with pytest.raises(
        ValueError, match="Cannot get/set VOLT settings while in CURR mode"
    ):
        gs200.voltage_range()

    with pytest.raises(
        ValueError, match="Cannot get/set VOLT settings while in CURR mode"
    ):
        gs200.voltage(1)


def test_get_parameters_as_components(gs200: YokogawaGS200) -> None:
    assert gs200.get_component("voltage_range") is gs200.voltage_range
    assert gs200.get_component("voltage") is gs200.voltage
