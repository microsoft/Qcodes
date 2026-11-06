from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import pytest

from qcodes.instrument_drivers.yokogawa import YokogawaGS200

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_mock import MockerFixture


SOURCE_LIMITS = (
    ("VOLT", 0.01, 0.012),
    ("VOLT", 0.1, 0.12),
    ("VOLT", 1.0, 1.2),
    ("VOLT", 10.0, 12.0),
    ("VOLT", 30.0, 32.0),
    ("CURR", 0.001, 0.0012),
    ("CURR", 0.01, 0.012),
    ("CURR", 0.1, 0.12),
    ("CURR", 0.2, 0.2),
)


@pytest.fixture(scope="function", name="gs200")
def _make_gs200() -> Iterator[YokogawaGS200]:
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


@pytest.mark.parametrize("mode,nominal,limit", SOURCE_LIMITS)
@pytest.mark.parametrize("sign", (-1, 1))
@pytest.mark.parametrize("ramp_mode", ("JUMP", "SOFTWARE", "HARDWARE"))
def test_generated_output_range(
    gs200: YokogawaGS200,
    mocker: MockerFixture,
    mode: Literal["VOLT", "CURR"],
    nominal: float,
    limit: float,
    sign: int,
    ramp_mode: Literal["JUMP", "SOFTWARE", "HARDWARE"],
) -> None:
    gs200.source_mode(mode)
    gs200.range(nominal)
    assert gs200.range() == nominal
    gs200.ramp_mode(ramp_mode)
    gs200.ramp_blocking(False)
    if ramp_mode == "HARDWARE":
        gs200.output("on")
    write = mocker.spy(gs200, "write")

    gs200.output_level(sign * limit)

    if ramp_mode == "HARDWARE":
        assert f":SOUR:LEV {sign * limit:E}" in write.call_args.args[0]
    else:
        write.assert_called_once_with(f":SOUR:LEV {sign * limit:.5e}")
        assert gs200.output_level() == sign * limit
    assert gs200.output_level.step is None
    assert gs200.output_level.inter_delay == 0
    assert gs200.ramp_mode() == ramp_mode


@pytest.mark.parametrize("mode,nominal,limit", SOURCE_LIMITS)
@pytest.mark.parametrize("sign", (-1, 1))
@pytest.mark.parametrize("ramp_mode", ("JUMP", "SOFTWARE", "HARDWARE"))
def test_reject_output_beyond_generated_range(
    gs200: YokogawaGS200,
    mocker: MockerFixture,
    mode: Literal["VOLT", "CURR"],
    nominal: float,
    limit: float,
    sign: int,
    ramp_mode: Literal["JUMP", "SOFTWARE", "HARDWARE"],
) -> None:
    gs200.source_mode(mode)
    gs200.range(nominal)
    gs200.ramp_mode(ramp_mode)
    gs200.ramp_blocking(False)
    if ramp_mode == "HARDWARE":
        gs200.output("on")
    write = mocker.spy(gs200, "write")

    with pytest.raises(ValueError, match="Desired output level not in range"):
        gs200.output_level(sign * math.nextafter(limit, math.inf))

    write.assert_not_called()
    assert gs200.output_level.step is None
    assert gs200.output_level.inter_delay == 0
    assert gs200.ramp_mode() == ramp_mode


@pytest.mark.parametrize(
    "mode,nominal,limit", (("VOLT", 0.01, 32.0), ("CURR", 0.001, 0.2))
)
@pytest.mark.parametrize("sign", (-1, 1))
def test_auto_range_output_limit(
    gs200: YokogawaGS200,
    mocker: MockerFixture,
    mode: Literal["VOLT", "CURR"],
    nominal: float,
    limit: float,
    sign: int,
) -> None:
    gs200.source_mode(mode)
    gs200.range(nominal)
    gs200.auto_range(True)
    write = mocker.spy(gs200, "write")

    gs200.output_level(sign * limit)
    write.assert_called_once_with(f":SOUR:LEV:AUTO {sign * limit:.5e}")
    write.reset_mock()
    with pytest.raises(ValueError, match="Desired output level not in range"):
        gs200.output_level(sign * math.nextafter(limit, math.inf))
    write.assert_not_called()


def test_software_ramp_within_generated_range(
    gs200: YokogawaGS200, mocker: MockerFixture
) -> None:
    gs200.voltage_range(10)
    gs200.voltage(9)
    gs200.ramp_step(0.5)
    gs200.ramp_rate(1000)
    gs200.ramp_mode("SOFTWARE")
    write = mocker.spy(gs200, "write")

    gs200.voltage(11)

    assert [call.args[0] for call in write.call_args_list] == [
        f":SOUR:LEV {voltage:.5e}" for voltage in (9.5, 10, 10.5, 11)
    ]
    assert gs200.voltage() == 11
    assert gs200.output_level.step is None
    assert gs200.output_level.inter_delay == 0
    assert gs200.ramp_mode() == "SOFTWARE"


@pytest.mark.parametrize("target", (12.0, -12.0, 12.1, -12.1))
def test_output_limit_uses_refreshed_range(
    gs200: YokogawaGS200, mocker: MockerFixture, target: float
) -> None:
    gs200.voltage_range(1)
    get_range = mocker.patch.object(gs200, "ask", side_effect=("1.0", "10.0"))
    write = mocker.spy(gs200, "write")

    if abs(target) <= 12:
        gs200.voltage(target)
        write.assert_called_once_with(f":SOUR:LEV {target:.5e}")
    else:
        with pytest.raises(ValueError, match="Desired output level not in range"):
            gs200.voltage(target)
        write.assert_not_called()
    assert get_range.call_args_list == [mocker.call(":SOUR:RANG?")] * 2


@pytest.mark.parametrize(
    "parameter,value",
    (
        ("voltage_range", 32.0),
        ("current_range", 0.24),
        ("voltage_limit", 32),
        ("current_limit", 0.24),
    ),
)
def test_nominal_ranges_and_protection_limits_unchanged(
    gs200: YokogawaGS200, mocker: MockerFixture, parameter: str, value: float
) -> None:
    write = mocker.spy(gs200, "write")
    with pytest.raises(ValueError):
        gs200.parameters[parameter](value)
    write.assert_not_called()


def test_hardware_ramp_still_requires_output_enabled(gs200: YokogawaGS200) -> None:
    gs200.voltage_range(10)
    gs200.ramp_mode("HARDWARE")
    with pytest.raises(RuntimeError, match="Need to enable output"):
        gs200.voltage(12)
