from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import pytest

from qcodes.instrument_drivers.yokogawa import YokogawaGS200
from qcodes.instrument_drivers.yokogawa.Yokogawa_GS200 import (
    ConditionRegister,
    ExtendedEventRegister,
    StandardEventRegister,
    StatusByte,
)

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


@pytest.mark.parametrize("mode", ("VOLT", "CURR"))
def test_range_and_output_level_are_floats(
    gs200: YokogawaGS200, mode: Literal["VOLT", "CURR"]
) -> None:
    gs200.source_mode(mode)
    if mode == "VOLT":
        gs200.voltage_range(1)
        gs200.voltage(0.5)
        source_range, level = gs200.voltage_range, gs200.voltage
    else:
        gs200.current_range(10e-3)
        gs200.current(5e-3)
        source_range, level = gs200.current_range, gs200.current

    for parameter in (source_range, gs200.range, level, gs200.output_level):
        value = parameter.get()
        assert type(value) is float

    assert source_range.get() == gs200.range.get()
    assert level.get() == gs200.output_level.get()


def test_status_registers_return_flags(gs200: YokogawaGS200) -> None:
    status_byte = gs200.status_byte()
    assert isinstance(status_byte, StatusByte)
    assert status_byte == StatusByte.RQS | StatusByte.EAV

    standard_event_register = gs200.standard_event_register()
    assert isinstance(standard_event_register, StandardEventRegister)
    assert standard_event_register == StandardEventRegister.CME

    extended_event_register = gs200.extended_event_register()
    assert isinstance(extended_event_register, ExtendedEventRegister)
    assert (
        extended_event_register == ExtendedEventRegister.EOP | ExtendedEventRegister.SCG
    )

    condition_register = gs200.condition_register()
    assert isinstance(condition_register, ConditionRegister)
    assert condition_register == ConditionRegister.RFP


@pytest.mark.parametrize("keyword,step", (("MIN", 1), ("MAX", 10000)))
def test_program_count_accepts_keywords(
    gs200: YokogawaGS200,
    mocker: MockerFixture,
    keyword: Literal["MIN", "MAX"],
    step: int,
) -> None:
    write = mocker.spy(gs200, "write")

    gs200.program.count(keyword)

    write.assert_called_once_with(f":PROG:COUN {keyword}")
    # The cache holds the keyword until the parameter is read again.
    assert gs200.program.count.cache.get(get_if_invalid=False) == keyword

    # The instrument converts the keyword to step 1 or 10000 (GS200 User's
    # Manual IM GS210-01EN, p. 13-14), but the simulator stores it verbatim,
    # so answer the query as the instrument would.
    ask_raw = mocker.patch.object(gs200, "ask_raw", return_value=str(step))

    value = gs200.program.count.get()

    ask_raw.assert_called_once_with(":PROG:COUN?")
    assert type(value) is int
    assert value == step
    assert gs200.program.count.cache.get(get_if_invalid=False) == step


def test_program_count_returns_int(gs200: YokogawaGS200) -> None:
    gs200.program.count(3)
    value = gs200.program.count.get()
    assert type(value) is int
    assert value == 3


def test_program_save_sends_file_name(
    gs200: YokogawaGS200, mocker: MockerFixture
) -> None:
    write = mocker.patch.object(gs200, "write")

    gs200.program.save("test1.csv")

    write.assert_called_once_with(":PROG:SAVE 'test1.csv'")
    assert gs200.program.save() == "test1.csv"


def test_current_parameters_use_ampere(gs200: YokogawaGS200) -> None:
    assert gs200.current.unit == "A"
    assert gs200.current_range.unit == "A"
    assert gs200.current_limit.unit == "A"


def test_ramp_units_follow_source_mode_at_init(mocker: MockerFixture) -> None:
    # The simulator always starts in VOLT mode, so pretend that the
    # instrument was left in CURR mode before connecting.
    original_ask_raw = YokogawaGS200.ask_raw

    def ask_raw(self: YokogawaGS200, cmd: str) -> str:
        if cmd == ":SOUR:FUNC?":
            return "CURR"
        return original_ask_raw(self, cmd)

    mocker.patch.object(YokogawaGS200, "ask_raw", ask_raw)
    gs200 = YokogawaGS200(
        "GS200_curr", address="GPIB0::1::INSTR", pyvisa_sim_file="Yokogawa_GS200.yaml"
    )
    try:
        assert gs200.source_mode() == "CURR"
        assert gs200.ramp_rate.unit == "A/s"
        assert gs200.ramp_step.unit == "A"
    finally:
        gs200.close()


def test_hardware_ramp_still_requires_output_enabled(gs200: YokogawaGS200) -> None:
    gs200.voltage_range(10)
    gs200.ramp_mode("HARDWARE")
    with pytest.raises(RuntimeError, match="Need to enable output"):
        gs200.voltage(12)
