from __future__ import annotations

from typing import Literal, TypeVar

import pytest
from typing_extensions import ParamSpec

from qcodes.instrument_drivers.Lakeshore import LakeshoreModel372
from qcodes.instrument_drivers.Lakeshore.lakeshore_base import (
    LakeshoreBaseSensorChannel,
)

P = ParamSpec("P")
T = TypeVar("T")


def instrument_fixture(
    scope: Literal["session", "package", "module", "class", "function"] = "function",
    name=None,
):
    def wrapper(func):
        @pytest.fixture(scope=scope, name=name)
        def wrapped_fixture():
            inst = func()
            try:
                yield inst
            finally:
                inst.close()

        return wrapped_fixture

    return wrapper


@instrument_fixture(scope="function")
def lakeshore_372():
    return LakeshoreModel372(
        "lakeshore_372_fixture",
        "GPIB::3::INSTR",
        pyvisa_sim_file="lakeshore_model372.yaml",
        device_clear=False,
    )


def test_pid_set(lakeshore_372) -> None:
    ls = lakeshore_372
    P, I, D = 1, 2, 3  # noqa  E741
    for h in (ls.warmup_heater, ls.analog_heater, ls.sample_heater):
        h.P(P)
        h.I(I)
        h.D(D)
        assert (h.P(), h.I(), h.D()) == (P, I, D)


def test_output_mode(lakeshore_372) -> None:
    ls = lakeshore_372
    mode = "off"
    input_channel = 1
    powerup_enable = True
    polarity = "unipolar"
    use_filter = True
    delay = 1
    for h in (ls.warmup_heater, ls.analog_heater, ls.sample_heater):
        h.mode(mode)
        h.input_channel(input_channel)
        h.powerup_enable(powerup_enable)
        h.polarity(polarity)
        h.use_filter(use_filter)
        h.delay(delay)
        assert h.mode() == mode
        assert h.input_channel() == input_channel
        assert h.powerup_enable() == powerup_enable
        assert h.polarity() == polarity
        assert h.use_filter() == use_filter
        assert h.delay() == delay


def test_range(lakeshore_372) -> None:
    ls = lakeshore_372
    output_range = "10mA"
    for h in (ls.warmup_heater, ls.analog_heater, ls.sample_heater):
        h.output_range(output_range)
        assert h.output_range() == output_range


def test_tlimit(lakeshore_372) -> None:
    ls = lakeshore_372
    tlimit = 5.1
    for ch in ls.channels:
        ch.t_limit(tlimit)
        assert ch.t_limit() == tlimit


def test_setpoint(lakeshore_372) -> None:
    ls = lakeshore_372
    setpoint = 5.1
    for h in (ls.warmup_heater, ls.analog_heater, ls.sample_heater):
        h.setpoint(setpoint)
        assert h.setpoint() == setpoint


def test_select_range_limits(lakeshore_372) -> None:
    h = lakeshore_372.sample_heater
    ranges = list(range(1, 9))
    h.range_limits(ranges)

    for i in ranges:
        h.set_range_from_temperature(i - 0.5)
        assert h.output_range() == h.INVERSE_RANGES[i]

    h.set_range_from_temperature(ranges[-1] + 0.5)
    assert h.output_range() == h.INVERSE_RANGES[len(ranges)]


def test_set_and_wait_unit_setpoint_reached(lakeshore_372) -> None:
    ls = lakeshore_372
    ls.sample_heater.setpoint(4)
    ls.sample_heater.wait_until_set_point_reached()


def test_blocking_t(lakeshore_372) -> None:
    h = lakeshore_372.sample_heater
    ranges = list(range(1, 9))
    h.range_limits(ranges)
    h.blocking_t(4)


def test_get_term_sum() -> None:
    available_terms = [0, 1, 2, 4, 8, 16, 32]

    assert [32, 8, 2, 1] == LakeshoreBaseSensorChannel._get_sum_terms(
        available_terms, 1 + 2 + 8 + 32
    )

    assert [32] == LakeshoreBaseSensorChannel._get_sum_terms(available_terms, 32)

    assert [16, 4, 1] == LakeshoreBaseSensorChannel._get_sum_terms(
        available_terms, 1 + 4 + 16
    )

    assert [0] == LakeshoreBaseSensorChannel._get_sum_terms(available_terms, 0)


def test_get_term_sum_with_some_powers_of_2_omitted() -> None:
    available_terms = [0, 16, 32]

    assert [32, 16] == LakeshoreBaseSensorChannel._get_sum_terms(
        available_terms, 16 + 32
    )

    assert [32] == LakeshoreBaseSensorChannel._get_sum_terms(available_terms, 32)

    assert [0] == LakeshoreBaseSensorChannel._get_sum_terms(available_terms, 0)


def test_get_term_sum_returns_empty_list() -> None:
    available_terms = [0, 16, 32]

    assert [] == LakeshoreBaseSensorChannel._get_sum_terms(available_terms, 15)


def test_get_term_sum_when_zero_is_not_in_available_terms() -> None:
    available_terms = [16, 32]

    assert [] == LakeshoreBaseSensorChannel._get_sum_terms(available_terms, 3)

    # Note that `_get_sum_terms` expects '0' to be in the available_terms,
    # hence for this particular case it will still return a list with '0' in
    # it although that '0' is not part of the available_terms
    assert [0] == LakeshoreBaseSensorChannel._get_sum_terms(available_terms, 0)
