import pytest

from qcodes.instrument_drivers.Lakeshore.Model_336 import (
    Model_336,  # pyright: ignore[reportDeprecated]
)


@pytest.fixture(scope="function")
def lakeshore_336():
    inst = Model_336(  # type: ignore
        "lakeshore_336_fixture",
        "GPIB::2::INSTR",
        pyvisa_sim_file="lakeshore_model336.yaml",
        device_clear=False,
    )
    try:
        yield inst
    finally:
        inst.close()


def test_pid_set(lakeshore_336) -> None:
    ls = lakeshore_336
    P, I, D = 1, 2, 3  # noqa  E741
    # Only current source outputs/heaters have PID parameters,
    # voltages source outputs/heaters do not.
    outputs = [ls.output_1, ls.output_2]
    for h in outputs:  # a.k.a. heaters
        h.P(P)
        h.I(I)
        h.D(D)
        assert (h.P(), h.I(), h.D()) == (P, I, D)


def test_output_mode(lakeshore_336) -> None:
    ls = lakeshore_336
    mode = "off"
    input_channel = "A"
    powerup_enable = True
    outputs = [getattr(ls, f"output_{n}") for n in range(1, 5)]
    for h in outputs:  # a.k.a. heaters
        h.mode(mode)
        h.input_channel(input_channel)
        h.powerup_enable(powerup_enable)
        assert h.mode() == mode
        assert h.input_channel() == input_channel
        assert h.powerup_enable() == powerup_enable


def test_range(lakeshore_336) -> None:
    ls = lakeshore_336
    output_range = "medium"
    outputs = [getattr(ls, f"output_{n}") for n in range(1, 5)]
    for h in outputs:  # a.k.a. heaters
        h.output_range(output_range)
        assert h.output_range() == output_range


def test_tlimit(lakeshore_336) -> None:
    ls = lakeshore_336
    tlimit = 5.1
    for ch in ls.channels:
        ch.t_limit(tlimit)
        assert ch.t_limit() == tlimit


def test_setpoint(lakeshore_336) -> None:
    ls = lakeshore_336
    setpoint = 5.1
    outputs = [getattr(ls, f"output_{n}") for n in range(1, 5)]
    for h in outputs:  # a.k.a. heaters
        h.setpoint(setpoint)
        assert h.setpoint() == setpoint


def test_select_range_limits(lakeshore_336) -> None:
    h = lakeshore_336.output_1
    ranges = [1, 2, 3]
    h.range_limits(ranges)

    for i in ranges:
        h.set_range_from_temperature(i - 0.5)
        assert h.output_range() == h.INVERSE_RANGES[i]

    i = 3
    h.set_range_from_temperature(i + 0.5)
    assert h.output_range() == h.INVERSE_RANGES[len(ranges)]


def test_set_and_wait_unit_setpoint_reached(lakeshore_336) -> None:
    ls = lakeshore_336
    ls.output_1.setpoint(4)
    ls.output_1.wait_until_set_point_reached()


def test_blocking_t(lakeshore_336) -> None:
    ls = lakeshore_336
    h = ls.output_1
    ranges = [1.2, 2.4, 3.1]
    h.range_limits(ranges)
    h.blocking_t(4)
