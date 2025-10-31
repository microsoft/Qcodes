import pytest

from qcodes.instrument_drivers.Lakeshore import LakeshoreModel336


@pytest.fixture(scope="function", name="lakeshore_336")
def _make_lakeshore_336():
    """Create a Lakeshore 336 instance using PyVISA-sim backend."""
    inst = LakeshoreModel336(
        "lakeshore_336",
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


@pytest.mark.parametrize("output_num", [1, 2, 3, 4])
@pytest.mark.parametrize("mode", ["off", "closed_loop", "zone", "open_loop"])
@pytest.mark.parametrize("input_channel", ["A", "B", "C", "D"])
def test_output_mode(lakeshore_336, output_num, mode, input_channel) -> None:
    ls = lakeshore_336
    mode = "off"
    h = getattr(ls, f"output_{output_num}")
    h.mode(mode)
    h.input_channel(input_channel)
    h.powerup_enable(True)
    assert h.mode() == mode
    assert h.input_channel() == input_channel
    assert h.powerup_enable()


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


def test_curve_parameters(lakeshore_336) -> None:
    # The curve numbers are assigned in the simulation pyvisa sim
    # YAML file for each sensor/channel, and properties of the
    # curves also include curve number in them to help testing
    for ch, curve_number in zip(lakeshore_336.channels, (42, 41, 40, 39)):
        assert ch.curve_number() == curve_number
        assert ch.curve_name().endswith(str(curve_number))
        assert ch.curve_sn().endswith(str(curve_number))
        assert ch.curve_format() == "V/K"
        assert str(int(ch.curve_limit())).endswith(str(curve_number))
        assert ch.curve_coefficient() == "negative"


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
    """Test that wait_until_set_point_reached completes in simulation mode."""
    ls = lakeshore_336
    ls.output_1.setpoint(4)
    # In simulation mode, wait_until_set_point_reached should return immediately
    # because _is_simulated check bypasses the wait loop
    ls.output_1.wait_until_set_point_reached()


def test_blocking_t(lakeshore_336) -> None:
    """Test that blocking_t completes in simulation mode."""
    ls = lakeshore_336
    h = ls.output_1
    ranges = [1.2, 2.4, 3.1]
    h.range_limits(ranges)
    # In simulation mode, blocking_t should return immediately
    # because _is_simulated check bypasses the wait loop
    h.blocking_t(4)
