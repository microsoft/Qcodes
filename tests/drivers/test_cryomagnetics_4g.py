import logging
from unittest.mock import ANY, MagicMock, call, patch

import pytest

from qcodes.instrument_drivers.cryomagnetics import (
    Cryomagnetics4GException,
    CryomagneticsModel4G,
    CryomagneticsOperatingState,
)


@pytest.fixture(name="cryo_instrument", scope="function")
def fixture_cryo_instrument():
    """
    Fixture to create and yield a CryomagneticsModel4G object and close it after testing.
    """
    instrument = CryomagneticsModel4G(
        "test_cryo_4g",
        "GPIB::1::INSTR",
        max_current_limits={0: (0.0, 0.0)},
        coil_constant=10.0,
        pyvisa_sim_file="cryo4g.yaml",
    )
    yield instrument
    instrument.close()


def test_initialization(cryo_instrument):
    assert cryo_instrument.name == "test_cryo_4g"
    assert cryo_instrument._address == "GPIB::1::INSTR"
    # assert cryo_instrument.terminator == "\n"


def test_get_field(cryo_instrument):
    cryo_instrument.units = MagicMock(return_value="T")
    cryo_instrument.field(5.0)
    assert cryo_instrument.field() == 5.0


def test_initialization_visa_sim(cryo_instrument):
    # Test to ensure correct initialization of the CryomagneticsModel4G instrument
    assert cryo_instrument.name == "test_cryo_4g"
    assert cryo_instrument._address == "GPIB::1::INSTR"


@pytest.mark.parametrize(
    "status_byte, expected_state, expected_exception, expected_log_message",
    [
        ("0", CryomagneticsOperatingState(holding=True), None, None),
        (
            "1",
            None,
            Cryomagnetics4GException,
            "Cannot ramp as the power supply is already ramping.",
        ),
        ("2", CryomagneticsOperatingState(standby=True), None, None),
        ("4", None, Cryomagnetics4GException, "Cannot ramp due to quench condition."),
        (
            "8",
            None,
            Cryomagnetics4GException,
            "Cannot ramp due to power module failure.",
        ),
    ],
)
def test_magnet_operating_state(
    cryo_instrument,
    caplog,
    status_byte,
    expected_state,
    expected_exception,
    expected_log_message,
):
    with patch.object(cryo_instrument, "ask", return_value=status_byte):
        if expected_exception:
            with pytest.raises(expected_exception, match=expected_log_message):
                cryo_instrument.magnet_operating_state()
            assert expected_log_message in caplog.text
        else:
            state = cryo_instrument.magnet_operating_state()
            assert state == expected_state


def test_set_field_successful(cryo_instrument, caplog):
    with (
        patch.object(cryo_instrument, "write") as mock_write,
    ):
        with caplog.at_level(logging.WARNING):
            cryo_instrument.set_field(0.1, block=False)
            calls = [
                call
                for call in mock_write.call_args_list
                if "LLIM" in call[0][0] or "SWEEP" in call[0][0]
            ]
            assert any("SWEEP UP" in str(call) for call in calls)
            assert "Magnetic field is ramping but not currently blocked!" in caplog.text


def test_set_field_blocking(cryo_instrument):
    with (
        patch.object(cryo_instrument, "write") as mock_write,
        patch.object(
            cryo_instrument,
            "wait_while_ramping",
            return_value=CryomagneticsOperatingState(holding=True, ramping=False),
        ) as mock_wait,
    ):
        cryo_instrument.set_field(0.5, block=True)

        # Check that the correct commands were sent
        calls = [
            call
            for call in mock_write.call_args_list
            if "LLIM" in str(call) or "ULIM" in str(call) or "SWEEP" in str(call)
        ]
        assert any("ULIM 5.0" in str(call) for call in calls)
        assert any("SWEEP UP" in str(call) for call in calls)

        # Ensure wait_while_ramping was called with the correct setpoint
        mock_wait.assert_called_once_with(0.5, threshold=ANY)


def test_wait_while_ramping_timeout(cryo_instrument):
    # Simulate _get_field always returning a value far from the setpoint
    with (
        patch.object(cryo_instrument, "_get_field", return_value=0.0),
        patch.object(cryo_instrument, "_sleep"),
    ):
        with pytest.raises(Cryomagnetics4GException, match=r"Timeout|stabilized"):
            cryo_instrument.wait_while_ramping(1.0, threshold=1e-4)


def test_wait_while_ramping_success(cryo_instrument):
    # Simulate _get_field returning values that reach the setpoint
    with (
        patch.object(cryo_instrument, "_sleep"),
        patch.object(
            cryo_instrument,
            "magnet_operating_state",
            return_value=CryomagneticsOperatingState(holding=True, ramping=False),
        ),
    ):
        state = cryo_instrument.wait_while_ramping(0.5, threshold=1e-2)
        assert state.holding is True
        assert state.ramping is False


def test_get_rate(cryo_instrument):
    with patch.object(cryo_instrument, "ask", return_value="5.0"):
        assert cryo_instrument._get_rate() == 5.0 * 60 * cryo_instrument.coil_constant


def test_set_rate(cryo_instrument):
    # Define the max_current_limits dictionary for testing
    cryo_instrument.max_current_limits = {
        0: (10.0, 1.0),  # Range 0: up to 10 A, max rate 1 A/s
        1: (50.0, 2.0),  # Range 1: up to 50 A, max rate 2 A/s
        2: (70.0, 0.001),  # Range 2: up to 70 A, max rate 0.001 A/s
    }

    with (
        patch.object(cryo_instrument, "write") as mock_write,
    ):
        # _set_rate() converts T/min to A/s for all ranges
        cryo_instrument._set_rate(1.0)
        expected_rate_0 = min(
            1.0 / cryo_instrument.coil_constant / 60,
            cryo_instrument.max_current_limits[0][1],
        )
        expected_rate_1 = min(
            1.0 / cryo_instrument.coil_constant / 60,
            cryo_instrument.max_current_limits[1][1],
        )
        expected_rate_2 = min(
            1.0 / cryo_instrument.coil_constant / 60,
            cryo_instrument.max_current_limits[2][1],
        )

        assert mock_write.call_args_list == [
            call(f"RATE 0 {expected_rate_0}"),
            call(f"RATE 1 {expected_rate_1}"),
            call(f"RATE 2 {expected_rate_2}"),
        ]


def test_initialize_max_current_limits(cryo_instrument):
    with patch.object(cryo_instrument, "write") as mock_write:
        cryo_instrument._initialize_max_current_limits()
        calls = [
            call(f"RANGE {range_index} {upper_limit}")
            for range_index, (
                upper_limit,
                _,
            ) in cryo_instrument.max_current_limits.items()
        ] + [
            call(f"RATE {range_index} {max_rate}")
            for range_index, (_, max_rate) in cryo_instrument.max_current_limits.items()
        ]
        assert mock_write.call_args_list == calls
