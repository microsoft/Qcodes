import logging
from unittest.mock import MagicMock, Mock, call, patch

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
    with patch.object(cryo_instrument, "ask", return_value="50.0 kG"):
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
        patch.object(cryo_instrument, "ask", return_value="2"),
        patch.object(cryo_instrument, "_get_field", return_value=0),
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
        patch.object(cryo_instrument, "_get_field", return_value=0),
    ):
        # Create a mock for the ask method
        mock_ask = Mock(side_effect=lambda x: "0" if x == "*STB?" else "")
        cryo_instrument.ask = mock_ask

        cryo_instrument.set_field(0.5, block=True)

        # Check that the ask method was called with the expected arguments
        mock_ask.assert_any_call("*STB?")

        calls = [
            call
            for call in mock_write.call_args_list
            if "LLIM" in str(call) or "ULIM" in str(call) or "SWEEP" in str(call)
        ]
        assert any("ULIM 5.0" in str(call) for call in calls)
        assert any("SWEEP UP" in str(call) for call in calls)


def test_wait_while_ramping(cryo_instrument):
    # Create a mock for the ask method
    mock_ask = Mock(side_effect=lambda x: "0" if x == "*STB?" else "")
    cryo_instrument.ask = mock_ask

    state = cryo_instrument.wait_while_ramping(0.5)

    # Check that the ask method was called with the expected arguments
    mock_ask.assert_any_call("*STB?")

    assert state.holding is True
    assert state.ramping is False


def test_get_rate(cryo_instrument):
    with patch.object(cryo_instrument, "ask", return_value="5.0"):
        assert cryo_instrument._get_rate() == 5.0 * 60 / cryo_instrument.coil_constant


def test_set_rate(cryo_instrument):
    # Define the max_current_limits dictionary for testing
    cryo_instrument.max_current_limits = {
        0: (10.0, 1.0),  # Range 0: up to 10 A, max rate 1 A/s
        1: (50.0, 2.0),  # Range 1: up to 50 A, max rate 2 A/s
    }

    with (
        patch.object(cryo_instrument, "write") as mock_write,
        patch.object(cryo_instrument, "_get_field", return_value=0.5),
    ):
        cryo_instrument._set_rate(1.0)
        expected_rate = min(
            1.0 * cryo_instrument.coil_constant / 60,
            cryo_instrument.max_current_limits[0][1],
        )
        assert mock_write.call_args_list == [call(f"RATE 0 {expected_rate}")]


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
