from unittest.mock import MagicMock, patch

import pytest
from qcodes.instrument_drivers.cryomagnetics.Cryomagnetics4G_visa import Cryomagnetics4GException, CryomagneticsModel4G
from pyvisa import VisaIOError


@pytest.fixture
def cryo_instrument():
    with patch(
        "pyvisa.ResourceManager.open_resource", return_value=MagicMock()
    ) as mock_resource:
        cryo = CryomagneticsModel4G("test_cryo", "GPIB::1::INSTR")
        yield cryo
        cryo.close()

def test_initialization(cryo_instrument):
    assert cryo_instrument.name == "test_cryo"
    assert cryo_instrument.address == "GPIB::1::INSTR"
    assert cryo_instrument.terminator == "\n"


def test_set_and_get_units(cryo_instrument):
    cryo_instrument.units("T")
    assert cryo_instrument.units() == "T"


def test_error_on_quench_condition(cryo_instrument):
    with patch.object(
        cryo_instrument, "ask", return_value="4"
    ):  # Quench condition should raise exception
        with pytest.raises(
            Cryomagnetics4GException, match="Cannot ramp due to quench condition."
        ):
            cryo_instrument.set_field(1.0)


def test_set_field_operation(cryo_instrument):
    with (
        patch.object(cryo_instrument, "write"),
        patch.object(cryo_instrument, "ask", return_value="0"),
        patch.object(cryo_instrument, "_get_field", return_value=0.5),
    ):
        cryo_instrument.set_field(0.1)
        cryo_instrument.write.assert_called_with("SWEEP DOWN")


def test_get_field(cryo_instrument):
    cryo_instrument.units = MagicMock(return_value="T")
    with patch.object(cryo_instrument, "ask", return_value="50.0 kG"):
        assert cryo_instrument.field() == 5.0


def test_set_field_zero(cryo_instrument):
    with patch.object(cryo_instrument, "write") as mock_write:
        cryo_instrument.field(0)
        mock_write.assert_called_with("SWEEP ZERO")


def test_write_raw_retry(cryo_instrument):
    with (
        patch.object(
            cryo_instrument, "write_raw", side_effect=[VisaIOError("Error"), None]
        ),
        patch.object(cryo_instrument, "device_clear") as mock_clear,
    ):
        cryo_instrument.write_raw("test")
        assert cryo_instrument.write_raw.call_count == 2
        mock_clear.assert_called_once()

def test_ask_raw_retry(cryo_instrument):
    with (
        patch.object(
            cryo_instrument, "ask_raw", side_effect=[VisaIOError("Error"), "42"]
        ),
        patch.object(cryo_instrument, "device_clear") as mock_clear,
    ):
        assert cryo_instrument.ask_raw("test") == "42"
        assert cryo_instrument.ask_raw.call_count == 2
        mock_clear.assert_called_once()
