from unittest.mock import patch

import pytest

from qcodes.instrument_drivers.cryomagnetics import CryomagneticsModelTM620


@pytest.fixture(name="tm620", scope="function")
def fixture_tm620():
    """
    Fixture to create and yield a CryomagneticsModelTM620 object and close it after testing.
    """
    instrument = CryomagneticsModelTM620(
        name="test_cryo_tm620",
        address="GPIB::2::INSTR",
        terminator="\r\n",
        pyvisa_sim_file="cryo_tm620.yaml",
    )
    yield instrument
    instrument.close()


def test_initialization(tm620):
    assert tm620.name == "test_cryo_tm620"
    assert tm620._address == "GPIB::2::INSTR"
    assert hasattr(tm620, "shield")
    assert hasattr(tm620, "magnet")


def test_get_A_success(tm620):
    with patch.object(tm620, "ask", return_value="55.12K"):
        assert tm620._get_A() == 55.12


def test_get_B_success(tm620):
    with patch.object(tm620, "ask", return_value="4.21K"):
        assert tm620._get_B() == 4.21


def test_parse_output_valid(tm620):
    assert tm620._parse_output("55.12K") == "55.12"
    assert tm620._parse_output("4.34C") == "4.34"


def test_parse_output_invalid(tm620, caplog):
    with caplog.at_level("ERROR"):
        with pytest.raises(ValueError, match="No floating point number found"):
            tm620._parse_output("Invalid output")
    assert "No floating point number found in output" in caplog.text


def test_convert_to_numerics_valid(tm620):
    assert tm620._convert_to_numeric("55.12") == 55.12
    assert tm620._convert_to_numeric("4.34") == 4.34


def test_convert_to_numerics_invalid(tm620, caplog):
    with caplog.at_level("ERROR"):
        with pytest.raises(ValueError, match="Unable to convert"):
            tm620._convert_to_numeric("not_a_number")
    assert "Error converting 'not_a_number' to float" in caplog.text
