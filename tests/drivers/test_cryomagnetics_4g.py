import io
import logging
from unittest.mock import MagicMock, patch

import pytest

from qcodes.instrument_drivers.cryomagnetics.Cryomagnetics4G_visa import (
    Cryomagnetics4GException,
    CryomagneticsModel4G,
)

# here the original test has a homemade log system that we don't want to
# reproduce / write tests for. Instead, we use normal logging from our
# instrument.visa module
iostream = io.StringIO()
logger = logging.getLogger("qcodes.instrument.visa")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(created)s - %(message)s")
lh = logging.StreamHandler(iostream)
logger.addHandler(lh)
lh.setLevel(logging.DEBUG)
lh.setFormatter(formatter)


@pytest.fixture(name="cryo_instrument", scope="function")
def fixture_cryo_instrument():
    """
    Fixture to create and yield a CryomagneticsModel4G object and close it after testing.
    """
    instrument = CryomagneticsModel4G(
        "test_cryo_4g",
        "GPIB::1::INSTR",
        max_current_limits={0: (0, 0)},
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


def test_magnet_state_standby(cryo_instrument):
    # Test magnet operating state when in standby
    with patch.object(cryo_instrument, "ask", return_value="2"):
        state = cryo_instrument.magnet_operating_state()
        assert state.standby is True
        assert state.ramping is False


def test_magnet_has_quenched(cryo_instrument):
    # Test magnet operating state when in standby
    with patch.object(cryo_instrument, "ask", return_value="4"):
        with pytest.raises(
            Cryomagnetics4GException, match="Cannot ramp due to quench condition."
        ):
            cryo_instrument.magnet_operating_state()


def test_magnet_power_failure(cryo_instrument):
    # Test magnet operating state when in standby
    with patch.object(cryo_instrument, "ask", return_value="8"):
        with pytest.raises(
            Cryomagnetics4GException, match="Cannot ramp due to power module failure."
        ):
            cryo_instrument.magnet_operating_state()


def test_set_field_successful(cryo_instrument):
    # Test setting the field successfully
    with (
        patch.object(cryo_instrument, "write") as mock_write,
        patch.object(cryo_instrument, "ask", return_value="2"),
        patch.object(cryo_instrument, "_get_field", return_value=0),
    ):
        cryo_instrument.set_field(0.1, block=False)
        calls = [
            call
            for call in mock_write.call_args_list
            if "LLIM" in call[0][0] or "SWEEP" in call[0][0]
        ]
        assert any("SWEEP UP" in str(call) for call in calls)


#
