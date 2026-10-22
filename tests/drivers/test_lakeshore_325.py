from typing import TYPE_CHECKING

import hypothesis.strategies as st
import pytest
from hypothesis import given

from qcodes.instrument_drivers.Lakeshore import (
    LakeshoreModel325,
    LakeshoreModel325Sensor,
    LakeshoreModel325Status,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(scope="function", name="lakeshore_325")
def _make_lakeshore_325() -> "Generator[LakeshoreModel325, None, None]":
    driver = LakeshoreModel325(
        "lakeshore_325",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="lakeshore_model325.yaml",
    )
    try:
        yield driver
    finally:
        driver.close()


def test_output_range(lakeshore_325: LakeshoreModel325) -> None:
    heater_1 = lakeshore_325.heater_1
    heater_2 = lakeshore_325.heater_2

    assert heater_1.output_range() == "Off"
    heater_1.output_range("High (25W)")
    assert heater_1.output_range() == "High (25W)"

    heater_2.output_range("Low (2.5W)")
    assert heater_2.output_range() == "Low (2.5W)"


def test_high_output_range_only_on_loop_1(lakeshore_325: LakeshoreModel325) -> None:
    with pytest.raises(ValueError, match="High \\(25W\\)"):
        lakeshore_325.heater_2.output_range("High (25W)")
    assert lakeshore_325.heater_2.output_range() == "Off"


@given(
    st.lists(
        st.sampled_from(list(LakeshoreModel325Status)),
        min_size=1,
        max_size=5,
        unique=True,
    ).map(
        sorted  # type: ignore[arg-type]
    )
)
def test_decode_sensor_status(list_of_codes: list[LakeshoreModel325Status]) -> None:
    """
    The sensor status is one of the status codes, or a sum thereof. Multiple
    status are possible as they are not necessarily mutually exclusive.
    The static method 'decode_sensor_status' in the Model_325_Sensor class can
    decode the status into status code(s).
    """
    codes = []
    for code in list_of_codes[::-1]:
        assert code.name is not None
        codes.append(code.name.replace("_", " "))
    codes_message = ", ".join(codes)
    sum_of_codes = int(sum(list_of_codes))
    status_messages = LakeshoreModel325Sensor.decode_sensor_status(sum_of_codes)
    assert codes_message == status_messages


def test_decode_sensor_status_0() -> None:
    status_codes = LakeshoreModel325Sensor.decode_sensor_status(0)
    assert status_codes == "OK"
