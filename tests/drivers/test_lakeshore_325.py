from typing import TYPE_CHECKING

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

from qcodes.instrument_drivers.Lakeshore import (
    LakeshoreModel325,
    LakeshoreModel325Sensor,
    LakeshoreModel325Status,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from pytest_mock import MockerFixture


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


def test_curve_index(lakeshore_325: LakeshoreModel325) -> None:
    sensor_a = lakeshore_325.sensor_A
    assert sensor_a.curve_index() == 2

    sensor_a.curve_index(23)
    assert sensor_a.curve_index() == 23

    # 0 means no curve, which the instrument also falls back to when the
    # curve does not match the input
    assert lakeshore_325.sensor_B.curve_index() == 0
    sensor_a.curve_index(0)
    assert sensor_a.curve_index() == 0


@pytest.mark.parametrize("value", [-1, 36, 2.5])
def test_curve_index_rejects_invalid(
    lakeshore_325: LakeshoreModel325, value: float
) -> None:
    with pytest.raises((TypeError, ValueError)):
        lakeshore_325.sensor_A.curve_index(value)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("value", [23.0, np.float64(23)], ids=["float", "numpy_float"])
def test_curve_index_accepts_whole_floats(
    lakeshore_325: LakeshoreModel325, mocker: "MockerFixture", value: float
) -> None:
    # whole-number floats, e.g. from numpy, are accepted and sent as int
    sensor_a = lakeshore_325.sensor_A
    write_raw = mocker.spy(lakeshore_325, "write_raw")

    sensor_a.curve_index(value)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]

    write_raw.assert_called_once_with("INCRV A, 23")
    assert sensor_a.curve_index() == 23


def test_sensor_curve(lakeshore_325: LakeshoreModel325) -> None:
    sensor_a = lakeshore_325.sensor_A
    assert sensor_a.curve.short_name == "curve_2"

    sensor_a.curve_index(23)
    assert sensor_a.curve.short_name == "curve_23"


def test_sensor_curve_none_selected(lakeshore_325: LakeshoreModel325) -> None:
    # curve index 0 means no curve; there is no curve 0 to query
    assert lakeshore_325.sensor_B.curve_index() == 0
    with pytest.raises(RuntimeError, match="No curve is selected"):
        _ = lakeshore_325.sensor_B.curve


def _fake_curve_ask(cmd: str) -> str:
    # a header with the V/K format and a point n with units n/1000 at n K
    if cmd.startswith("CRVHDR?"):
        return "DT-470,00011134,2,325.0,1"
    if cmd.startswith("CRVPT?"):
        point = int(cmd.split(",")[1])
        return f"{point / 1000},{float(point)}"
    raise ValueError(f"unexpected query {cmd}")


def test_curves_include_user_curve_35(
    lakeshore_325: LakeshoreModel325, mocker: "MockerFixture"
) -> None:
    # curves 1-20 are standard and 21-35 user curves
    assert len(lakeshore_325.curve) == 35
    assert lakeshore_325.curve[-1].short_name == "curve_35"

    mocker.patch.object(lakeshore_325, "ask", side_effect=_fake_curve_ask)
    write = mocker.patch.object(lakeshore_325, "write")
    lakeshore_325.upload_curve(
        35,
        name="DT-470",
        serial_number="00011134",
        data_dict={"Temperature (K)": [470.0], "V": [0.10191]},
    )
    assert write.call_args_list[-1] == mocker.call("CRVPT 35, 1, 0.10191, 470")


def test_curve_get_data_reads_200_points(
    lakeshore_325: LakeshoreModel325, mocker: "MockerFixture"
) -> None:
    ask = mocker.patch.object(lakeshore_325, "ask", side_effect=_fake_curve_ask)

    data = lakeshore_325.curve[20].get_data()

    assert data["Temperature (K)"] == [float(point) for point in range(1, 201)]
    assert data["V"] == [point / 1000 for point in range(1, 201)]
    ask.assert_any_call("CRVPT? 21, 200")


def test_curve_set_data_uses_6_digits(
    lakeshore_325: LakeshoreModel325, mocker: "MockerFixture"
) -> None:
    # the instrument stores curve points to 6 digits
    write = mocker.patch.object(lakeshore_325, "write")

    lakeshore_325.curve[20].set_data(
        {"Temperature (K)": [470.0, 1.4], "V": [0.10191, 1.70333]}
    )

    assert write.call_args_list == [
        mocker.call("CRVPT 21, 1, 0.10191, 470"),
        mocker.call("CRVPT 21, 2, 1.70333, 1.4"),
    ]


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
