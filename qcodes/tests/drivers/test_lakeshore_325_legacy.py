import hypothesis.strategies as st
from hypothesis import given

from qcodes.instrument_drivers.Lakeshore import Model_325

STATUS = Model_325.Status  # type: ignore[attr-defined]


@given(
    st.lists(st.sampled_from(list(STATUS)), min_size=1, max_size=5, unique=True).map(
        sorted
    )
)
def test_decode_sensor_status(list_of_codes) -> None:
    """
    The sensor status is one of the status codes, or a sum thereof. Multiple
    status are possible as they are not necessarily mutually exclusive.
    The static method 'decode_sensor_status' in the Model_325_Sensor class can
    decode the status into status code(s).
    """
    codes = [code.name.replace("_", " ") for code in list_of_codes[::-1]]
    codes_message = ", ".join(codes)
    sum_of_codes = int(sum(list_of_codes))
    status_messages = Model_325.Model_325_Sensor.decode_sensor_status(sum_of_codes)  # type: ignore[attr-defined]
    assert codes_message == status_messages


def test_decode_sensor_status_0() -> None:
    status_codes = Model_325.Model_325_Sensor.decode_sensor_status(0)  # type: ignore[attr-defined]
    assert status_codes == "OK"
