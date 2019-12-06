from hypothesis import given
import hypothesis.strategies as st
from qcodes.instrument_drivers.Lakeshore import Model_325

SENSOR_STATUS_CODES = {
        1:  "invalid reading",
        16:  "temp underrange",
        32:  "temp overrange",
        64:  "sensor units zero",
        128: "sensor units overrang"
    }


@given(
    st.lists(st.sampled_from(list(SENSOR_STATUS_CODES.keys())),
             1, 5, unique=True).map(sorted)
)
def test_decode_sensor_status(list_of_codes):
    """
    The sensor status is one of the status codes, or a sum thereof. Multiple
    status are possible as they are not necessarily mutually exclusive.
    The static method 'decode_sensor_status' in the Model_325_Sensor class can
    decode the status into status code(s).
    """
    sum_of_codes = sum(list_of_codes)
    codes = ', '.join(SENSOR_STATUS_CODES[i] for i in list_of_codes[::-1])
    status_codes = Model_325.Model_325_Sensor.decode_sensor_status(sum_of_codes)
    assert codes == status_codes


def test_decode_sensor_status_0():
    status_codes = Model_325.Model_325_Sensor.decode_sensor_status(0)
    assert status_codes == 'OK'
