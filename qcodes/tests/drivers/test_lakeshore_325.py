import numpy as np
from hypothesis import given
import hypothesis.strategies as st
from qcodes.instrument_drivers.Lakeshore import Model_325

COMPONENTS = sorted(Model_325.Model_325_Sensor.sensor_status_codes.keys())


@given(
    st.lists(st.sampled_from(COMPONENTS[1:]), 1, 5, unique=True).map(sorted)
)
def test_decode_sensor_status(component):
    """
    The sensor status is one of the status codes, or a sum thereof. Multiple
    status are possible as they are not necessarily mutually exclusive.
    The static method '_get_sum_terms' in the Model_325_Sensor class would
    decode the status into status code(s).
    """
    status = sum(component)
    status_codes = Model_325.Model_325_Sensor._get_sum_terms(COMPONENTS,
                                                             status)
    assert np.all(status_codes == component[::-1])


def test_decode_sensor_status_0():
    status_codes = Model_325.Model_325_Sensor._get_sum_terms(COMPONENTS, 0)
    assert np.all(status_codes == [0])
