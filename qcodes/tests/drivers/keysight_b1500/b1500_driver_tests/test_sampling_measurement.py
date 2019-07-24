import numpy as np
import pytest

from qcodes.tests.drivers.keysight_b1500.b1500_driver_tests.test_b1500 import b1500


@pytest.fixture
def smu(b1500):
    return b1500.smu1


def test_timing_parameters_is_none_at_init(smu):
    assert smu._timing_parameters['interval'] is None
    assert smu._timing_parameters['number'] is None
    assert smu._timing_parameters['h_bias'] is None
    assert smu._timing_parameters['h_base'] is None


def test_sampling_measurement_requires_timing_parameters_to_be_set(smu):
    with pytest.raises(Exception, match='set timing parameters first'):
        smu.sampling_measurement.get()


def test_sampling_measurement(smu):
    smu.timing_parameters(h_bias=0, interval=0.1, number=2)
    assert isinstance(smu.sampling_measurement.get(), np.ndarray)
