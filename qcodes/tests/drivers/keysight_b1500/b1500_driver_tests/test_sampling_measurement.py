from unittest.mock import Mock

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
    N_SAMPLES = 7
    data_to_return = np.random.rand(N_SAMPLES)
    STATUS = 'N'
    CHANNEL = 'A'
    TYPE = 'I'
    prefix = f'{STATUS}{CHANNEL}{TYPE}'
    visa_data_response = ','.join([prefix + f'{d:+012.3E}' for d in data_to_return])

    original_ask = smu.root_instrument.ask

    def return_predefined_data_on_xe(cmd: str) -> str:
        if cmd == 'XE':
            return visa_data_response
        else:
            return original_ask(cmd)

    smu.root_instrument.ask = Mock(spec_set=smu.root_instrument.ask)
    smu.root_instrument.ask.side_effect = return_predefined_data_on_xe

    smu.timing_parameters(h_bias=0, interval=0.1, number=N_SAMPLES)
    actual_data = smu.sampling_measurement.get()

    np.testing.assert_allclose(actual_data, data_to_return, atol=1e-3)

    smu.root_instrument.ask.assert_called_once_with('XE')
