from unittest.mock import Mock

import numpy as np
import pytest
from qcodes.instrument_drivers.Keysight.keysightb1500 import constants


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


@pytest.fixture
def smu_sampling_measurement_output():
    n_samples = 7
    data_to_return = np.random.rand(n_samples)
    return n_samples, data_to_return


@pytest.fixture
def smu_sampling_measurement(smu, smu_sampling_measurement_output):
    data_to_return = smu_sampling_measurement_output[1]
    STATUS = 'N'
    CHANNEL = 'A'
    TYPE = 'I'
    prefix = f'{STATUS}{CHANNEL}{TYPE}'
    visa_data_response = ','.join([prefix + f'{d:+012.3E}' for d in data_to_return])
    smu_sm = smu
    original_ask = smu_sm.root_instrument.ask

    def return_predefined_data_on_xe(cmd: str) -> str:
        if cmd == 'XE':
            return visa_data_response
        else:
            return original_ask(cmd)

    smu_sm.root_instrument.ask = Mock(spec_set=smu.root_instrument.ask)
    smu_sm.root_instrument.ask.side_effect = return_predefined_data_on_xe
    return smu_sm, STATUS, CHANNEL, TYPE


def test_sampling_measurement(smu_sampling_measurement, smu_sampling_measurement_output):
    N_SAMPLES = smu_sampling_measurement_output[0]
    data_to_return = smu_sampling_measurement_output[1]
    smu_sampling_measurement.timing_parameters(h_bias=0, interval=0.1, number=N_SAMPLES)
    actual_data = smu_sampling_measurement.sampling_measurement.get()

    np.testing.assert_allclose(actual_data, data_to_return, atol=1e-3)
    smu_sampling_measurement.root_instrument.ask.assert_called_once_with('XE')


def test_sampling_measurement_compliance_requires_data_from_samplingmeasurement(smu):
    with pytest.raises(Exception, match='First run sampling_measurement method to generate the data'):
        smu.sampling_measurement.compliance()


def test_sampling_measurement_compliance(smu_sampling_measurement, smu_sampling_measurement_output):
    n_samples = smu_sampling_measurement_output[0]
    smu_sampling_measurement, STATUS, CHANNEL, TYPE = smu_sampling_measurement
    smu_sampling_measurement.timing_parameters(h_bias=0, interval=0.1, number=n_samples)
    smu_sampling_measurement.sampling_measurement.get()
    compliance_list_string = [STATUS]*n_samples
    compliance_list = [constants.ComplianceErrorList[i[0]].value for i in compliance_list_string]

    assert isinstance(smu_sampling_measurement.sampling_measurement.compliance(), list)
    np.testing.assert_array_equal(smu_sampling_measurement.sampling_measurement.compliance(), compliance_list)


def test_sampling_measurement_data(smu_sampling_measurement, smu_sampling_measurement_output):
    n_samples = smu_sampling_measurement_output[0]
    smu_sampling_measurement, STATUS, CHANNEL, TYPE = smu_sampling_measurement
    smu_sampling_measurement.timing_parameters(h_bias=0, interval=0.1, number=n_samples)
    smu_sampling_measurement.sampling_measurement.get()

    expected_channel_output = [CHANNEL] * n_samples
    expected_channel_output = [constants.ChannelName[i].value for i in expected_channel_output]
    expected_type_output = [TYPE] * n_samples

    data_type = smu_sampling_measurement.sampling_measurement.data.type
    data_channel = smu_sampling_measurement.sampling_measurement.data.channel
    np.testing.assert_array_equal(data_type, expected_type_output)
    np.testing.assert_array_equal(data_channel, expected_channel_output)
