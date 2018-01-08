import pytest

from hypothesis import given
import hypothesis.strategies as hst

import qcodes as qc
from qcodes.dataset.measurements import Measurement
from qcodes.tests.instrument_mocks import DummyInstrument


@pytest.fixture  # scope is "function" per default
def DAC():
    dac = DummyInstrument('dummy_dac', gates=['ch1', 'ch2'])
    yield dac
    dac.close()


@pytest.fixture
def DMM():
    dmm = DummyInstrument('dummy_dmm', gates=['v1', 'v2'])
    yield dmm
    dmm.close()


def test_register_parameter(DAC, DMM):

    parameters = [DAC.ch1, DAC.ch2, DMM.v1, DMM.v2]
    not_parameters = ['', 'Parameter', 0, 1.1, Measurement]

    meas = Measurement()

    for not_a_parameter in not_parameters:
        with pytest.raises(ValueError):
            meas.register_parameter(not_a_parameter)

    my_param = DAC.ch1
    meas.register_parameter(my_param)
    assert len(meas.parameters) == 1
    paramspec = meas.parameters[my_param.full_name]
    assert paramspec.name == my_param.full_name
    assert paramspec.label == my_param.label
    assert paramspec.unit == my_param.unit
    assert paramspec.type == 'number'

    # registering the same parameter twice should lead
    # to a replacement/update

    my_param.unit = my_param.unit + '/s'
    meas.register_parameter(my_param)
    assert len(meas.parameters) == 1
    paramspec = meas.parameters[my_param.full_name]
    assert paramspec.name == my_param.full_name
    assert paramspec.label == my_param.label
    assert paramspec.unit == my_param.unit
    assert paramspec.type == 'number'

    for parameter in parameters:
        with pytest.raises(ValueError):
            meas.register_parameter(my_param, setpoints=(parameter,))
