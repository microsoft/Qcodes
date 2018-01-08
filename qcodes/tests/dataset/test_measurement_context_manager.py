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


def test_register_parameter_numbers(DAC, DMM):
    """
    Test the registration of scalar QCoDeS parameters
    """

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
    paramspec = meas.parameters[str(my_param)]
    assert paramspec.name == str(my_param)
    assert paramspec.label == my_param.label
    assert paramspec.unit == my_param.unit
    assert paramspec.type == 'number'

    for parameter in parameters:
        with pytest.raises(ValueError):
            meas.register_parameter(my_param, setpoints=(parameter,))
        with pytest.raises(ValueError):
            meas.register_parameter(my_param, basis=(parameter,))

    meas.register_parameter(DAC.ch2)
    meas.register_parameter(DMM.v1)
    meas.register_parameter(DMM.v2)
    meas.register_parameter(my_param, basis=(DAC.ch2,),
                            setpoints=(DMM.v1, DMM.v2))

    assert list(meas.parameters.keys()) == [str(my_param), str(DAC.ch2),
                                            str(DMM.v1), str(DMM.v2)]
    paramspec = meas.parameters[str(my_param)]
    assert paramspec.name == str(my_param)
    assert paramspec.inferred_from == ', '.join([str(DAC.ch2)])
    assert paramspec.depends_on == ', '.join([str(DMM.v1), str(DMM.v2)])

    meas = Measurement()

    meas.register_parameter(DAC.ch1)
    meas.register_parameter(DAC.ch2, setpoints=(DAC.ch1,))
    with pytest.raises(ValueError):
        meas.register_parameter(DMM.v1, setpoints=(DAC.ch2,))


def test_unregister_parameter(DAC, DMM):
    """
    Test the unregistering of parameters.
    """

    DAC.add_parameter('impedance',
                      get_cmd=lambda: 5)

    meas = Measurement()

    meas.register_parameter(DAC.ch2)
    meas.register_parameter(DMM.v1)
    meas.register_parameter(DMM.v2)
    meas.register_parameter(DAC.ch1, basis=(DMM.v1, DMM.v2),
                            setpoints=(DAC.ch2,))

    with pytest.raises(ValueError):
        meas.unregister_parameter(DAC.ch2)
    with pytest.raises(ValueError):
        meas.unregister_parameter(str(DAC.ch2))
    with pytest.raises(ValueError):
        meas.unregister_parameter(DMM.v1)
    with pytest.raises(ValueError):
        meas.unregister_parameter(DMM.v2)

    meas.unregister_parameter(DAC.ch1)
    assert list(meas.parameters.keys()) == [str(DAC.ch2), str(DMM.v1),
                                            str(DMM.v2)]

    meas.unregister_parameter(DAC.ch2)
    assert list(meas.parameters.keys()) == [str(DMM.v1), str(DMM.v2)]
