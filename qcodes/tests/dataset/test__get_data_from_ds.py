import hypothesis.strategies as hst
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from numpy.testing import assert_allclose

import qcodes as qc
from qcodes.dataset.data_export import _get_data_from_ds, get_data_by_id
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.param_spec import ParamSpecBase
from qcodes.dataset.measurements import Measurement
from qcodes.utils.deprecate import QCoDeSDeprecationWarning


def test_get_data_by_id_order(dataset):
    """
    Test that the added values of setpoints end up associated with the correct
    setpoint parameter, irrespective of the ordering of those setpoint
    parameters
    """
    indepA = ParamSpecBase('indep1', "numeric")
    indepB = ParamSpecBase('indep2', "numeric")
    depAB = ParamSpecBase('depAB', "numeric")
    depBA = ParamSpecBase('depBA', "numeric")

    idps = InterDependencies_(
        dependencies={depAB: (indepA, indepB), depBA: (indepB, indepA)})

    dataset.set_interdependencies(idps)

    dataset.mark_started()

    dataset.add_results([{'depAB': 12,
                          'indep2': 2,
                          'indep1': 1}])

    dataset.add_results([{'depBA': 21,
                          'indep2': 2,
                          'indep1': 1}])
    dataset.mark_completed()

    with pytest.warns(QCoDeSDeprecationWarning):
        data1 = get_data_by_id(dataset.run_id)

    data2 = _get_data_from_ds(dataset)

    for data in (data1, data2):
        data_dict = {el['name']: el['data'] for el in data[0]}
        assert data_dict['indep1'] == 1
        assert data_dict['indep2'] == 2

        data_dict = {el['name']: el['data'] for el in data[1]}
        assert data_dict['indep1'] == 1
        assert data_dict['indep2'] == 2


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.usefixtures("experiment")
def test_datasaver_multidimarrayparameter_as_array(
        SpectrumAnalyzer,
        bg_writing
):
    array_param = SpectrumAnalyzer.multidimspectrum
    meas = Measurement()
    meas.register_parameter(array_param, paramtype='array')
    assert len(meas.parameters) == 4
    inserted_data = array_param.get()
    with meas.run(write_in_background=bg_writing) as datasaver:
        datasaver.add_result((array_param, inserted_data))

    expected_shape = (1, 100, 50, 20)

    datadicts = _get_data_from_ds(datasaver.dataset)
    assert len(datadicts) == 1
    for datadict_list in datadicts:
        assert len(datadict_list) == 4
        for datadict in datadict_list:

            datadict['data'].shape = (np.prod(expected_shape),)
            if datadict['name'] == "dummy_SA_Frequency0":
                temp_data = np.linspace(array_param.start,
                                        array_param.stop,
                                        array_param.npts[0])
                expected_data = np.repeat(temp_data,
                                          expected_shape[2] * expected_shape[3])
            if datadict['name'] == "dummy_SA_Frequency1":
                temp_data = np.linspace(array_param.start,
                                        array_param.stop,
                                        array_param.npts[1])
                expected_data = np.tile(np.repeat(temp_data, expected_shape[3]),
                                        expected_shape[1])
            if datadict['name'] == "dummy_SA_Frequency2":
                temp_data = np.linspace(array_param.start,
                                        array_param.stop,
                                        array_param.npts[2])
                expected_data = np.tile(temp_data,
                                        expected_shape[1] * expected_shape[2])
            if datadict['name'] == "dummy_SA_multidimspectrum":
                expected_data = inserted_data.ravel()
            assert_allclose(datadict['data'], expected_data)


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.usefixtures("experiment")
def test_datasaver_multidimarrayparameter_as_numeric(SpectrumAnalyzer,
                                                     bg_writing):
    """
    Test that storing a multidim Array parameter as numeric unravels the
    parameter as expected.
    """

    array_param = SpectrumAnalyzer.multidimspectrum
    meas = Measurement()
    meas.register_parameter(array_param, paramtype='numeric')
    expected_shape = array_param.shape
    dims = len(array_param.shape)
    assert len(meas.parameters) == dims + 1

    points_expected = np.prod(array_param.npts)
    inserted_data = array_param.get()
    with meas.run(write_in_background=bg_writing) as datasaver:
        datasaver.add_result((array_param, inserted_data))

    assert datasaver.points_written == points_expected

    datadicts = _get_data_from_ds(datasaver.dataset)
    assert len(datadicts) == 1
    for datadict_list in datadicts:
        assert len(datadict_list) == 4
        for datadict in datadict_list:

            datadict['data'].shape = (np.prod(expected_shape),)
            if datadict['name'] == "dummy_SA_Frequency0":
                temp_data = np.linspace(array_param.start,
                                        array_param.stop,
                                        array_param.npts[0])
                expected_data = np.repeat(temp_data,
                                          expected_shape[1] * expected_shape[2])
            if datadict['name'] == "dummy_SA_Frequency1":
                temp_data = np.linspace(array_param.start,
                                        array_param.stop,
                                        array_param.npts[1])
                expected_data = np.tile(np.repeat(temp_data, expected_shape[2]),
                                        expected_shape[0])
            if datadict['name'] == "dummy_SA_Frequency2":
                temp_data = np.linspace(array_param.start,
                                        array_param.stop,
                                        array_param.npts[2])
                expected_data = np.tile(temp_data,
                                        expected_shape[0] * expected_shape[1])
            if datadict['name'] == "dummy_SA_multidimspectrum":
                expected_data = inserted_data.ravel()
            assert_allclose(datadict['data'], expected_data)


@settings(max_examples=5, deadline=None,
          suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(N=hst.integers(min_value=5, max_value=500))
@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.parametrize("storage_type", ['numeric', 'array'])
@pytest.mark.usefixtures("experiment")
def test_datasaver_array_parameters_channel(channel_array_instrument,
                                            DAC, N, storage_type,
                                            bg_writing):
    array_param = channel_array_instrument.A.dummy_array_parameter
    meas = Measurement()
    meas.register_parameter(DAC.ch1)
    meas.register_parameter(array_param, setpoints=[DAC.ch1], paramtype=storage_type)

    M = array_param.shape[0]
    with meas.run(write_in_background=bg_writing) as datasaver:
        for set_v in np.linspace(0, 0.01, N):
            datasaver.add_result((DAC.ch1, set_v),
                                 (array_param, array_param.get()))

    datadicts = _get_data_from_ds(datasaver.dataset)
    # one dependent parameter
    assert len(datadicts) == 1
    datadicts = datadicts[0]
    assert len(datadicts) == len(meas.parameters)
    for datadict in datadicts:
        if storage_type == "array":
            assert datadict["data"].shape == (N, M)
        else:
            assert datadict["data"].shape == (N * M,)


@settings(max_examples=5, deadline=None,
          suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(N=hst.integers(min_value=5, max_value=500))
@pytest.mark.usefixtures("experiment")
@pytest.mark.parametrize("bg_writing", [True, False])
def test_datasaver_array_parameters_array(channel_array_instrument, DAC, N,
                                          bg_writing):
    """
    Test that storing array parameters inside a loop works as expected
    """
    storage_type = "array"
    array_param = channel_array_instrument.A.dummy_array_parameter
    dependency_name = 'dummy_channel_inst_ChanA_array_setpoint_param_this_setpoint'

    # Now for a real measurement

    meas = Measurement()

    meas.register_parameter(DAC.ch1, paramtype='numeric')
    meas.register_parameter(array_param, setpoints=[DAC.ch1], paramtype=storage_type)

    assert len(meas.parameters) == 3

    M = array_param.shape[0]
    dac_datapoints = np.linspace(0, 0.01, N)
    with meas.run(write_in_background=bg_writing) as datasaver:
        for set_v in dac_datapoints:
            datasaver.add_result((DAC.ch1, set_v),
                                 (array_param, array_param.get()))

    datadicts = _get_data_from_ds(datasaver.dataset)
    # one dependent parameter
    assert len(datadicts) == 1
    datadicts = datadicts[0]
    assert len(datadicts) == len(meas.parameters)
    for datadict in datadicts:
        if datadict['name'] == 'dummy_dac_ch1':
            expected_data = np.repeat(dac_datapoints, M).reshape(N, M)
        if datadict['name'] == dependency_name:
            expected_data = np.tile(np.linspace(5, 9, 5), (N, 1))
        if datadict['name'] == 'dummy_channel_inst_ChanA_dummy_array_parameter':
            expected_data = np.empty((N, M))
            expected_data[:] = 2.
        assert_allclose(datadict['data'], expected_data)

        assert datadict["data"].shape == (N, M)


@pytest.mark.parametrize("bg_writing", [True, False])
def test_datasaver_multidim_array(experiment, bg_writing):
    """
    Test that inserting multidim parameters as arrays works as expected
    """
    meas = Measurement(experiment)
    size1 = 10
    size2 = 15

    data_mapping = {name: i for i, name in
                    zip(range(4), ['x1', 'x2', 'y1', 'y2'])}

    x1 = qc.ManualParameter('x1')
    x2 = qc.ManualParameter('x2')
    y1 = qc.ManualParameter('y1')
    y2 = qc.ManualParameter('y2')

    meas.register_parameter(x1, paramtype='array')
    meas.register_parameter(x2, paramtype='array')
    meas.register_parameter(y1, setpoints=[x1, x2], paramtype='array')
    meas.register_parameter(y2, setpoints=[x1, x2], paramtype='array')
    data = np.random.rand(4, size1, size2)
    expected = {'x1': data[0, :, :],
                'x2': data[1, :, :],
                'y1': data[2, :, :],
                'y2': data[3, :, :]}
    with meas.run(write_in_background=bg_writing) as datasaver:
        datasaver.add_result((str(x1), expected['x1']),
                             (str(x2), expected['x2']),
                             (str(y1), expected['y1']),
                             (str(y2), expected['y2']))

    datadicts = _get_data_from_ds(datasaver.dataset)
    assert len(datadicts) == 2
    for datadict_list in datadicts:
        assert len(datadict_list) == 3
        for datadict in datadict_list:
            dataindex = data_mapping[datadict["name"]]
            expected_data = data[dataindex : dataindex + 1, :, :]
            assert_allclose(datadict["data"], expected_data)

            assert datadict["data"].shape == (1, size1, size2)


@pytest.mark.parametrize("bg_writing", [True, False])
def test_datasaver_multidim_numeric(experiment, bg_writing):
    """
    Test that inserting multidim parameters as numeric works as expected
    """
    meas = Measurement(experiment)
    size1 = 10
    size2 = 15
    x1 = qc.ManualParameter('x1')
    x2 = qc.ManualParameter('x2')
    y1 = qc.ManualParameter('y1')
    y2 = qc.ManualParameter('y2')

    data_mapping = {name: i for i, name in
                    zip(range(4), ['x1', 'x2', 'y1', 'y2'])}

    meas.register_parameter(x1, paramtype='numeric')
    meas.register_parameter(x2, paramtype='numeric')
    meas.register_parameter(y1, setpoints=[x1, x2], paramtype='numeric')
    meas.register_parameter(y2, setpoints=[x1, x2], paramtype='numeric')
    data = np.random.rand(4, size1, size2)
    with meas.run(write_in_background=bg_writing) as datasaver:
        datasaver.add_result((str(x1), data[0, :, :]),
                             (str(x2), data[1, :, :]),
                             (str(y1), data[2, :, :]),
                             (str(y2), data[3, :, :]))

    datadicts = _get_data_from_ds(datasaver.dataset)
    assert len(datadicts) == 2
    for datadict_list in datadicts:
        assert len(datadict_list) == 3
        for datadict in datadict_list:
            dataindex = data_mapping[datadict['name']]
            expected_data = data[dataindex, :, :].ravel()
            assert_allclose(datadict['data'], expected_data)

            assert datadict['data'].shape == (size1 * size2,)
