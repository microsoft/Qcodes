import pytest
import numpy as np
from numpy.testing import assert_allclose

from qcodes.dataset.data_export import get_data_by_id, _get_data_from_ds
from qcodes.dataset.descriptions.param_spec import ParamSpecBase
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.measurements import Measurement


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

    for data in (get_data_by_id(dataset.run_id), _get_data_from_ds(dataset)):
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
