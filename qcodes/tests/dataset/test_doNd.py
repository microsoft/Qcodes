"""
These are the basic black box tests for the doNd functions.
"""
import hypothesis.strategies as hst
import matplotlib.pyplot as plt
import numpy as np
import pytest
from hypothesis import given, settings

from qcodes import config
from qcodes.dataset.data_set import DataSet
from qcodes.instrument.parameter import Parameter
from qcodes.tests.dataset.conftest import empty_temp_db, experiment
from qcodes.tests.instrument_mocks import (ArraySetPointParam,
                                           Multi2DSetPointParam,
                                           Multi2DSetPointParam2Sizes,
                                           MultiSetPointParam)
from qcodes.utils import validators
from qcodes.utils.dataset.doNd import do0d, do1d, do2d
from qcodes.utils.validators import Arrays

from .conftest import ArrayshapedParam

temp_db = empty_temp_db
temp_exp = experiment


@pytest.fixture(autouse=True)
def set_tmp_output_dir(tmpdir):
    old_config = config.user.mainfolder
    try:
        config.user.mainfolder = str(tmpdir)
        yield
    finally:
        config.user.mainfolder = old_config


@pytest.fixture()
def plot_close():
    yield
    plt.close('all')


@pytest.fixture()
def _param():
    p = Parameter('simple_parameter',
                  set_cmd=None,
                  get_cmd=lambda: 1)
    return p


@pytest.fixture()
def _param_complex():
    p = Parameter('simple_complex_parameter',
                  set_cmd=None,
                  get_cmd=lambda: 1 + 1j,
                  vals=validators.ComplexNumbers())
    return p


@pytest.fixture()
def _param_set():
    p = Parameter('simple_setter_parameter',
                  set_cmd=None,
                  get_cmd=None)
    return p


@pytest.fixture()
def _param_set_2():
    p = Parameter('simple_setter_parameter_2',
                  set_cmd=None,
                  get_cmd=None)
    return p


def _param_func(_p):
    """
    A private utility function.
    """
    _new_param = Parameter('modified_parameter',
                           set_cmd=None,
                           get_cmd=lambda: _p.get()*2)
    return _new_param


@pytest.fixture()
def _param_callable(_param):
    return _param_func(_param)


def test_param_callable(_param_callable):
    _param_modified = _param_callable
    assert _param_modified.get() == 2


@pytest.mark.usefixtures("plot_close", "temp_exp", "temp_db")
@pytest.mark.parametrize('period, plot', [(None, True), (None, False),
                         (1, True), (1, False)])
def test_do0d_with_real_parameter(_param, period, plot):
    do0d(_param, write_period=period, do_plot=plot)


@pytest.mark.usefixtures("plot_close", "temp_exp", "temp_db")
@pytest.mark.parametrize('period, plot', [(None, True), (None, False),
                         (1, True), (1, False)])
def test_do0d_with_complex_parameter(_param_complex, period, plot):
    do0d(_param_complex, write_period=period, do_plot=plot)


@pytest.mark.usefixtures("plot_close", "temp_exp", "temp_db")
@pytest.mark.parametrize('period, plot', [(None, True), (None, False),
                         (1, True), (1, False)])
def test_do0d_with_a_callable(_param_callable, period, plot):
    do0d(_param_callable, write_period=period, do_plot=plot)


@pytest.mark.usefixtures("plot_close", "temp_exp", "temp_db")
@pytest.mark.parametrize('period, plot', [(None, True), (None, False),
                         (1, True), (1, False)])
def test_do0d_with_2_parameters(_param, _param_complex, period, plot):
    do0d(_param, _param_complex, write_period=period, do_plot=plot)


@pytest.mark.usefixtures("plot_close", "temp_exp", "temp_db")
@pytest.mark.parametrize('period, plot', [(None, True), (None, False),
                         (1, True), (1, False)])
def test_do0d_with_parameter_and_a_callable(_param_complex, _param_callable,
                                            period, plot):
    do0d(_param_callable, _param_complex, write_period=period, do_plot=plot)


@pytest.mark.usefixtures("plot_close", "temp_exp", "temp_db")
def test_do0d_output_type_real_parameter(_param):
    data = do0d(_param)
    assert isinstance(data[0], DataSet) is True


@pytest.mark.usefixtures("plot_close", "temp_exp", "temp_db")
def test_do0d_output_type_complex_parameter(_param_complex):
    data_complex = do0d(_param_complex)
    assert isinstance(data_complex[0], DataSet) is True


@pytest.mark.usefixtures("plot_close", "temp_exp", "temp_db")
def test_do0d_output_type_callable(_param_callable):
    data_func = do0d(_param_callable)
    assert isinstance(data_func[0], DataSet) is True


@pytest.mark.usefixtures("plot_close", "temp_exp", "temp_db")
def test_do0d_output_data(_param):
    exp = do0d(_param)
    data = exp[0]
    assert data.parameters == _param.name
    loaded_data = data.get_parameter_data()['simple_parameter']['simple_parameter']
    assert loaded_data == np.array([_param.get()])


@pytest.mark.usefixtures("temp_exp", "temp_db")
@pytest.mark.parametrize("multiparamtype", [MultiSetPointParam,
                                            Multi2DSetPointParam,
                                            Multi2DSetPointParam2Sizes])
@given(n_points_pws=hst.integers(min_value=1, max_value=1000))
@settings(deadline=None)
def test_do0d_verify_shape(_param, _param_complex, multiparamtype,
                           dummyinstrument, n_points_pws):
    arrayparam = ArraySetPointParam(name='arrayparam')
    multiparam = multiparamtype(name='multiparam')
    paramwsetpoints = dummyinstrument.A.dummy_parameter_with_setpoints
    dummyinstrument.A.dummy_start(0)
    dummyinstrument.A.dummy_stop(1)
    dummyinstrument.A.dummy_n_points(n_points_pws)

    results = do0d(arrayparam, multiparam, paramwsetpoints,
                   _param, _param_complex,
                   do_plot=False)
    expected_shapes = {}
    for i, name in enumerate(multiparam.full_names):
        expected_shapes[name] = tuple(multiparam.shapes[i])
    expected_shapes['arrayparam'] = tuple(arrayparam.shape)
    expected_shapes['simple_parameter'] = ()
    expected_shapes['simple_complex_parameter'] = ()
    expected_shapes[paramwsetpoints.full_name] = (n_points_pws, )
    assert results[0].description.shapes == expected_shapes

    ds = results[0]

    assert ds.description.shapes == expected_shapes

    data = ds.get_parameter_data()

    for name, data in data.items():
        for param_data in data.values():
            assert param_data.shape == expected_shapes[name]


@pytest.mark.usefixtures("temp_exp", "temp_db")
def test_do0d_parameter_with_array_vals():
    param = ArrayshapedParam(name='paramwitharrayval', vals=Arrays(shape=(10,)))
    results = do0d(param)
    expected_shapes = {'paramwitharrayval': (10,)}
    assert results[0].description.shapes == expected_shapes


@pytest.mark.usefixtures("plot_close", "temp_exp", "temp_db")
@pytest.mark.parametrize('delay', [0, 0.1, 1])
def test_do1d_with_real_parameter(_param_set, _param, delay):

    start = 0
    stop = 1
    num_points = 1

    do1d(_param_set, start, stop, num_points, delay, _param)


@pytest.mark.usefixtures("plot_close", "temp_exp", "temp_db")
@pytest.mark.parametrize('delay', [0, 0.1, 1])
def test_do1d_with_complex_parameter(_param_set, _param_complex, delay):

    start = 0
    stop = 1
    num_points = 1

    do1d(_param_set, start, stop, num_points, delay, _param_complex)


@pytest.mark.usefixtures("plot_close", "temp_exp", "temp_db")
@pytest.mark.parametrize('delay', [0, 0.1, 1])
def test_do1d_with_2_parameter(_param_set, _param, _param_complex, delay):

    start = 0
    stop = 1
    num_points = 1

    do1d(_param_set, start, stop, num_points, delay, _param, _param_complex)


@pytest.mark.usefixtures("plot_close", "temp_exp", "temp_db")
@pytest.mark.parametrize('delay', [0, 0.1, 1])
def test_do1d_output_type_real_parameter(_param_set, _param, delay):

    start = 0
    stop = 1
    num_points = 1

    data = do1d(_param_set, start, stop, num_points, delay, _param)
    assert isinstance(data[0], DataSet) is True


@pytest.mark.usefixtures("plot_close", "temp_exp", "temp_db")
def test_do1d_output_data(_param, _param_set):

    start = 0
    stop = 1
    num_points = 5
    delay = 0

    exp = do1d(_param_set, start, stop, num_points, delay, _param)
    data = exp[0]

    assert data.parameters == f'{_param_set.name},{_param.name}'
    loaded_data = data.get_parameter_data()['simple_parameter']

    np.testing.assert_array_equal(loaded_data[_param.name], np.ones(5))
    np.testing.assert_array_equal(loaded_data[_param_set.name], np.linspace(0, 1, 5))


@pytest.mark.usefixtures("temp_exp", "temp_db")
@pytest.mark.parametrize("multiparamtype", [MultiSetPointParam,
                                            Multi2DSetPointParam,
                                            Multi2DSetPointParam2Sizes])
@given(num_points=hst.integers(min_value=1, max_value=10),
       n_points_pws=hst.integers(min_value=1, max_value=1000))
@settings(deadline=None)
def test_do1d_verify_shape(_param, _param_complex, _param_set, multiparamtype,
                           dummyinstrument, num_points, n_points_pws):
    arrayparam = ArraySetPointParam(name='arrayparam')
    multiparam = multiparamtype(name='multiparam')
    paramwsetpoints = dummyinstrument.A.dummy_parameter_with_setpoints
    dummyinstrument.A.dummy_start(0)
    dummyinstrument.A.dummy_stop(1)
    dummyinstrument.A.dummy_n_points(n_points_pws)

    start = 0
    stop = 1
    delay = 0

    results = do1d(_param_set, start, stop, num_points, delay,
                   arrayparam, multiparam, paramwsetpoints,
                   _param, _param_complex,
                   do_plot=False)
    expected_shapes = {}
    for i, name in enumerate(multiparam.full_names):
        expected_shapes[name] = (num_points, ) + tuple(multiparam.shapes[i])
    expected_shapes['arrayparam'] = (num_points, ) + tuple(arrayparam.shape)
    expected_shapes['simple_parameter'] = (num_points, )
    expected_shapes['simple_complex_parameter'] = (num_points, )
    expected_shapes[paramwsetpoints.full_name] = (num_points, n_points_pws)
    assert results[0].description.shapes == expected_shapes


@pytest.mark.usefixtures("temp_exp", "temp_db")
def test_do1d_parameter_with_array_vals(_param_set):
    param = ArrayshapedParam(name='paramwitharrayval', vals=Arrays(shape=(10,)))
    start = 0
    stop = 1
    num_points = 15  #  make param
    delay = 0

    results = do1d(_param_set, start, stop, num_points, delay,
                   param, do_plot=False)
    expected_shapes = {'paramwitharrayval': (num_points, 10)}

    ds = results[0]

    assert ds.description.shapes == expected_shapes

    data = ds.get_parameter_data()

    for name, data in data.items():
        for param_data in data.values():
            assert param_data.shape == expected_shapes[name]


@pytest.mark.usefixtures("plot_close", "temp_exp", "temp_db")
@pytest.mark.parametrize('sweep, columns', [(False, False), (False, True),
                         (True, False), (True, True)])
def test_do2d(_param, _param_complex, _param_set, _param_set_2, sweep, columns):

    start_p1 = 0
    stop_p1 = 1
    num_points_p1 = 1
    delay_p1 = 0

    start_p2 = 0.1
    stop_p2 = 1.1
    num_points_p2 = 2
    delay_p2 = 0.01

    do2d(_param_set, start_p1, stop_p1, num_points_p1, delay_p1,
         _param_set_2, start_p2, stop_p2, num_points_p2, delay_p2,
         _param, _param_complex, set_before_sweep=sweep, flush_columns=columns)


@pytest.mark.usefixtures("plot_close", "temp_exp", "temp_db")
def test_do2d_output_type(_param, _param_complex, _param_set, _param_set_2):

    start_p1 = 0
    stop_p1 = 0.5
    num_points_p1 = 1
    delay_p1 = 0

    start_p2 = 0.1
    stop_p2 = 0.75
    num_points_p2 = 2
    delay_p2 = 0.025

    data = do2d(_param_set, start_p1, stop_p1, num_points_p1, delay_p1,
                _param_set_2, start_p2, stop_p2, num_points_p2, delay_p2,
                _param, _param_complex)
    assert isinstance(data[0], DataSet) is True


@pytest.mark.usefixtures("plot_close", "temp_exp", "temp_db")
def test_do2d_output_data(_param, _param_complex, _param_set, _param_set_2):

    start_p1 = 0
    stop_p1 = 0.5
    num_points_p1 = 5
    delay_p1 = 0

    start_p2 = 0.5
    stop_p2 = 1
    num_points_p2 = 5
    delay_p2 = 0.0

    exp = do2d(_param_set, start_p1, stop_p1, num_points_p1, delay_p1,
               _param_set_2, start_p2, stop_p2, num_points_p2, delay_p2,
               _param, _param_complex)
    data = exp[0]

    assert data.parameters == (f'{_param_set.name},{_param_set_2.name},'
                               f'{_param.name},{_param_complex.name}')
    loaded_data = data.get_parameter_data()
    expected_data_1 = np.ones(25).reshape(num_points_p1, num_points_p2)

    np.testing.assert_array_equal(loaded_data[_param.name][_param.name],
                                  expected_data_1)
    expected_data_2 = (1+1j)*np.ones(25).reshape(num_points_p1, num_points_p2)
    np.testing.assert_array_equal(
        loaded_data[_param_complex.name][_param_complex.name],
        expected_data_2
    )

    expected_setpoints_1 = np.repeat(
        np.linspace(start_p1, stop_p1, num_points_p1),
        num_points_p2).reshape(num_points_p1, num_points_p2)
    np.testing.assert_array_equal(
        loaded_data[_param_complex.name][_param_set.name],
        expected_setpoints_1
    )

    expected_setpoints_2 = np.tile(
        np.linspace(start_p2, stop_p2, num_points_p2),
                    num_points_p1).reshape(num_points_p1, num_points_p2)
    np.testing.assert_array_equal(
        loaded_data[_param_complex.name][_param_set_2.name],
        expected_setpoints_2
    )


@pytest.mark.usefixtures("temp_exp", "temp_db")
@pytest.mark.parametrize('sweep, columns', [(False, False), (False, True),
                         (True, False), (True, True)])
@pytest.mark.parametrize("multiparamtype", [MultiSetPointParam,
                                            Multi2DSetPointParam,
                                            Multi2DSetPointParam2Sizes])
@given(num_points_p1=hst.integers(min_value=1, max_value=10),
       num_points_p2=hst.integers(min_value=1, max_value=10),
       n_points_pws=hst.integers(min_value=1, max_value=1000))
@settings(deadline=None)
def test_do2d_verify_shape(_param, _param_complex, _param_set, _param_set_2,
                           multiparamtype,
                           dummyinstrument,
                           sweep, columns,
                           num_points_p1, num_points_p2, n_points_pws):
    arrayparam = ArraySetPointParam(name='arrayparam')
    multiparam = multiparamtype(name='multiparam')
    paramwsetpoints = dummyinstrument.A.dummy_parameter_with_setpoints
    dummyinstrument.A.dummy_start(0)
    dummyinstrument.A.dummy_stop(1)
    dummyinstrument.A.dummy_n_points(n_points_pws)

    start_p1 = 0
    stop_p1 = 1
    delay_p1 = 0

    start_p2 = 0.1
    stop_p2 = 1.1
    delay_p2 = 0

    results = do2d(_param_set, start_p1, stop_p1, num_points_p1, delay_p1,
                   _param_set_2, start_p2, stop_p2, num_points_p2, delay_p2,
                   arrayparam, multiparam, paramwsetpoints,
                   _param, _param_complex,
                   set_before_sweep=sweep,
                   flush_columns=columns, do_plot=False)
    expected_shapes = {}
    for i, name in enumerate(multiparam.full_names):
        expected_shapes[name] = ((num_points_p1, num_points_p2) +
                                 tuple(multiparam.shapes[i]))
    expected_shapes['arrayparam'] = ((num_points_p1, num_points_p2) +
                                     tuple(arrayparam.shape))
    expected_shapes['simple_parameter'] = (num_points_p1, num_points_p2)
    expected_shapes['simple_complex_parameter'] = (num_points_p1, num_points_p2)
    expected_shapes[paramwsetpoints.full_name] = (num_points_p1,
                                                  num_points_p2,
                                                  n_points_pws)

    assert results[0].description.shapes == expected_shapes
    ds = results[0]

    data = ds.get_parameter_data()

    for name, data in data.items():
        for param_data in data.values():
            assert param_data.shape == expected_shapes[name]


@pytest.mark.usefixtures("plot_close", "temp_exp", "temp_db")
def test_do1d_additional_setpoints(_param, _param_complex, _param_set):
    additional_setpoints = [Parameter(
        f'additional_setter_parameter_{i}',
        set_cmd=None,
        get_cmd=None) for i in range(2)]
    start_p1 = 0
    stop_p1 = 0.5
    num_points_p1 = 5
    delay_p1 = 0

    for x in range(3):
        for y in range(4):
            additional_setpoints[0](x)
            additional_setpoints[1](y)
            results = do1d(
                _param_set, start_p1, stop_p1, num_points_p1, delay_p1,
                _param, _param_complex,
                additional_setpoints=additional_setpoints)
            for deps in results[0].description.interdeps.dependencies.values():
                assert len(deps) == 1 + len(additional_setpoints)
            # Calling the fixture won't work here due to loop-scope.
            # Thus, we make an explicit call to close plots. This will be
            # repeated in similarly design tests.
            plt.close('all')


@given(num_points_p1=hst.integers(min_value=1, max_value=10))
@pytest.mark.usefixtures("temp_exp", "temp_db")
def test_do1d_additional_setpoints_shape(_param, _param_complex, _param_set,
                                         num_points_p1):
    arrayparam = ArraySetPointParam(name='arrayparam')
    array_shape = arrayparam.shape
    additional_setpoints = [Parameter(
        f'additional_setter_parameter_{i}',
        set_cmd=None,
        get_cmd=None) for i in range(2)]
    start_p1 = 0
    stop_p1 = 0.5
    delay_p1 = 0

    x = 1
    y = 2

    additional_setpoints[0](x)
    additional_setpoints[1](y)
    results = do1d(
        _param_set, start_p1, stop_p1, num_points_p1, delay_p1,
        _param, arrayparam,
        additional_setpoints=additional_setpoints,
        do_plot=False)
    expected_shapes = {
        'arrayparam': (1, 1, num_points_p1, array_shape[0]),
        'simple_parameter': (1, 1, num_points_p1)
    }
    assert results[0].description.shapes == expected_shapes


@pytest.mark.usefixtures("plot_close", "temp_exp", "temp_db")
def test_do2d_additional_setpoints(_param, _param_complex,
                                   _param_set, _param_set_2):
    additional_setpoints = [Parameter(
        f'additional_setter_parameter_{i}',
        set_cmd=None,
        get_cmd=None) for i in range(2)]
    start_p1 = 0
    stop_p1 = 0.5
    num_points_p1 = 5
    delay_p1 = 0

    start_p2 = 0.5
    stop_p2 = 1
    num_points_p2 = 5
    delay_p2 = 0.0
    for x in range(3):
        for y in range(4):
            additional_setpoints[0](x)
            additional_setpoints[1](y)
            results = do2d(
                _param_set, start_p1, stop_p1, num_points_p1, delay_p1,
                _param_set_2, start_p2, stop_p2, num_points_p2, delay_p2,
                _param, _param_complex,
                additional_setpoints=additional_setpoints)
            for deps in results[0].description.interdeps.dependencies.values():
                assert len(deps) == 2 + len(additional_setpoints)
            plt.close('all')


@given(num_points_p1=hst.integers(min_value=1, max_value=10),
       num_points_p2=hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
@pytest.mark.usefixtures("temp_exp", "temp_db")
def test_do2d_additional_setpoints_shape(
        _param, _param_complex,
        _param_set, _param_set_2,
        num_points_p1, num_points_p2):
    arrayparam = ArraySetPointParam(name='arrayparam')
    array_shape = arrayparam.shape
    additional_setpoints = [Parameter(
        f'additional_setter_parameter_{i}',
        set_cmd=None,
        get_cmd=None) for i in range(2)]
    start_p1 = 0
    stop_p1 = 0.5
    delay_p1 = 0

    start_p2 = 0.5
    stop_p2 = 1
    delay_p2 = 0.0

    x = 1
    y = 2

    additional_setpoints[0](x)
    additional_setpoints[1](y)
    results = do2d(
        _param_set, start_p1, stop_p1, num_points_p1, delay_p1,
        _param_set_2, start_p2, stop_p2, num_points_p2, delay_p2,
        _param, _param_complex, arrayparam,
        additional_setpoints=additional_setpoints,
        do_plot=False)
    expected_shapes = {
        'arrayparam': (1, 1, num_points_p1, num_points_p2, array_shape[0]),
        'simple_complex_parameter': (1, 1, num_points_p1, num_points_p2),
        'simple_parameter': (1, 1, num_points_p1, num_points_p2)
    }
    assert results[0].description.shapes == expected_shapes
