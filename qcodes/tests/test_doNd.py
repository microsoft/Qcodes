"""
These are the basic black box tests for the doNd functions.
"""
from qcodes.dataset.data_set import DataSet
from qcodes.utils.dataset.doNd import do0d, do1d, do2d
from qcodes.instrument.parameter import Parameter
from qcodes import config, new_experiment
from qcodes.utils import validators

import pytest
import matplotlib.pyplot as plt

config.user.mainfolder = "output"  # set output folder for doNd's
new_experiment("doNd-tests", sample_name="no sample")


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


@pytest.mark.usefixtures("plot_close")
@pytest.mark.parametrize('period, plot', [(None, True), (None, False),
                         (1, True), (1, False)])
def test_do0d_with_real_parameter(_param, period, plot):
    do0d(_param, write_period=period, do_plot=plot)


@pytest.mark.usefixtures("plot_close")
@pytest.mark.parametrize('period, plot', [(None, True), (None, False),
                         (1, True), (1, False)])
def test_do0d_with_complex_parameter(_param_complex, period, plot):
    do0d(_param_complex, write_period=period, do_plot=plot)


@pytest.mark.usefixtures("plot_close")
@pytest.mark.parametrize('period, plot', [(None, True), (None, False),
                         (1, True), (1, False)])
def test_do0d_with_a_callable(_param_callable, period, plot):
    do0d(_param_callable, write_period=period, do_plot=plot)


@pytest.mark.usefixtures("plot_close")
@pytest.mark.parametrize('period, plot', [(None, True), (None, False),
                         (1, True), (1, False)])
def test_do0d_with_multiparameters(_param, _param_complex, period, plot):
    do0d(_param, _param_complex, write_period=period, do_plot=plot)


@pytest.mark.usefixtures("plot_close")
@pytest.mark.parametrize('period, plot', [(None, True), (None, False),
                         (1, True), (1, False)])
def test_do0d_with_parameter_and_a_callable(_param_complex, _param_callable,
                                            period, plot):
    do0d(_param_callable, _param_complex, write_period=period, do_plot=plot)


@pytest.mark.usefixtures("plot_close")
def test_do0d_output_type_real_parameter(_param):
    data = do0d(_param)
    assert isinstance(data[0], DataSet) is True


@pytest.mark.usefixtures("plot_close")
def test_do0d_output_type_complex_parameter(_param_complex):
    data_complex = do0d(_param_complex)
    assert isinstance(data_complex[0], DataSet) is True


@pytest.mark.usefixtures("plot_close")
def test_do0d_output_type_callable(_param_callable):
    data_func = do0d(_param_callable)
    assert isinstance(data_func[0], DataSet) is True


@pytest.mark.usefixtures("plot_close")
def test_do0d_output_data(_param):
    exp = do0d(_param)
    data = exp[0]
    assert data.parameters == _param.name
    assert data.get_values(_param.name)[0][0] == _param.get()


@pytest.mark.usefixtures("plot_close")
@pytest.mark.parametrize('delay', [0, 0.1, 1])
def test_do1d_with_real_parameter(_param_set, _param, delay):

    start = 0
    stop = 1
    num_points = 1

    do1d(_param_set, start, stop, num_points, delay, _param)


@pytest.mark.usefixtures("plot_close")
@pytest.mark.parametrize('delay', [0, 0.1, 1])
def test_do1d_with_complex_parameter(_param_set, _param_complex, delay):

    start = 0
    stop = 1
    num_points = 1

    do1d(_param_set, start, stop, num_points, delay, _param_complex)


@pytest.mark.usefixtures("plot_close")
@pytest.mark.parametrize('delay', [0, 0.1, 1])
def test_do1d_with_multiparameter(_param_set, _param, _param_complex, delay):

    start = 0
    stop = 1
    num_points = 1

    do1d(_param_set, start, stop, num_points, delay, _param, _param_complex)


@pytest.mark.usefixtures("plot_close")
@pytest.mark.parametrize('delay', [0, 0.1, 1])
def test_do1d_output_type_real_parameter(_param_set, _param, delay):

    start = 0
    stop = 1
    num_points = 1

    data = do1d(_param_set, start, stop, num_points, delay, _param)
    assert isinstance(data[0], DataSet) is True


@pytest.mark.usefixtures("plot_close")
def test_do1d_output_data(_param, _param_set):

    start = 0
    stop = 1
    num_points = 5
    delay = 0

    exp = do1d(_param_set, start, stop, num_points, delay, _param)
    data = exp[0]

    assert data.parameters == f'{_param_set.name},{_param.name}'
    assert data.get_values(_param.name) == [[1]] * 5
    assert data.get_values(_param_set.name) == [[0], [0.25], [0.5], [0.75], [1]]


@pytest.mark.usefixtures("plot_close")
@pytest.mark.parametrize('sweep, columns', [(False, False), (False, True),
                         (True, False), (True, True)])
def test_do2d(_param, _param_complex, _param_set, sweep, columns):

    start_p1 = 0
    stop_p1 = 1
    num_points_p1 = 1
    delay_p1 = 0

    start_p2 = 0.1
    stop_p2 = 1.1
    num_points_p2 = 2
    delay_p2 = 0.01

    do2d(_param_set, start_p1, stop_p1, num_points_p1, delay_p1,
         _param_set, start_p2, stop_p2, num_points_p2, delay_p2,
         _param, _param_complex, set_before_sweep=sweep, flush_columns=columns)


@pytest.mark.usefixtures("plot_close")
def test_do2d_output_type(_param, _param_complex, _param_set):

    start_p1 = 0
    stop_p1 = 0.5
    num_points_p1 = 1
    delay_p1 = 0

    start_p2 = 0.1
    stop_p2 = 0.75
    num_points_p2 = 2
    delay_p2 = 0.025

    data = do2d(_param_set, start_p1, stop_p1, num_points_p1, delay_p1,
                _param_set, start_p2, stop_p2, num_points_p2, delay_p2,
                _param, _param_complex)
    assert isinstance(data[0], DataSet) is True


@pytest.mark.usefixtures("plot_close")
def test_do2d_output_data(_param, _param_complex, _param_set):

    start_p1 = 0
    stop_p1 = 0.5
    num_points_p1 = 5
    delay_p1 = 0

    start_p2 = 0.5
    stop_p2 = 1
    num_points_p2 = 5
    delay_p2 = 0.0

    exp = do2d(_param_set, start_p1, stop_p1, num_points_p1, delay_p1,
               _param_set, start_p2, stop_p2, num_points_p2, delay_p2,
               _param, _param_complex)
    data = exp[0]

    assert data.parameters == f'{_param_set.name},{_param.name},' \
                              f'{_param_complex.name}'
    assert data.get_values(_param.name) == [[1]] * 25
    assert data.get_values(_param_complex.name) == [[(1+1j)]] * 25
    assert data.get_values(_param_set.name) == [[0.5], [0.5], [0.625], [0.625],
                                                [0.75], [0.75], [0.875],
                                                [0.875], [1], [1]] * 5


def test_do1d_additional_setpoints(_param, _param_complex, _param_set):
    additional_setpoints = [Parameter(
        'simple_setter_parameter',
        set_cmd=None,
        get_cmd=None) for _ in range(2)]
    start_p1 = 0
    stop_p1 = 0.5
    num_points_p1 = 5
    delay_p1 = 0

    for x in range(3):
        for y in range(4):
            additional_setpoints[0](x)
            additional_setpoints[1](y)
            do1d(_param_set, start_p1, stop_p1, num_points_p1, delay_p1,
                 _param, _param_complex,
                 additional_setpoints=additional_setpoints)
            # Calling the fixture won't work here due to loop-scope.
            # Thus, we make an explicit call to close plots. This will be
            # repeated in similarly design tests.
            plt.close('all')


def test_do2d_additional_setpoints(_param, _param_complex, _param_set):
    additional_setpoints = [Parameter(
        'simple_setter_parameter',
        set_cmd=None,
        get_cmd=None) for _ in range(2)]
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
            do2d(_param_set, start_p1, stop_p1, num_points_p1, delay_p1,
                 _param_set, start_p2, stop_p2, num_points_p2, delay_p2,
                 _param, _param_complex,
                 additional_setpoints=additional_setpoints)
            plt.close('all')
