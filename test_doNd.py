"""
These are the basic black box tests for the doNd functions.
"""

from qdev_wrappers.dataset.doNd import do0d, do1d, do2d
from typing import Tuple, List, Optional
from qcodes.instrument.parameter import Parameter
from qcodes import config, new_experiment, load_by_id
from qcodes.utils import validators

import pytest

config.user.mainfolder = "output"  # set ouput folder for doNd's
new_experiment("doNd-tests", sample_name="no sample")


@pytest.fixture()
def _parameters():

    _param = Parameter('simple_parameter',
                   set_cmd=None,
                   get_cmd=lambda: 1)

    _paramComplex = Parameter('simple_complex_parameter',
                   set_cmd=None,
                   get_cmd=lambda: 1 + 1j,
                   vals=validators.ComplexNumbers())

    _param_set = Parameter('simple_setter_parameter',
                       set_cmd=None,
                       get_cmd=None)

    return _param, _paramComplex, _param_set


def _param_func(_p):
    """
    A private utility function.
    """
    _new_param = Parameter('modified_parameter',
                           set_cmd= None,
                           get_cmd= lambda: _p.get()*2)
    return _new_param


@pytest.fixture()
def _param_callable(_parameters):
    _param, _, _ = _parameters
    return _param_func(_param)


def test_param_callable(_param_callable):
    _param_modified = _param_callable
    assert _param_modified.get() == 2


@pytest.mark.parametrize('period, plot', [(None, True), (None, False),
                         (1, True), (1, False)])
def test_do0d(_parameters, _param_callable, period, plot):

    _param, _paramComplex, _ = _parameters

    # Note that the following tests can be refactored as seperate tests.
    # In that case, with the parametrization, one would have 20 test cases
    # instead of 4. The followings represent the minimum set of cases to be
    # satisfied.

    do0d(_param, write_period=period, do_plot=plot)
    do0d(_paramComplex, write_period=period, do_plot=plot)
    do0d(_param_callable, write_period=period, do_plot=plot)
    do0d(_param, _paramComplex, write_period=period, do_plot=plot)
    do0d(_param_callable, _paramComplex, write_period=period, do_plot=plot)


def test_do0d_out_type_1(_parameters):

    _param, _, _ = _parameters
    data = do0d(_param)
    assert type(data[0]) == int


def test_do0d_out_type_2(_parameters):

    _, _paramComplex, _ = _parameters
    dataComplex = do0d(_paramComplex)
    assert type(dataComplex[0]) == int


def test_do0d_out_type_3(_parameters, _param_callable):

    _param, _, _ = _parameters
    dataFunc = do0d(_param_callable)
    assert type(dataFunc[0]) == int


def test_do0d_data(_parameters):

    _param, _, _ = _parameters
    exp = do0d(_param)
    data = load_by_id(exp[0])
    assert data.parameters == _param.name
    assert data.get_values(_param.name)[0][0] == _param.get()


@pytest.mark.parametrize('delay', [0, 0.1, 1])
def test_do1d(_parameters, delay):

    start = 0
    stop = 1
    num_points = 1

    _param, _paramComplex, _param_set = _parameters

    #Following tests represents the minimum set of cases and can be refactored.

    do1d(_param_set, start, stop, num_points, delay, _param)
    do1d(_param_set, start, stop, num_points, delay, _paramComplex)
    do1d(_param_set, start, stop, num_points, delay, _param, _paramComplex)


@pytest.mark.parametrize('delay', [0, 0.1, 1])
def test_do1d_out_type(_parameters, delay):

    start = 0
    stop = 1
    num_points = 1

    _param, _, _param_set = _parameters

    data = do1d(_param_set, start, stop, num_points, delay, _param)
    assert type(data[0]) == int


def test_do1d_data(_parameters):

    start = 0
    stop = 1
    num_points = 5
    delay = 0

    _param, _, _param_set = _parameters
    exp = do1d(_param_set, start, stop, num_points, delay, _param)
    data = load_by_id(exp[0])

    assert data.parameters == f'{_param_set.name},{_param.name}'
    assert data.get_values(_param.name) == [[1]] * 5
    assert data.get_values(_param_set.name) == [[0], [0.25], [0.5], [0.75], [1]]


@pytest.mark.parametrize('sweep, columns', [(False, False), (False, True),
                         (True, False), (True, True)])
def test_do2d(_parameters, sweep, columns):

    start_p1 = 0
    stop_p1 = 1
    num_points_p1 = 1
    delay_p1 = 0

    start_p2 = 0.1
    stop_p2 = 1.1
    num_points_p2 = 2
    delay_p2 = 0.01

    _param, _paramComplex, _param_set = _parameters

    do2d(_param_set, start_p1, stop_p1, num_points_p1, delay_p1,
         _param_set, start_p2, stop_p2, num_points_p2, delay_p2,
         _param, _paramComplex, set_before_sweep=sweep, flush_columns=columns)


def test_do2d_out_type(_parameters):

    start_p1 = 0
    stop_p1 = 0.5
    num_points_p1 = 1
    delay_p1 = 0

    start_p2 = 0.1
    stop_p2 = 0.75
    num_points_p2 = 2
    delay_p2 = 0.025

    _param, _paramComplex, _param_set = _parameters

    data = do2d(_param_set, start_p1, stop_p1, num_points_p1, delay_p1,
                 _param_set, start_p2, stop_p2, num_points_p2, delay_p2,
                 _param, _paramComplex)

    assert type(data[0]) == int


def test_do2d_data(_parameters):

    start_p1 = 0
    stop_p1 = 0.5
    num_points_p1 = 5
    delay_p1 = 0

    start_p2 = 0.5
    stop_p2 = 1
    num_points_p2 = 5
    delay_p2 = 0.0

    _param, _paramComplex, _param_set = _parameters

    exp = do2d(_param_set, start_p1, stop_p1, num_points_p1, delay_p1,
                 _param_set, start_p2, stop_p2, num_points_p2, delay_p2,
                 _param, _paramComplex)

    data = load_by_id(exp[0])

    assert data.parameters == f'{_param_set.name},{_param.name},{_paramComplex.name}'
    assert data.get_values(_param.name) == [[1]] * 25
    assert data.get_values(_paramComplex.name) == [[(1+1j)]] * 25
    assert data.get_values(_param_set.name) == [[0.5], [0.5], [0.625], [0.625],
                                                [0.75], [0.75], [0.875], [0.875],
                                                [1], [1]] * 5
