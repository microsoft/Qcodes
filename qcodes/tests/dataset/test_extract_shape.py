from hypothesis import given
import hypothesis.strategies as hst
import numpy as np
import pytest

from qcodes.dataset.descriptions.detect_shapes import get_shape_of_measurement
from qcodes.instrument.parameter import Parameter
from qcodes.tests.instrument_mocks import ArraySetPointParam, MultiSetPointParam, Multi2DSetPointParam, Multi2DSetPointParam2Sizes, DummyChannelInstrument


@given(loop_shape=hst.lists(hst.integers(min_value=1), min_size=1, max_size=10))
def test_get_shape_for_parameter_from_len(loop_shape) -> None:
    a = Parameter(name='a', initial_cache_value=10)
    shape = get_shape_of_measurement(a, *loop_shape)
    assert shape == {'a': tuple(loop_shape)}


@given(loop_shape=hst.lists(hst.integers(min_value=1, max_value=1000),
                            min_size=1, max_size=10))
@pytest.mark.parametrize("range_func", [range, np.arange])
def test_get_shape_for_parameter_from_sequence(loop_shape, range_func) -> None:
    a = Parameter(name='a', initial_cache_value=10)
    loop_sequence = (range_func(x) for x in loop_shape)
    shape = get_shape_of_measurement(a, *loop_sequence)
    assert shape == {'a': tuple(loop_shape)}


@given(loop_shape=hst.lists(hst.integers(min_value=1), min_size=1, max_size=10))
def test_get_shape_for_array_parameter_from_len(loop_shape):
    a = ArraySetPointParam(name='a')
    shape = get_shape_of_measurement(a, *loop_shape)
    expected_shape = tuple(a.shape) + tuple(loop_shape)
    assert shape == {'a': expected_shape}


@given(loop_shape=hst.lists(hst.integers(min_value=1, max_value=1000),
                            min_size=1, max_size=10))
@pytest.mark.parametrize("range_func", [range, np.arange])
def test_get_shape_for_array_parameter_from_shape(loop_shape, range_func) -> None:
    a = ArraySetPointParam(name='a')
    loop_sequence = (range_func(x) for x in loop_shape)
    shape = get_shape_of_measurement(a, *loop_sequence)
    expected_shape = tuple(a.shape) + tuple(loop_shape)
    assert shape == {'a': expected_shape}


@given(loop_shape=hst.lists(hst.integers(min_value=1), min_size=1, max_size=10))
@pytest.mark.parametrize("multiparamtype", [MultiSetPointParam,
                                            Multi2DSetPointParam,
                                            Multi2DSetPointParam2Sizes])
def test_get_shape_for_multiparam_from_len(loop_shape, multiparamtype):
    param = multiparamtype(name='meas_param')
    shapes = get_shape_of_measurement(param, *loop_shape)
    expected_shapes = {}
    for i, name in enumerate(param.full_names):
        expected_shapes[name] = tuple(param.shapes[i]) + tuple(loop_shape)
    assert shapes == expected_shapes


@given(loop_shape=hst.lists(hst.integers(min_value=1, max_value=1000),
                            min_size=1, max_size=10))
@pytest.mark.parametrize("multiparamtype", [MultiSetPointParam,
                                            Multi2DSetPointParam,
                                            Multi2DSetPointParam2Sizes])
@pytest.mark.parametrize("range_func", [range, np.arange])
def test_get_shae_for_multiparam_from_shape(loop_shape, multiparamtype, range_func):
    param = multiparamtype(name='meas_param')
    loop_sequence = (range_func(x) for x in loop_shape)
    shapes = get_shape_of_measurement(param, *loop_sequence)
    expected_shapes = {}
    for i, name in enumerate(param.full_names):
        expected_shapes[name] = tuple(param.shapes[i]) + tuple(loop_shape)
    assert shapes == expected_shapes


@pytest.fixture(name='dummyinstrument')
def _make_dummy_instrument():
    inst = DummyChannelInstrument('dummyinstrument')
    try:
        yield inst
    finally:
        inst.close()


@given(loop_shape=hst.lists(hst.integers(min_value=1), min_size=1, max_size=10),
       n_points=hst.integers(min_value=1, max_value=1000))
def test_get_shape_for_pws_from_len(dummyinstrument, loop_shape, n_points):
    param = dummyinstrument.A.dummy_parameter_with_setpoints
    dummyinstrument.A.dummy_n_points(n_points)
    shapes = get_shape_of_measurement(param, *loop_shape)

    expected_shapes = {}
    expected_shapes[param.full_name] = tuple(param.vals.shape) + tuple(loop_shape)
    assert shapes == expected_shapes
    assert (dummyinstrument.A.dummy_n_points(),) == param.vals.shape


@pytest.mark.parametrize("range_func", [range, np.arange])
@given(loop_shape=hst.lists(hst.integers(min_value=1, max_value=1000), min_size=1, max_size=10),
       n_points=hst.integers(min_value=1, max_value=1000))
def test_get_shape_for_pws_from_shape(dummyinstrument, loop_shape, range_func,
                                      n_points):
    param = dummyinstrument.A.dummy_parameter_with_setpoints
    dummyinstrument.A.dummy_n_points(n_points)
    loop_sequence = (range_func(x) for x in loop_shape)
    shapes = get_shape_of_measurement(param, *loop_sequence)
    expected_shapes = {}
    expected_shapes[param.full_name] = tuple(param.vals.shape) + tuple(loop_shape)
    assert shapes == expected_shapes
    assert (dummyinstrument.A.dummy_n_points(),) == param.vals.shape

