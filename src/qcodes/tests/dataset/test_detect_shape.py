import hypothesis.strategies as hst
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from qcodes.dataset.descriptions.detect_shapes import detect_shape_of_measurement
from qcodes.instrument_drivers.mock_instruments import (
    ArraySetPointParam,
    Multi2DSetPointParam,
    Multi2DSetPointParam2Sizes,
    MultiSetPointParam,
)
from qcodes.parameters import Parameter
from qcodes.validators import Arrays

from .conftest import ArrayshapedParam


@given(loop_shape=hst.lists(hst.integers(min_value=1), min_size=1, max_size=10))
def test_get_shape_for_parameter_from_len(loop_shape) -> None:
    a = Parameter(name='a', initial_cache_value=10)
    shape = detect_shape_of_measurement((a,), loop_shape)
    assert shape == {'a': tuple(loop_shape)}


@given(loop_shape=hst.lists(hst.integers(min_value=1, max_value=1000),
                            min_size=1, max_size=10))
@pytest.mark.parametrize("range_func", [range, np.arange])
def test_get_shape_for_parameter_from_sequence(loop_shape, range_func) -> None:
    a = Parameter(name='a', initial_cache_value=10)
    loop_sequence = tuple(range_func(x) for x in loop_shape)
    shape = detect_shape_of_measurement((a,), loop_sequence)
    assert shape == {'a': tuple(loop_shape)}


@given(loop_shape=hst.lists(hst.integers(min_value=1), min_size=1, max_size=10))
def test_get_shape_for_array_parameter_from_len(loop_shape) -> None:
    a = ArraySetPointParam(name='a')
    shape = detect_shape_of_measurement((a,), loop_shape)
    expected_shape = tuple(loop_shape) + tuple(a.shape)
    assert shape == {'a': expected_shape}


@given(loop_shape=hst.lists(hst.integers(min_value=1, max_value=1000),
                            min_size=1, max_size=10))
@pytest.mark.parametrize("range_func", [range, np.arange])
def test_get_shape_for_array_parameter_from_shape(loop_shape, range_func) -> None:
    a = ArraySetPointParam(name='a')
    loop_sequence = tuple(range_func(x) for x in loop_shape)
    shape = detect_shape_of_measurement((a,), loop_sequence)
    expected_shape = tuple(loop_shape) + tuple(a.shape)
    assert shape == {'a': expected_shape}


@given(loop_shape=hst.lists(hst.integers(min_value=1), min_size=1, max_size=10))
@pytest.mark.parametrize("multiparamtype", [MultiSetPointParam,
                                            Multi2DSetPointParam,
                                            Multi2DSetPointParam2Sizes])
def test_get_shape_for_multiparam_from_len(loop_shape, multiparamtype) -> None:
    param = multiparamtype(name='meas_param')
    shapes = detect_shape_of_measurement((param,), loop_shape)
    expected_shapes = {}
    for i, name in enumerate(param.full_names):
        expected_shapes[name] = tuple(loop_shape) + tuple(param.shapes[i])
    assert shapes == expected_shapes


@given(loop_shape=hst.lists(hst.integers(min_value=1, max_value=1000),
                            min_size=1, max_size=10))
@pytest.mark.parametrize("multiparamtype", [MultiSetPointParam,
                                            Multi2DSetPointParam,
                                            Multi2DSetPointParam2Sizes])
@pytest.mark.parametrize("range_func", [range, np.arange])
def test_get_shape_for_multiparam_from_shape(
        loop_shape,
        multiparamtype,
        range_func
) -> None:
    param = multiparamtype(name='meas_param')
    loop_sequence = tuple(range_func(x) for x in loop_shape)
    shapes = detect_shape_of_measurement((param,), loop_sequence)
    expected_shapes = {}
    for i, name in enumerate(param.full_names):
        expected_shapes[name] = tuple(loop_shape) + tuple(param.shapes[i])
    assert shapes == expected_shapes


@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(loop_shape=hst.lists(hst.integers(min_value=1), min_size=1, max_size=10),
       n_points=hst.integers(min_value=1, max_value=1000))
def test_get_shape_for_pws_from_len(dummyinstrument, loop_shape, n_points) -> None:
    param = dummyinstrument.A.dummy_parameter_with_setpoints
    dummyinstrument.A.dummy_n_points(n_points)
    shapes = detect_shape_of_measurement((param,), loop_shape)

    expected_shapes = {}
    expected_shapes[param.full_name] = (tuple(loop_shape) +
                                        tuple(param.vals.shape))
    assert shapes == expected_shapes
    assert (dummyinstrument.A.dummy_n_points(),) == param.vals.shape


@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
@pytest.mark.parametrize("range_func", [range, np.arange])
@given(
    loop_shape=hst.lists(
        hst.integers(min_value=1, max_value=1000),
        min_size=1,
        max_size=10
    ),
    n_points=hst.integers(min_value=1, max_value=1000)
)
def test_get_shape_for_pws_from_shape(
    dummyinstrument, loop_shape, range_func, n_points
) -> None:
    param = dummyinstrument.A.dummy_parameter_with_setpoints
    dummyinstrument.A.dummy_n_points(n_points)
    loop_sequence = tuple(range_func(x) for x in loop_shape)
    shapes = detect_shape_of_measurement((param,), loop_sequence)
    expected_shapes = {}
    expected_shapes[param.full_name] = (tuple(loop_shape) +
                                        tuple(param.vals.shape))
    assert shapes == expected_shapes
    assert (dummyinstrument.A.dummy_n_points(),) == param.vals.shape


@given(
    loop_shape=hst.lists(
        hst.integers(min_value=1, max_value=1000),
        min_size=1,
        max_size=10
    )
)
def test_get_shape_for_param_with_array_validator_from_shape(loop_shape) -> None:
    param = ArrayshapedParam(name="paramwitharrayval", vals=Arrays(shape=(10,)))

    shapes = detect_shape_of_measurement((param,), loop_shape)
    assert isinstance(param.vals, Arrays)
    assert param.vals.shape is not None
    assert shapes == {"paramwitharrayval": tuple(loop_shape) + param.vals.shape}


@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(
    loop_shape=hst.lists(hst.integers(min_value=1), min_size=1, max_size=10),
    n_points_1=hst.integers(min_value=1, max_value=1000),
    n_points_2=hst.integers(min_value=1, max_value=1000),
)
def test_get_shape_for_multiple_parameters(
    dummyinstrument, loop_shape, n_points_1, n_points_2
) -> None:
    param1 = dummyinstrument.A.dummy_parameter_with_setpoints
    dummyinstrument.A.dummy_n_points(n_points_1)
    param2 = dummyinstrument.B.dummy_parameter_with_setpoints
    dummyinstrument.B.dummy_n_points(n_points_2)
    shapes = detect_shape_of_measurement((param1, param2), loop_shape)

    expected_shapes = {}
    expected_shapes[param1.full_name] = (tuple(loop_shape) +
                                         tuple(param1.vals.shape))
    expected_shapes[param2.full_name] = (tuple(loop_shape) +
                                         tuple(param2.vals.shape))
    assert shapes == expected_shapes
    assert (dummyinstrument.A.dummy_n_points(),) == param1.vals.shape
    assert (dummyinstrument.B.dummy_n_points(),) == param2.vals.shape
