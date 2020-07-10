from typing import Dict

import hypothesis.strategies as hst
import numpy as np
import pytest
from hypothesis import given, settings

from qcodes.dataset.measurements import Measurement

# parameterize over storage type, shape, structured and not structured, data types


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.parametrize("storage_type", ['numeric', 'array', None])
@settings(deadline=None, max_examples=10)
@given(n_points=hst.integers(min_value=1, max_value=101))
def test_cache_1d_num(experiment, DAC, DMM, n_points, bg_writing, storage_type):
    meas = Measurement()

    meas.register_parameter(DAC.ch1, paramtype=storage_type)
    meas.register_parameter(DMM.v1, setpoints=(DAC.ch1,), paramtype=storage_type)

    array_used = _array_param_used_in_tree(meas)

    with meas.run(write_in_background=bg_writing) as datasaver:
        dataset = datasaver.dataset
        for i, v in enumerate(np.linspace(-1, 1, n_points)):
            DAC.ch1.set(v)
            datasaver.add_result((DAC.ch1, v),
                                 (DMM.v1, DMM.v1.get()))
            datasaver.flush_data_to_database(block=True)
            data = dataset.cache.data()
            n_rows_written = i+1
            if array_used:
                shape = (n_rows_written, 1)
            else:
                shape = (n_rows_written,)
            assert data[DMM.v1.full_name][DMM.v1.full_name].shape == shape
            assert data[DMM.v1.full_name][DAC.ch1.full_name].shape == shape
            _assert_parameter_data_is_identical(dataset.get_parameter_data(),
                                                data)


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.parametrize("storage_type", ['numeric', 'array', None])
@settings(deadline=None, max_examples=10)
@given(n_points_outer=hst.integers(min_value=1, max_value=11),
       n_points_inner=hst.integers(min_value=1, max_value=11))
def test_cache_2d_num(experiment, DAC, DMM, n_points_outer,
                      n_points_inner, bg_writing, storage_type):
    meas = Measurement()

    meas.register_parameter(DAC.ch1, paramtype=storage_type)
    meas.register_parameter(DAC.ch2, paramtype=storage_type)
    meas.register_parameter(DMM.v1, setpoints=(DAC.ch1, DAC.ch2), paramtype=storage_type)
    array_used = _array_param_used_in_tree(meas)
    n_rows_written = 0
    with meas.run(write_in_background=bg_writing) as datasaver:
        dataset = datasaver.dataset
        for v1 in np.linspace(-1, 1, n_points_outer):
            for v2 in np.linspace(-1, 1, n_points_inner):
                DAC.ch1.set(v1)
                DAC.ch2.set(v2)
                datasaver.add_result((DAC.ch1, v1),
                                     (DAC.ch2, v2),
                                     (DMM.v1, DMM.v1.get()))
                datasaver.flush_data_to_database(block=True)
                n_rows_written += 1
                data = dataset.cache.data()
                if array_used:
                    shape = (n_rows_written, 1)
                else:
                    shape = (n_rows_written,)
                assert data[DMM.v1.full_name][DMM.v1.full_name].shape == shape
                assert data[DMM.v1.full_name][DAC.ch1.full_name].shape == shape
                _assert_parameter_data_is_identical(dataset.get_parameter_data(),
                                                    data)


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.parametrize("storage_type", ['numeric', 'array', None])
@settings(deadline=None, max_examples=10)
@given(n_points=hst.integers(min_value=1, max_value=21))
def test_cache_1d_array_in_1d(experiment, DAC, channel_array_instrument, n_points, bg_writing, storage_type):
    param = channel_array_instrument.A.dummy_array_parameter
    meas = Measurement()
    meas.register_parameter(DAC.ch1, paramtype=storage_type)
    meas.register_parameter(param, setpoints=(DAC.ch1,), paramtype=storage_type)
    array_used = _array_param_used_in_tree(meas)

    setpoint_name = "_".join((param.instrument.full_name, param.setpoint_names[0]))

    with meas.run(write_in_background=bg_writing) as datasaver:
        dataset = datasaver.dataset
        for i, v1 in enumerate(np.linspace(-1, 1, n_points)):
            datasaver.add_result((DAC.ch1, v1),
                                 (param, param.get()))
            datasaver.flush_data_to_database(block=True)
            data = dataset.cache.data()
            n_rows_written = i+1
            if array_used:
                shape = (n_rows_written, param.shape[0])
            else:
                shape = (n_rows_written * param.shape[0],)
            if storage_type != 'array':
                # TODO the expansion both for cache and get_parameter_data is buggy iff all types are array
                assert data[param.full_name][DAC.ch1.full_name].shape == shape
            assert data[param.full_name][setpoint_name].shape == shape
            assert data[param.full_name][param.full_name].shape == shape
            _assert_parameter_data_is_identical(dataset.get_parameter_data(),
                                                data)


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.parametrize("storage_type", ['numeric', 'array', None])
@settings(deadline=None, max_examples=10)
@given(n_points=hst.integers(min_value=1, max_value=21))
def test_cache_multiparam_in_1d(experiment, DAC, channel_array_instrument, n_points, bg_writing, storage_type):
    param = channel_array_instrument.A.dummy_2d_multi_parameter
    meas = Measurement()
    meas.register_parameter(DAC.ch1, paramtype=storage_type)
    meas.register_parameter(param, setpoints=(DAC.ch1,), paramtype=storage_type)
    array_used = _array_param_used_in_tree(meas)

    with meas.run(write_in_background=bg_writing) as datasaver:
        dataset = datasaver.dataset
        for i, v1 in enumerate(np.linspace(-1, 1, n_points)):
            datasaver.add_result((DAC.ch1, v1),
                                 (param, param.get()))
            datasaver.flush_data_to_database(block=True)
            data = dataset.cache.data()
            n_rows_written = i+1
            for j, subparam in enumerate(param.names):
                if array_used:
                    expected_shape = (n_rows_written,) + param.shapes[j]
                else:
                    expected_shape = n_rows_written * np.prod(param.shapes[j])
                assert data[subparam][subparam].shape == expected_shape
                if storage_type != 'array':
                    # TODO the expansion both for cache and get_parameter_data is buggy iff all types are array
                    assert data[subparam][DAC.ch1.full_name].shape == expected_shape
                for setpoint_name in param.setpoint_full_names[j]:
                    assert data[subparam][setpoint_name].shape == expected_shape
            _assert_parameter_data_is_identical(dataset.get_parameter_data(),
                                                data)


def _assert_parameter_data_is_identical(expected: Dict[str, Dict[str, np.ndarray]],
                                        actual: Dict[str, Dict[str, np.ndarray]]):
    assert expected.keys() == actual.keys()

    for outer_key in expected.keys():
        expected_inner = expected[outer_key]
        actual_inner = actual[outer_key]
        assert expected_inner.keys() == actual_inner.keys()
        for inner_key in expected_inner.keys():
            np.testing.assert_array_equal(expected_inner[inner_key],
                                          actual_inner[inner_key])


def _array_param_used_in_tree(measurement: Measurement) -> bool:
    found_array = False
    for paramspecbase in measurement.parameters.values():
        if paramspecbase.type == 'array':
            found_array = True
    return found_array
