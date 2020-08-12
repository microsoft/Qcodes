from typing import Dict

import hypothesis.strategies as hst
import numpy as np
import pytest
from hypothesis import given, settings

from qcodes.dataset.measurements import Measurement
from qcodes.instrument.parameter import expand_setpoints_helper


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.parametrize("setpoints_type", ['text', 'numeric'])
@settings(deadline=None, max_examples=10)
@given(n_points=hst.integers(min_value=1, max_value=101))
def test_cache_1d(experiment, DAC, DMM, n_points, bg_writing,
                  channel_array_instrument, setpoints_type):

    setpoints_param, setpoints_values = _prepare_setpoints_1d(DAC, channel_array_instrument,
                                                                                   n_points, setpoints_type)

    meas = Measurement()

    meas.register_parameter(setpoints_param)

    meas_parameters = (DMM.v1,
                       channel_array_instrument.A.dummy_multi_parameter,
                       channel_array_instrument.A.dummy_scalar_multi_parameter,
                       channel_array_instrument.A.dummy_2d_multi_parameter,
                       channel_array_instrument.A.dummy_2d_multi_parameter_2,
                       channel_array_instrument.A.dummy_array_parameter,
                       channel_array_instrument.A.dummy_complex_array_parameter,
                       channel_array_instrument.A.dummy_complex,
                       channel_array_instrument.A.dummy_parameter_with_setpoints,
                       channel_array_instrument.A.dummy_parameter_with_setpoints_complex,
                       )
    channel_array_instrument.A.dummy_start(0)
    channel_array_instrument.A.dummy_stop(10)
    channel_array_instrument.A.dummy_n_points(10)
    for param in meas_parameters:
        meas.register_parameter(param, setpoints=(setpoints_param,))

    with meas.run(write_in_background=bg_writing) as datasaver:
        dataset = datasaver.dataset
        _assert_parameter_data_is_identical(dataset.get_parameter_data(), dataset.cache.data())
        for i, v in enumerate(setpoints_values):
            setpoints_param.set(v)

            meas_vals = [(param, param.get()) for param in meas_parameters[:-2]]
            meas_vals += expand_setpoints_helper(meas_parameters[-2])
            meas_vals += expand_setpoints_helper(meas_parameters[-1])

            datasaver.add_result((setpoints_param, v),
                                 *meas_vals)
            datasaver.flush_data_to_database(block=True)
            data = dataset.cache.data()
            _assert_parameter_data_is_identical(dataset.get_parameter_data(),
                                                data)
    _assert_parameter_data_is_identical(dataset.get_parameter_data(),
                                        dataset.cache.data())
    assert dataset.cache._loaded_from_completed_ds is True
    _assert_parameter_data_is_identical(dataset.get_parameter_data(),
                                        dataset.cache.data())


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.parametrize("setpoints_type", ['text', 'numeric'])
@settings(deadline=None, max_examples=10)
@given(n_points=hst.integers(min_value=1, max_value=101))
def test_cache_1d_every_other_point(experiment, DAC, DMM, n_points, bg_writing,
                                    channel_array_instrument, setpoints_type):

    setpoints_param, setpoints_values = _prepare_setpoints_1d(DAC, channel_array_instrument,
                                                                                   n_points, setpoints_type)

    meas = Measurement()

    meas.register_parameter(setpoints_param)

    meas_parameters = (DMM.v1,
                       channel_array_instrument.A.temperature,
                       channel_array_instrument.B.temperature
                       )
    for param in meas_parameters:
        meas.register_parameter(param, setpoints=(setpoints_param,))

    with meas.run(write_in_background=bg_writing) as datasaver:
        dataset = datasaver.dataset
        _assert_parameter_data_is_identical(dataset.get_parameter_data(), dataset.cache.data())
        for i, v in enumerate(setpoints_values):
            setpoints_param.set(v)

            meas_vals = [(param, param.get()) for param in meas_parameters]

            if i % 2 == 0:
                datasaver.add_result((setpoints_param, v),
                                     *meas_vals)
            else:
                datasaver.add_result((setpoints_param, v),
                                     *meas_vals[0:2])
            datasaver.flush_data_to_database(block=True)
            data = dataset.cache.data()
            assert len(data['dummy_channel_inst_ChanA_temperature']['dummy_channel_inst_ChanA_temperature']) == i + 1
            assert len(data['dummy_channel_inst_ChanB_temperature']['dummy_channel_inst_ChanB_temperature']) == i//2 + 1
            _assert_parameter_data_is_identical(dataset.get_parameter_data(),
                                                data)
    _assert_parameter_data_is_identical(dataset.get_parameter_data(),
                                        dataset.cache.data())
    assert dataset.cache._loaded_from_completed_ds is True
    _assert_parameter_data_is_identical(dataset.get_parameter_data(),
                                        dataset.cache.data())



def _prepare_setpoints_1d(DAC, channel_array_instrument, n_points, setpoints_type):
    if setpoints_type == 'numeric':
        setpoints_param = DAC.ch1
        setpoints_values = np.linspace(-1, 1, n_points)
    else:
        setpoints_param = channel_array_instrument.A.dummy_text
        setpoints_values = ['A', 'B', 'C', 'D']
    return setpoints_param, setpoints_values


@pytest.mark.parametrize("bg_writing", [True, False])
@settings(deadline=None, max_examples=10)
@given(n_points_outer=hst.integers(min_value=1, max_value=11),
       n_points_inner=hst.integers(min_value=1, max_value=11))
def test_cache_2d(experiment, DAC, DMM, n_points_outer,
                      n_points_inner, bg_writing, channel_array_instrument):
    meas = Measurement()

    meas.register_parameter(DAC.ch1)
    meas.register_parameter(DAC.ch2)

    meas_parameters = (DMM.v1,
                       channel_array_instrument.A.dummy_multi_parameter,
                       channel_array_instrument.A.dummy_scalar_multi_parameter,
                       channel_array_instrument.A.dummy_2d_multi_parameter,
                       channel_array_instrument.A.dummy_2d_multi_parameter_2,
                       channel_array_instrument.A.dummy_array_parameter,
                       channel_array_instrument.A.dummy_complex_array_parameter,
                       channel_array_instrument.A.dummy_complex,
                       channel_array_instrument.A.dummy_parameter_with_setpoints,
                       channel_array_instrument.A.dummy_parameter_with_setpoints_complex,
                       )
    channel_array_instrument.A.dummy_start(0)
    channel_array_instrument.A.dummy_stop(10)
    channel_array_instrument.A.dummy_n_points(10)
    for param in meas_parameters:
        meas.register_parameter(param, setpoints=(DAC.ch1, DAC.ch2))
    n_rows_written = 0
    with meas.run(write_in_background=bg_writing) as datasaver:
        dataset = datasaver.dataset
        _assert_parameter_data_is_identical(dataset.get_parameter_data(), dataset.cache.data())
        for v1 in np.linspace(-1, 1, n_points_outer):
            for v2 in np.linspace(-1, 1, n_points_inner):
                DAC.ch1.set(v1)
                DAC.ch2.set(v2)
                meas_vals = [(param, param.get()) for param in meas_parameters[:-2]]
                meas_vals += expand_setpoints_helper(meas_parameters[-2])
                meas_vals += expand_setpoints_helper(meas_parameters[-1])

                datasaver.add_result((DAC.ch1, v1),
                                     (DAC.ch2, v2),
                                     *meas_vals)
                datasaver.flush_data_to_database(block=True)
                n_rows_written += 1
                data = dataset.cache.data()
                _assert_parameter_data_is_identical(dataset.get_parameter_data(),
                                                    data)
    _assert_parameter_data_is_identical(dataset.get_parameter_data(),
                                        dataset.cache.data())


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.parametrize("storage_type", ['numeric', 'array', None])
@settings(deadline=None, max_examples=10)
@given(n_points_outer=hst.integers(min_value=1, max_value=11),
       n_points_inner=hst.integers(min_value=1, max_value=11))
def test_cache_2d_num_with_multiple_storage_types(experiment, DAC, DMM, n_points_outer,
                      n_points_inner, bg_writing, storage_type):
    meas = Measurement()

    meas.register_parameter(DAC.ch1, paramtype=storage_type)
    meas.register_parameter(DAC.ch2, paramtype=storage_type)
    meas.register_parameter(DMM.v1, setpoints=(DAC.ch1, DAC.ch2), paramtype=storage_type)
    array_used = _array_param_used_in_tree(meas)
    n_rows_written = 0
    with meas.run(write_in_background=bg_writing) as datasaver:
        dataset = datasaver.dataset
        _assert_parameter_data_is_identical(dataset.get_parameter_data(), dataset.cache.data())
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
    _assert_parameter_data_is_identical(dataset.get_parameter_data(),
                                        dataset.cache.data())


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
                # with explicit array types the shape is incorrect
                # https://github.com/QCoDeS/Qcodes/issues/2105
                assert data[param.full_name][DAC.ch1.full_name].shape == shape
            assert data[param.full_name][setpoint_name].shape == shape
            assert data[param.full_name][param.full_name].shape == shape
            _assert_parameter_data_is_identical(dataset.get_parameter_data(),
                                                data)
    _assert_parameter_data_is_identical(dataset.get_parameter_data(),
                                        dataset.cache.data())


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
                    # with explicit array types the shape is incorrect
                    # https://github.com/QCoDeS/Qcodes/issues/2105
                    assert data[subparam][DAC.ch1.full_name].shape == expected_shape
                for setpoint_name in param.setpoint_full_names[j]:
                    assert data[subparam][setpoint_name].shape == expected_shape
            _assert_parameter_data_is_identical(dataset.get_parameter_data(),
                                                data)
    _assert_parameter_data_is_identical(dataset.get_parameter_data(),
                                        dataset.cache.data())


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.parametrize("storage_type", ['array', None])
@pytest.mark.parametrize("outer_param_type", ['numeric', 'text'])
@settings(deadline=None, max_examples=10)
@given(n_points=hst.integers(min_value=1, max_value=21))
def test_cache_complex_array_param_in_1d(experiment, DAC, channel_array_instrument, n_points, bg_writing, storage_type, outer_param_type):
    param = channel_array_instrument.A.dummy_complex_array_parameter
    meas = Measurement()
    if outer_param_type == 'numeric':
        outer_param = DAC.ch1
        outer_setpoints = np.linspace(-1, 1, n_points)
        outer_storage_type = storage_type
    else:
        outer_param = channel_array_instrument.A.dummy_text
        outer_setpoints = ['A', 'B', 'C', 'D']
        outer_storage_type = 'text'
    meas.register_parameter(outer_param, paramtype=outer_storage_type)
    meas.register_parameter(param, setpoints=(outer_param,), paramtype=storage_type)
    array_used = _array_param_used_in_tree(meas)
    with meas.run(write_in_background=bg_writing) as datasaver:
        dataset = datasaver.dataset
        for i, v1 in enumerate(outer_setpoints):
            datasaver.add_result((outer_param, v1),
                                 (param, param.get()))
            datasaver.flush_data_to_database(block=True)
            data = dataset.cache.data()
            n_rows_written = i+1

            if array_used:
                expected_shape = (n_rows_written,) + param.shape
            else:
                expected_shape = n_rows_written * np.prod(param.shape)
            assert data[param.full_name][param.full_name].shape == expected_shape
            if storage_type != 'array':
                # with explicit array types the shape is incorrect
                # https://github.com/QCoDeS/Qcodes/issues/2105
                assert data[param.full_name][outer_param.full_name].shape == expected_shape
            for setpoint_name in param.setpoint_full_names:
                assert data[param.full_name][setpoint_name].shape == expected_shape
            _assert_parameter_data_is_identical(dataset.get_parameter_data(),
                                                data)
    _assert_parameter_data_is_identical(dataset.get_parameter_data(),
                                        dataset.cache.data())


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
