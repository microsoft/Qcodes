from typing import Dict

import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as hst
import pytest

from qcodes.dataset.measurements import Measurement

# parameterize over storage type, shape, structured and not structured, data types


@pytest.mark.parametrize("bg_writing", [True, False])
@settings(deadline=None, max_examples=10)
@given(n_points=hst.integers(min_value=1, max_value=101))
def test_cache_1d_num(experiment, DAC, DMM, n_points, bg_writing):
    meas = Measurement()

    meas.register_parameter(DAC.ch1)
    meas.register_parameter(DMM.v1, setpoints=(DAC.ch1,))

    with meas.run(write_in_background=bg_writing) as datasaver:
        dataset = datasaver.dataset
        for i, v in enumerate(np.linspace(-1, 1, n_points)):
            DAC.ch1.set(v)
            datasaver.add_result((DAC.ch1, v),
                                 (DMM.v1, DMM.v1.get()))
            datasaver.flush_data_to_database()
            if bg_writing:
                dataset._bg_writer.queue.join()
            data = dataset.cache.data()
            assert data[DMM.v1.full_name][DAC.ch1.full_name].shape == (i+1, )
            assert data[DMM.v1.full_name][DMM.v1.full_name].shape == (i+1,)
            _assert_parameter_data_is_identical(dataset.get_parameter_data(),
                                                data)


@pytest.mark.parametrize("bg_writing", [True, False])
@settings(deadline=None, max_examples=10)
@given(n_points_outer=hst.integers(min_value=1, max_value=11),
       n_points_inner=hst.integers(min_value=1, max_value=11))
def test_cache_2d_num(experiment, DAC, DMM, n_points_outer,
                      n_points_inner, bg_writing):
    meas = Measurement()

    meas.register_parameter(DAC.ch1)
    meas.register_parameter(DAC.ch2)
    meas.register_parameter(DMM.v1, setpoints=(DAC.ch1, DAC.ch2))

    i = 0
    with meas.run(write_in_background=bg_writing) as datasaver:
        dataset = datasaver.dataset
        for v1 in np.linspace(-1, 1, n_points_outer):
            for v2 in np.linspace(-1, 1, n_points_inner):
                DAC.ch1.set(v1)
                DAC.ch2.set(v2)
                datasaver.add_result((DAC.ch1, v1),
                                     (DAC.ch2, v2),
                                     (DMM.v1, DMM.v1.get()))
                datasaver.flush_data_to_database()
                if bg_writing:
                    dataset._bg_writer.queue.join()
                i += 1
                data = dataset.cache.data()
                assert data[DMM.v1.full_name][DAC.ch1.full_name].shape == (i, )
                assert data[DMM.v1.full_name][DMM.v1.full_name].shape == (i,)
                _assert_parameter_data_is_identical(dataset.get_parameter_data(),
                                                    data)


@pytest.mark.parametrize("bg_writing", [True, False])
@pytest.mark.parametrize("storage_type", ['numeric', 'array'])
@settings(deadline=None, max_examples=10)
@given(n_points=hst.integers(min_value=1, max_value=21))
def test_cache_1d_array_in_1d(experiment, DAC, channel_array_instrument, n_points, bg_writing, storage_type):
    param = channel_array_instrument.A.dummy_array_parameter
    meas = Measurement()
    meas.register_parameter(DAC.ch1, paramtype=storage_type)
    meas.register_parameter(param, setpoints=(DAC.ch1,), paramtype=storage_type)
    setpoint_name = "_".join((param.instrument.full_name, param.setpoint_names[0]))

    with meas.run(write_in_background=bg_writing) as datasaver:
        dataset = datasaver.dataset
        for i, v1 in enumerate(np.linspace(-1, 1, n_points)):
            datasaver.add_result((DAC.ch1, v1),
                                 (param, param.get()))
            datasaver.flush_data_to_database()
            if bg_writing:
                dataset._bg_writer.queue.join()
            data = dataset.cache.data()
            if storage_type == 'numeric':
                assert data[param.full_name][setpoint_name].shape == ((i + 1) * param.shape[0],)
                assert data[param.full_name][param.full_name].shape == ((i + 1) * param.shape[0],)
            else:
                assert data[param.full_name][setpoint_name].shape == (i+1,) + param.shape
                assert data[param.full_name][param.full_name].shape == (i+1,) + param.shape
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
