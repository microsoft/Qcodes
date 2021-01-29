import hypothesis.strategies as hst

import pytest
from hypothesis import HealthCheck, given, settings


from qcodes.dataset.measurements import Measurement
from qcodes.instrument.parameter import expand_setpoints_helper


from .test_data_set_cache import (_prepare_setpoints_1d,
    _assert_parameter_data_is_identical)


@pytest.mark.parametrize("bg_writing", [False])
@pytest.mark.parametrize("setpoints_type", ['numeric'])
@settings(deadline=None, max_examples=10,
          suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(n_points=hst.integers(min_value=11, max_value=11))
def test_cache_1d(experiment, DAC, DMM, n_points, bg_writing,
                  channel_array_instrument, setpoints_type):

    setpoints_param, setpoints_values = _prepare_setpoints_1d(
        DAC, channel_array_instrument,
        n_points, setpoints_type
    )

    meas = Measurement()

    meas.register_parameter(setpoints_param)

    meas_parameters = (DMM.v1,
                       DMM.v2)
    for param in meas_parameters:
        meas.register_parameter(param, setpoints=(setpoints_param,))

    with meas.run(write_in_background=bg_writing) as datasaver:
        dataset = datasaver.dataset
        # _assert_parameter_data_is_identical(dataset.get_parameter_data(), dataset.cache.data())
        for i, v in enumerate(setpoints_values):
            setpoints_param.set(v)

            meas_vals = [(param, param.get()) for param in meas_parameters]

            datasaver.add_result((setpoints_param, v),
                                 *meas_vals)
            # datasaver.flush_data_to_database(block=True)
            # data = dataset.cache.data()
    #         _assert_parameter_data_is_identical(dataset.get_parameter_data(),
    #                                             data)
    # _assert_parameter_data_is_identical(dataset.get_parameter_data(),
    #                                     dataset.cache.data())
    # assert dataset.cache._loaded_from_completed_ds is True
    # _assert_parameter_data_is_identical(dataset.get_parameter_data(),
    #                                     dataset.cache.data())
