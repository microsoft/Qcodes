import logging

import numpy as np
import hypothesis.strategies as hst
from hypothesis import HealthCheck, given, example, settings

from qcodes.dataset.measurements import Measurement


@given(n_points=hst.integers(min_value=1, max_value=100))
@example(n_points=5)
@settings(deadline=None, suppress_health_check=(HealthCheck.function_scoped_fixture,))
def test_datasaver_1d(experiment, DAC, DMM, caplog,
                      n_points):
    meas = Measurement()
    meas.register_parameter(DAC.ch1)
    meas.register_parameter(DMM.v1, setpoints=(DAC.ch1,))

    n_points_expected = 5

    meas.set_shapes({DMM.v1.full_name: (n_points_expected,)})

    with meas.run() as datasaver:

        for set_v in np.linspace(0, 1, n_points):
            DAC.ch1()
            datasaver.add_result((DAC.ch1, set_v),
                                 (DMM.v1, DMM.v1()))

    ds = datasaver.dataset
    caplog.clear()
    data = ds.get_parameter_data()

    for dataarray in data[DMM.v1.full_name].values():
        assert dataarray.shape == (n_points,)

    if n_points == n_points_expected:
        assert len(caplog.record_tuples) == 0
    elif n_points > n_points_expected:
        assert len(caplog.record_tuples) == 2
        exp_module = "qcodes.dataset.sqlite.queries"
        exp_level = logging.WARNING
        exp_msg = ("Tried to set data shape for {} in "
                   "dataset {} "
                   "from metadata when loading "
                   "but found inconsistent lengths {} and {}")
        assert caplog.record_tuples[0] == (exp_module,
                                           exp_level,
                                           exp_msg.format(DMM.v1.full_name,
                                                          DMM.v1.full_name,
                                                          n_points,
                                                          n_points_expected))
        assert caplog.record_tuples[1] == (exp_module,
                                           exp_level,
                                           exp_msg.format(DAC.ch1.full_name,
                                                          DMM.v1.full_name,
                                                          n_points,
                                                          n_points_expected))


@settings(deadline=None, suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(n_points_1=hst.integers(min_value=1, max_value=50),
       n_points_2=hst.integers(min_value=1, max_value=50))
@example(n_points_1=5, n_points_2=10)
def test_datasaver_2d(experiment, DAC, DMM, caplog,
                      n_points_1, n_points_2):
    meas = Measurement()
    meas.register_parameter(DAC.ch1)
    meas.register_parameter(DAC.ch2)
    meas.register_parameter(DMM.v1, setpoints=(DAC.ch1,
                                               DAC.ch2))

    n_points_expected_1 = 5
    n_points_expected_2 = 10

    meas.set_shapes({DMM.v1.full_name: (n_points_expected_1,
                                        n_points_expected_2,)})

    with meas.run() as datasaver:

        for set_v_1 in np.linspace(0, 1, n_points_1):
            for set_v_2 in np.linspace(0, 1, n_points_2):
                datasaver.add_result((DAC.ch1, set_v_1),
                                     (DAC.ch2, set_v_2),
                                     (DMM.v1, DMM.v1()))

    ds = datasaver.dataset
    caplog.clear()
    data = ds.get_parameter_data()

    if n_points_1*n_points_2 == n_points_expected_1*n_points_expected_2:
        assert len(caplog.record_tuples) == 0
        for dataarray in data[DMM.v1.full_name].values():
            assert dataarray.shape == (n_points_expected_1, n_points_expected_2)
    elif n_points_1*n_points_2 > n_points_expected_1*n_points_expected_2:
        assert len(caplog.record_tuples) == 3
        exp_module = "qcodes.dataset.sqlite.queries"
        exp_level = logging.WARNING
        exp_msg = ("Tried to set data shape for {} in "
                   "dataset {} "
                   "from metadata when loading "
                   "but found inconsistent lengths {} and {}")
        assert caplog.record_tuples[0] == (
            exp_module,
            exp_level,
            exp_msg.format(
                DMM.v1.full_name,
                DMM.v1.full_name,
                n_points_1*n_points_2,
                n_points_expected_1*n_points_expected_2
            )
        )
        assert caplog.record_tuples[1] == (
            exp_module,
            exp_level,
            exp_msg.format(
                DAC.ch1.full_name,
                DMM.v1.full_name,
                n_points_1*n_points_2,
                n_points_expected_1*n_points_expected_2)
        )
        assert caplog.record_tuples[2] == (
            exp_module,
            exp_level,
            exp_msg.format(
                DAC.ch2.full_name,
                DMM.v1.full_name,
                n_points_1*n_points_2,
                n_points_expected_1*n_points_expected_2
            )
        )
