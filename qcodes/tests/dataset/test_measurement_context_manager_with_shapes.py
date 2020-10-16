import logging

import numpy as np


from qcodes.dataset.measurements import Measurement


def test_datasaver_1d_wrong_shape(experiment, DAC, DMM,
                                  caplog):
    meas = Measurement()
    meas.register_parameter(DAC.ch1)
    meas.register_parameter(DMM.v1, setpoints=(DAC.ch1,))

    n_points = 10
    n_points_expected = 5

    meas.set_shapes({DMM.v1.full_name: (n_points_expected,)})

    with meas.run() as datasaver:

        for set_v in np.linspace(0, 1, n_points):
            DAC.ch1()
            datasaver.add_result((DAC.ch1, set_v),
                                 (DMM.v1, DMM.v1()))

    ds = datasaver.dataset
    ds.get_parameter_data()
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
