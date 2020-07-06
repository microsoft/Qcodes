import numpy as np

from qcodes.dataset.measurements import Measurement

# parameterize over storage type, shape, structured and not structured, data types


def test_cache_1d(experiment, DAC, DMM):
    meas = Measurement()

    meas.register_parameter(DAC.ch1)
    meas.register_parameter(DMM.v1, setpoints=(DAC.ch1,))

    with meas.run() as datasaver:
        dataset = datasaver.dataset
        for i, v in enumerate(np.linspace(-1, 1, 1001)):
            DAC.ch1.set(v)
            datasaver.add_result((DAC.ch1, v),
                                 (DMM.v1, DMM.v1.get()))
            datasaver.flush_data_to_database()
            data = dataset.cache.data()
            assert data[DMM.v1.full_name][DAC.ch1.full_name].shape == (i+1, )
            assert data[DMM.v1.full_name][DMM.v1.full_name].shape == (i+1,)


def test_cache_2d(experiment, DAC, DMM):
    meas = Measurement()

    meas.register_parameter(DAC.ch1)
    meas.register_parameter(DAC.ch2)
    meas.register_parameter(DMM.v1, setpoints=(DAC.ch1, DAC.ch2))

    i = 0
    with meas.run() as datasaver:
        dataset = datasaver.dataset
        for v1 in np.linspace(-1, 1, 11):
            for v2 in np.linspace(-1, 1, 11):
                DAC.ch1.set(v1)
                DAC.ch2.set(v2)
                datasaver.add_result((DAC.ch1, v1),
                                     (DAC.ch2, v2),
                                     (DMM.v1, DMM.v1.get()))
                datasaver.flush_data_to_database()
                i += 1
                data = dataset.cache.data()
                assert data[DMM.v1.full_name][DAC.ch1.full_name].shape == (i, )
                assert data[DMM.v1.full_name][DMM.v1.full_name].shape == (i,)


