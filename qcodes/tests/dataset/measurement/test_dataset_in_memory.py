import numpy as np

from qcodes.dataset.data_set_in_memory import DataSetInMem


def test_dataset_in_memory_smoke_test(meas_with_registered_param, DMM, DAC):
    with meas_with_registered_param.run(dataset_class=DataSetInMem) as datasaver:
        for set_v in np.linspace(0, 25, 10):
            DAC.ch1.set(set_v)
            get_v = DMM.v1()
            datasaver.add_result((DAC.ch1, set_v),
                                 (DMM.v1, get_v))
