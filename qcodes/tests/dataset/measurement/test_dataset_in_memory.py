import os

import numpy as np

from qcodes.dataset.data_set_in_memory import DataSetInMem


# def test_dataset_in_memory_smoke_test(meas_with_registered_param, DMM, DAC, tmp_path):
#     with meas_with_registered_param.run(dataset_class=DataSetInMem) as datasaver:
#         for set_v in np.linspace(0, 25, 10):
#             DAC.ch1.set(set_v)
#             get_v = DMM.v1()
#             datasaver.add_result((DAC.ch1, set_v), (DMM.v1, get_v))
#
#     dataset = datasaver.dataset
#     os.chdir(tmp_path)
#     dataset.export(export_type="netcdf", path=".")
#     loaded_ds = DataSetInMem.load_from_netcdf("qcodes_1.nc")
#     assert dataset.the_same_dataset_as(loaded_ds)


# todo missing from runs table
# snapshot, completed timestamp, parameters (do we care), verify other metadata
# When should metadata be added. In the old dataset it used to be added as
# soon as you call add_metadata


# add a test to import from 0.26 data (missing parent dataset links)
