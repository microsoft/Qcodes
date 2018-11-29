from typing import List
import json

from qcodes.dataset.measurements import Measurement
from qcodes.data.data_set import load_data
from qcodes.data.data_set import DataSet as OldDataSet
import numpy as np


def setup_measurement(dataset: OldDataSet) -> Measurement:
    """
    Register parameters for all DataArrays in a given QCoDeS legacy dataset

    This tries to infer the name, label and unit along with any setpoints
    for the given array.
    """
    meas = Measurement()
    for arrayname, array in dataset.arrays.items():
        if array.is_setpoint:
            setarrays = None
        else:
            setarrays = [setarray.name for setarray in array.set_arrays]
        meas.register_custom_parameter(name=array.name,
                                       label=array.label,
                                       unit=array.unit,
                                       setpoints = setarrays
                                       )
    return meas


def store_array_to_database(datasaver, array):
    dims = len(array.shape)
    if dims == 2:
        for index1, i in enumerate(array.set_arrays[0]):
            for index2, j in enumerate(array.set_arrays[1][index1]):
                datasaver.add_result((array.set_arrays[0].name, i),
                                     (array.set_arrays[1].name, j),
                                     (array.name, array[index1,index2]))
    elif dims == 1:
        for index, i in enumerate(array.set_arrays[0]):
            datasaver.add_result((array.set_arrays[0].name, i),
                                 (array.name, array[index]))
    else:
        raise NotImplementedError('The exporter only currently handles 1 and 2 Dimentional data')
    return datasaver.run_id


def store_array_to_database_alt(meas, array):
    dims = len(array.shape)
    if dims == 2:
        outer_data = np.empty(array.shape[1])
        with meas.run() as datasaver:
            for index1, i in enumerate(array.set_arrays[0]):
                outer_data[:] = i
                datasaver.add_result((array.set_arrays[0].name, outer_data),
                                     (array.set_arrays[1].name, array.set_arrays[1][index1,:]),
                                     (array.name, array[index1,:]))
    elif dims == 1:
        with meas.run() as datasaver:
            for index, i in enumerate(array.set_arrays[0]):
                datasaver.add_result((array.set_arrays[0].name, i),
                                     (array.name, array[index]))
    else:
        raise NotImplementedError('The exporter only currently handles 1 and 2 Dimentional data')
    return datasaver.run_id


def import_dat_file(location: str) -> List[int]:
    """
    This imports a QCoDeS legacy DataSet
    """


    loaded_data = load_data(location)
    meas = setup_measurement(loaded_data)
    run_ids = []
    with meas.run() as datasaver:
        datasaver.dataset.add_metadata('snapshot', json.dumps(loaded_data.snapshot()))
        for arrayname, array in loaded_data.arrays.items():
            if not array.is_setpoint:
                run_id = store_array_to_database(datasaver, array)
                run_ids.append(run_id)
    return run_ids
