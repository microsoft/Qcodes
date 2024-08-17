from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from qcodes.dataset.measurements import DataSaver, Measurement

if TYPE_CHECKING:
    from pathlib import Path

    from qcodes_loop.data.data_array import DataArray
    from qcodes_loop.data.data_set import DataSet as OldDataSet

    from qcodes.dataset.experiment_container import Experiment


def setup_measurement(
    dataset: OldDataSet, exp: Experiment | None = None
) -> Measurement:
    """
    Register parameters for all :class:.`DataArrays` in a given QCoDeS legacy dataset

    This tries to infer the `name`, `label` and `unit` along with any `setpoints`
    for the given array.

    Args:
        dataset: Legacy dataset to register parameters from.
        exp: experiment that the legacy dataset should be bound to. If
            None the default experiment is used. See the
            docs of :class:`qcodes.dataset.Measurement` for more details.
    """
    meas = Measurement(exp=exp)
    for arrayname, array in dataset.arrays.items():
        if array.is_setpoint:
            setarrays = None
        else:
            setarrays = [setarray.array_id for setarray in array.set_arrays]
        meas.register_custom_parameter(
            name=array.array_id, label=array.label, unit=array.unit, setpoints=setarrays
        )
    return meas


def store_array_to_database(datasaver: DataSaver, array: DataArray) -> int:
    assert array.shape is not None
    dims = len(array.shape)
    assert array.array_id is not None
    if dims == 2:
        for index1, i in enumerate(array.set_arrays[0]):
            for index2, j in enumerate(array.set_arrays[1][index1]):
                datasaver.add_result(
                    (array.set_arrays[0].array_id, i),
                    (array.set_arrays[1].array_id, j),
                    (array.array_id, array[index1, index2]),
                )
    elif dims == 1:
        for index, i in enumerate(array.set_arrays[0]):
            datasaver.add_result(
                (array.set_arrays[0].array_id, i), (array.array_id, array[index])
            )
    else:
        raise NotImplementedError(
            "The exporter only currently handles 1 and 2 Dimensional data"
        )
    return datasaver.run_id


def store_array_to_database_alt(meas: Measurement, array: DataArray) -> int:
    assert array.shape is not None
    dims = len(array.shape)
    assert array.array_id is not None
    if dims == 2:
        outer_data = np.empty(
            array.shape[1]  # pyright: ignore[reportGeneralTypeIssues]
        )
        with meas.run() as datasaver:
            for index1, i in enumerate(array.set_arrays[0]):
                outer_data[:] = i
                datasaver.add_result(
                    (array.set_arrays[0].array_id, outer_data),
                    (array.set_arrays[1].array_id, array.set_arrays[1][index1, :]),
                    (array.array_id, array[index1, :]),
                )
    elif dims == 1:
        with meas.run() as datasaver:
            for index, i in enumerate(array.set_arrays[0]):
                datasaver.add_result(
                    (array.set_arrays[0].array_id, i), (array.array_id, array[index])
                )
    else:
        raise NotImplementedError(
            "The exporter only currently handles 1 and 2 Dimensional data"
        )
    return datasaver.run_id


def import_dat_file(location: str | Path, exp: Experiment | None = None) -> list[int]:
    """
    This imports a QCoDeS legacy :class:`qcodes.data.data_set.DataSet`
    into the database.

    Args:
        location: Path to file containing legacy dataset
        exp: Specify the experiment to store data to.
            If None the default one is used. See the
            docs of :class:`qcodes.dataset.Measurement` for more details.
    """
    try:
        from qcodes_loop.data.data_set import load_data
    except ImportError as e:
        raise ImportError(
            "The legacy importer requires qcodes_loop to be installed."
        ) from e

    loaded_data = load_data(str(location))
    meas = setup_measurement(loaded_data, exp=exp)
    run_ids = []
    with meas.run() as datasaver:
        datasaver.dataset.add_metadata("snapshot", json.dumps(loaded_data.snapshot()))
        for arrayname, array in loaded_data.arrays.items():
            if not array.is_setpoint:
                run_id = store_array_to_database(datasaver, array)
                run_ids.append(run_id)
    return run_ids
