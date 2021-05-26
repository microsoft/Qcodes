from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Dict, Hashable, Mapping, Union, cast

import numpy as np

from ..descriptions.versioning import serialization as serial
from .export_to_pandas import (
    _data_to_dataframe,
    _generate_pandas_index,
    _same_setpoints,
)

if TYPE_CHECKING:
    import xarray as xr

    from qcodes.dataset.data_set import DataSet, ParameterData


def _load_to_xarray_dataarray_dict_no_metadata(
    dataset: DataSet, datadict: Mapping[str, Mapping[str, np.ndarray]]
) -> Dict[str, xr.DataArray]:
    import xarray as xr

    data_xrdarray_dict: Dict[str, xr.DataArray] = {}

    for name, subdict in datadict.items():
        index = _generate_pandas_index(subdict)
        if index is not None and len(index.unique()) != len(index):
            for _name in subdict:
                data_xrdarray_dict[_name] = _data_to_dataframe(
                    subdict, index).reset_index().to_xarray()[_name]
                paramspec_dict = dataset.paramspecs[_name]._to_dict()
                data_xrdarray_dict[_name].attrs.update(paramspec_dict.items())
        else:
            xrdarray: xr.DataArray = _data_to_dataframe(
                subdict, index).to_xarray()[name]
            data_xrdarray_dict[name] = xrdarray
            paramspec_dict = dataset.paramspecs[name]._to_dict()
            xrdarray.attrs.update(paramspec_dict.items())

    return data_xrdarray_dict


def load_to_xarray_dataarray_dict(
    dataset: DataSet, datadict: Mapping[str, Mapping[str, np.ndarray]]
) -> Dict[str, xr.DataArray]:
    dataarrays = _load_to_xarray_dataarray_dict_no_metadata(dataset, datadict)

    for dataarray in dataarrays.values():
        _add_metadata_to_xarray(dataset, dataarray)
    return dataarrays


def _add_metadata_to_xarray(
        dataset: DataSet,
        xrdataset: Union[xr.Dataset, xr.DataArray]
) -> None:
    xrdataset.attrs.update({
        "ds_name": dataset.name,
        "sample_name": dataset.sample_name,
        "exp_name": dataset.exp_name,
        "snapshot": dataset.snapshot_raw or "null",
        "guid": dataset.guid,
        "run_timestamp": dataset.run_timestamp() or "",
        "completed_timestamp": dataset.completed_timestamp() or "",
        "captured_run_id": dataset.captured_run_id,
        "captured_counter": dataset.captured_counter,
        "run_id": dataset.run_id,
        "run_description": serial.to_json_for_storage(dataset.description)
    })
    if dataset.run_timestamp_raw is not None:
        xrdataset.attrs["run_timestamp_raw"] = dataset.run_timestamp_raw
    if dataset.completed_timestamp_raw is not None:
        xrdataset.attrs[
            "completed_timestamp_raw"] = dataset.completed_timestamp_raw
    if len(dataset.metadata) > 0:
        for metadata_tag, metadata in dataset.metadata.items():
            xrdataset.attrs[metadata_tag] = metadata


def load_to_xarray_dataset(dataset: DataSet, data: ParameterData) -> xr.Dataset:
    import xarray as xr

    if not _same_setpoints(data):
        warnings.warn(
            "Independent parameter setpoints are not equal. "
            "Check concatenated output carefully. Please "
            "consider using `to_xarray_dataarray_dict` to export each "
            "independent parameter to its own datarray."
        )

    data_xrdarray_dict = _load_to_xarray_dataarray_dict_no_metadata(dataset, data)

    # Casting Hashable for the key type until python/mypy#1114
    # and python/typing#445 are resolved.
    xrdataset = xr.Dataset(
        cast(Dict[Hashable, xr.DataArray], data_xrdarray_dict))

    for dim in xrdataset.dims:
        if "index" != dim:
            paramspec_dict = dataset.paramspecs[str(dim)]._to_dict()
            xrdataset.coords[str(dim)].attrs.update(paramspec_dict.items())

    _add_metadata_to_xarray(dataset, xrdataset)

    return xrdataset
