from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Dict, Hashable, Mapping, Union, cast

import numpy as np

from qcodes.dataset.linked_datasets.links import links_to_str

from ..descriptions.versioning import serialization as serial
from .export_to_pandas import (
    _data_to_dataframe,
    _generate_pandas_index,
    _same_setpoints,
)

if TYPE_CHECKING:
    import xarray as xr

    from qcodes.dataset.data_set import ParameterData
    from qcodes.dataset.data_set_protocol import DataSetProtocol


def _load_to_xarray_dataarray_dict_no_metadata(
    dataset: DataSetProtocol, datadict: Mapping[str, Mapping[str, np.ndarray]]
) -> Dict[str, xr.DataArray]:
    import xarray as xr

    data_xrdarray_dict: Dict[str, xr.DataArray] = {}

    for name, subdict in datadict.items():
        index = _generate_pandas_index(subdict)
        if index is not None and len(index.unique()) != len(index):
            for _name in subdict:
                data_xrdarray_dict[_name] = _data_to_dataframe(
                    subdict, index).reset_index().to_xarray()[_name]
        else:
            xrdarray: xr.DataArray = (
                _data_to_dataframe(subdict, index).to_xarray().get(name, xr.DataArray())
            )
            data_xrdarray_dict[name] = xrdarray

    return data_xrdarray_dict


def load_to_xarray_dataarray_dict(
    dataset: DataSetProtocol, datadict: Mapping[str, Mapping[str, np.ndarray]]
) -> Dict[str, xr.DataArray]:
    dataarrays = _load_to_xarray_dataarray_dict_no_metadata(dataset, datadict)

    for dataname, dataarray in dataarrays.items():
        _add_param_spec_to_xarray_coords(dataset, dataarray)
        paramspec_dict = _paramspec_dict_with_extras(dataset, str(dataname))
        dataarray.attrs.update(paramspec_dict.items())
        _add_metadata_to_xarray(dataset, dataarray)

    return dataarrays


def _add_metadata_to_xarray(
    dataset: DataSetProtocol, xrdataset: Union[xr.Dataset, xr.DataArray]
) -> None:
    xrdataset.attrs.update(
        {
            "ds_name": dataset.name,
            "sample_name": dataset.sample_name,
            "exp_name": dataset.exp_name,
            "snapshot": dataset._snapshot_raw or "null",
            "guid": dataset.guid,
            "run_timestamp": dataset.run_timestamp() or "",
            "completed_timestamp": dataset.completed_timestamp() or "",
            "captured_run_id": dataset.captured_run_id,
            "captured_counter": dataset.captured_counter,
            "run_id": dataset.run_id,
            "run_description": serial.to_json_for_storage(dataset.description),
            "parent_dataset_links": links_to_str(dataset.parent_dataset_links),
        }
    )
    if dataset.run_timestamp_raw is not None:
        xrdataset.attrs["run_timestamp_raw"] = dataset.run_timestamp_raw
    if dataset.completed_timestamp_raw is not None:
        xrdataset.attrs[
            "completed_timestamp_raw"] = dataset.completed_timestamp_raw
    if len(dataset.metadata) > 0:
        for metadata_tag, metadata in dataset.metadata.items():
            xrdataset.attrs[metadata_tag] = metadata


def load_to_xarray_dataset(dataset: DataSetProtocol, data: ParameterData) -> xr.Dataset:
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

    _add_param_spec_to_xarray_coords(dataset, xrdataset)
    _add_param_spec_to_xarray_data_vars(dataset, xrdataset)
    _add_metadata_to_xarray(dataset, xrdataset)

    return xrdataset


def _add_param_spec_to_xarray_coords(
    dataset: DataSetProtocol, xrdataset: Union[xr.Dataset, xr.DataArray]
) -> None:
    for coord in xrdataset.coords:
        if coord != "index":
            paramspec_dict = _paramspec_dict_with_extras(dataset, str(coord))
            xrdataset.coords[str(coord)].attrs.update(paramspec_dict.items())


def _add_param_spec_to_xarray_data_vars(
    dataset: DataSetProtocol, xrdataset: xr.Dataset
) -> None:
    for data_var in xrdataset.data_vars:
        paramspec_dict = _paramspec_dict_with_extras(dataset, str(data_var))
        xrdataset.data_vars[str(data_var)].attrs.update(paramspec_dict.items())


def _paramspec_dict_with_extras(
    dataset: DataSetProtocol, dim_name: str
) -> Dict[str, object]:
    paramspec_dict = dict(dataset.paramspecs[str(dim_name)]._to_dict())
    # units and long_name have special meaning in xarray that closely
    # matches how qcodes uses unit and label so we copy these attributes
    # https://xarray.pydata.org/en/stable/getting-started-guide/quick-overview.html#attributes
    paramspec_dict["units"] = paramspec_dict.get("unit", "")
    paramspec_dict["long_name"] = paramspec_dict.get("label", "")
    return paramspec_dict
