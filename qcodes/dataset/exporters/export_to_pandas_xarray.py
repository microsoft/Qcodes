from __future__ import annotations

import warnings

from typing import Dict, TYPE_CHECKING, Union, Iterator, cast, Hashable

import numpy as np

from ..descriptions.versioning import serialization as serial

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr
    from qcodes.dataset.data_set import DataSet, ParameterData


def _load_to_xarray_dataarray_dict_no_metadata(
        dataset: DataSet,
        datadict: Dict[str, Dict[str, np.ndarray]]
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
        dataset: DataSet,
        datadict: Dict[str, Dict[str, np.ndarray]]
) -> Dict[str, xr.DataArray]:
    dataarrays = _load_to_xarray_dataarray_dict_no_metadata(dataset, datadict)

    for dataarray in dataarrays.values():
        _add_metadata_to_xarray(dataset, dataarray)
    return dataarrays


def _data_to_dataframe(data: Dict[str, np.ndarray], index: Union[pd.Index, pd.MultiIndex]) -> pd.DataFrame:
    import pandas as pd
    if len(data) == 0:
        return pd.DataFrame()
    dependent_col_name = list(data.keys())[0]
    dependent_data = data[dependent_col_name]
    if dependent_data.dtype == np.dtype('O'):
        # ravel will not fully unpack a numpy array of arrays
        # which are of "object" dtype. This can happen if a variable
        # length array is stored in the db. We use concatenate to
        # flatten these
        mydata = np.concatenate(dependent_data)
    else:
        mydata = dependent_data.ravel()
    df = pd.DataFrame(mydata, index=index,
                      columns=[dependent_col_name])
    return df


def _generate_pandas_index(data: Dict[str, np.ndarray]) -> Union[pd.Index, pd.MultiIndex]:
    # the first element in the dict given by parameter_tree is always the dependent
    # parameter and the index is therefore formed from the rest
    import pandas as pd
    keys = list(data.keys())
    if len(data) <= 1:
        index = None
    elif len(data) == 2:
        index = pd.Index(data[keys[1]].ravel(), name=keys[1])
    else:
        index_data = tuple(np.concatenate(data[key])
                           if data[key].dtype == np.dtype('O')
                           else data[key].ravel()
                           for key in keys[1:])
        index = pd.MultiIndex.from_arrays(
            index_data,
            names=keys[1:])
    return index


def load_to_dataframe_dict(datadict: ParameterData) -> Dict[str, pd.DataFrame]:
    dfs = {}
    for name, subdict in datadict.items():
        index = _generate_pandas_index(subdict)
        dfs[name] = _data_to_dataframe(subdict, index)
    return dfs


def _parameter_data_identical(param_dict_a: Dict[str, np.ndarray],
                              param_dict_b: Dict[str, np.ndarray]) -> bool:

    try:
        np.testing.assert_equal(param_dict_a, param_dict_b)
    except AssertionError:
        return False

    return True


def _same_setpoints(datadict: ParameterData) -> bool:

    def _get_setpoints(dd: ParameterData) -> Iterator[Dict[str, np.ndarray]]:

        for dep_name, param_dict in dd.items():
            out = {
                name: vals for name, vals in param_dict.items() if name != dep_name
            }
            yield out

    sp_iterator = _get_setpoints(datadict)

    try:
        first = next(sp_iterator)
    except StopIteration:
        return True

    return all(_parameter_data_identical(first, rest) for rest in sp_iterator)


def load_to_concatenated_dataframe(
        datadict: ParameterData
) -> "pd.DataFrame":
    import pandas as pd

    if not _same_setpoints(datadict):
        warnings.warn(
            "Independent parameter setpoints are not equal. "
            "Check concatenated output carefully. Please "
            "consider using `to_pandas_dataframe_dict` to export each "
            "independent parameter to its own dataframe."
        )

    dfs_dict = load_to_dataframe_dict(datadict)
    df = pd.concat(list(dfs_dict.values()), axis=1)

    return df


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
    if len(dataset._metadata) > 0:
        xrdataset.attrs['extra_metadata'] = {}

        for metadata_tag, metadata in dataset._metadata.items():
            xrdataset.attrs['extra_metadata'][metadata_tag] = metadata


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

    return xrdataset
