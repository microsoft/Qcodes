from __future__ import annotations

import logging
import warnings
from math import prod
from typing import TYPE_CHECKING, Literal, cast

from qcodes.dataset.linked_datasets.links import links_to_str

from ..descriptions.versioning import serialization as serial
from .export_to_pandas import (
    _data_to_dataframe,
    _generate_pandas_index,
    _same_setpoints,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import xarray as xr

    from qcodes.dataset.data_set_protocol import DataSetProtocol, ParameterData

_LOG = logging.getLogger(__name__)


def _calculate_index_shape(idx: pd.Index | pd.MultiIndex) -> dict[Hashable, int]:
    # heavily inspired by xarray.core.dataset.from_dataframe
    import pandas as pd
    from xarray.core.indexes import PandasIndex, remove_unused_levels_categories
    from xarray.core.variable import Variable, calculate_dimensions

    idx = remove_unused_levels_categories(idx)

    if isinstance(idx, pd.MultiIndex) and not idx.is_unique:
        raise ValueError(
            "cannot convert a DataFrame with a non-unique MultiIndex into xarray"
        )
    index_vars: dict[Hashable, Variable] = {}

    if isinstance(idx, pd.MultiIndex):
        dims = tuple(
            name if name is not None else f"level_{n}"
            for n, name in enumerate(idx.names)
        )
        for dim, lev in zip(dims, idx.levels):
            xr_idx = PandasIndex(lev, dim)
            index_vars.update(xr_idx.create_variables())
    else:
        index_name = idx.name if idx.name is not None else "index"
        dims = (index_name,)
        xr_idx = PandasIndex(idx, index_name)
        index_vars.update(xr_idx.create_variables())

    expanded_shape = calculate_dimensions(index_vars)
    return expanded_shape


def _load_to_xarray_dataarray_dict_no_metadata(
    dataset: DataSetProtocol,
    datadict: Mapping[str, Mapping[str, np.ndarray]],
    *,
    use_multi_index: Literal["auto", "always", "never"] = "auto",
) -> dict[str, xr.DataArray]:
    import pandas as pd
    import xarray as xr

    if use_multi_index not in ("auto", "always", "never"):
        raise ValueError(
            f"Invalid value for use_multi_index. Expected one of 'auto', 'always', 'never' but got {use_multi_index}"
        )

    data_xrdarray_dict: dict[str, xr.DataArray] = {}

    for name, subdict in datadict.items():
        index = _generate_pandas_index(subdict)

        if index is None:
            xrdarray: xr.DataArray = (
                _data_to_dataframe(subdict, index=index)
                .to_xarray()
                .get(name, xr.DataArray())
            )
            data_xrdarray_dict[name] = xrdarray
        else:
            index_unique = len(index.unique()) == len(index)

            df = _data_to_dataframe(subdict, index)

            if not index_unique:
                # index is not unique so we fallback to using a counter as index
                # and store the index as a variable
                xrdata_temp = df.reset_index().to_xarray()
                for _name in subdict:
                    data_xrdarray_dict[_name] = xrdata_temp[_name]
            else:
                calc_index = _calculate_index_shape(index)
                index_prod = prod(calc_index.values())
                # if the product of the len of individual index dims == len(total_index)
                # we are on a grid

                on_grid = index_prod == len(index)

                export_with_multi_index = (
                    not on_grid
                    and dataset.description.shapes is None
                    and use_multi_index == "auto"
                ) or use_multi_index == "always"

                if export_with_multi_index:
                    assert isinstance(df.index, pd.MultiIndex)

                    if hasattr(xr, "Coordinates"):
                        coords = xr.Coordinates.from_pandas_multiindex(
                            df.index, "multi_index"
                        )
                        xrdarray = xr.DataArray(df[name], coords=coords)
                    else:
                        # support xarray < 2023.8.0, can be removed when we drop support for that
                        xrdarray = xr.DataArray(df[name], [("multi_index", df.index)])
                else:
                    xrdarray = df.to_xarray().get(name, xr.DataArray())

                data_xrdarray_dict[name] = xrdarray

    return data_xrdarray_dict


def load_to_xarray_dataarray_dict(
    dataset: DataSetProtocol,
    datadict: Mapping[str, Mapping[str, np.ndarray]],
    *,
    use_multi_index: Literal["auto", "always", "never"] = "auto",
) -> dict[str, xr.DataArray]:
    dataarrays = _load_to_xarray_dataarray_dict_no_metadata(
        dataset, datadict, use_multi_index=use_multi_index
    )

    for dataname, dataarray in dataarrays.items():
        _add_param_spec_to_xarray_coords(dataset, dataarray)
        paramspec_dict = _paramspec_dict_with_extras(dataset, str(dataname))
        dataarray.attrs.update(paramspec_dict.items())
        _add_metadata_to_xarray(dataset, dataarray)

    return dataarrays


def _add_metadata_to_xarray(
    dataset: DataSetProtocol, xrdataset: xr.Dataset | xr.DataArray
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
        xrdataset.attrs["completed_timestamp_raw"] = dataset.completed_timestamp_raw
    if len(dataset.metadata) > 0:
        for metadata_tag, metadata in dataset.metadata.items():
            xrdataset.attrs[metadata_tag] = metadata


def load_to_xarray_dataset(
    dataset: DataSetProtocol,
    data: ParameterData,
    *,
    use_multi_index: Literal["auto", "always", "never"] = "auto",
) -> xr.Dataset:
    import xarray as xr

    if not _same_setpoints(data):
        warnings.warn(
            "Independent parameter setpoints are not equal. "
            "Check concatenated output carefully. Please "
            "consider using `to_xarray_dataarray_dict` to export each "
            "independent parameter to its own datarray."
        )

    data_xrdarray_dict = _load_to_xarray_dataarray_dict_no_metadata(
        dataset, data, use_multi_index=use_multi_index
    )

    # Casting Hashable for the key type until python/mypy#1114
    # and python/typing#445 are resolved.
    xrdataset = xr.Dataset(cast("dict[Hashable, xr.DataArray]", data_xrdarray_dict))

    _add_param_spec_to_xarray_coords(dataset, xrdataset)
    _add_param_spec_to_xarray_data_vars(dataset, xrdataset)
    _add_metadata_to_xarray(dataset, xrdataset)

    return xrdataset


def _add_param_spec_to_xarray_coords(
    dataset: DataSetProtocol, xrdataset: xr.Dataset | xr.DataArray
) -> None:
    for coord in xrdataset.coords:
        if coord not in ("index", "multi_index"):
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
) -> dict[str, object]:
    paramspec_dict = dict(dataset.paramspecs[str(dim_name)]._to_dict())
    # units and long_name have special meaning in xarray that closely
    # matches how qcodes uses unit and label so we copy these attributes
    # https://xarray.pydata.org/en/stable/getting-started-guide/quick-overview.html#attributes
    paramspec_dict["units"] = paramspec_dict.get("unit", "")
    paramspec_dict["long_name"] = paramspec_dict.get("label", "")
    return paramspec_dict


def xarray_to_h5netcdf_with_complex_numbers(
    xarray_dataset: xr.Dataset, file_path: str | Path, compute: bool = True
) -> None:
    import cf_xarray as cfxr
    from pandas import MultiIndex

    has_multi_index = any(
        isinstance(xarray_dataset.indexes[index_name], MultiIndex)
        for index_name in xarray_dataset.indexes
    )

    if has_multi_index:
        # as of xarray 2023.8.0 there is no native support
        # for multi index so use cf_xarray for that
        internal_ds = cfxr.coding.encode_multi_index_as_compress(
            xarray_dataset,
        )
    else:
        internal_ds = xarray_dataset

    data_var_kinds = [
        internal_ds.data_vars[data_var].dtype.kind for data_var in internal_ds.data_vars
    ]
    coord_kinds = [internal_ds.coords[coord].dtype.kind for coord in internal_ds.coords]
    allow_invalid_netcdf = "c" in data_var_kinds or "c" in coord_kinds

    with warnings.catch_warnings():
        # see http://xarray.pydata.org/en/stable/howdoi.html
        # for how to export complex numbers
        if allow_invalid_netcdf:
            warnings.filterwarnings(
                "ignore",
                module="h5netcdf",
                message="You are writing invalid netcdf features",
                category=UserWarning,
            )
        maybe_write_job = internal_ds.to_netcdf(
            path=file_path,
            engine="h5netcdf",
            invalid_netcdf=allow_invalid_netcdf,
            compute=compute,
        )
        if not compute and maybe_write_job is not None:
            # Dask and therefor tqdm.dask is slow to
            # import and only used here so defer the import
            # to when required.
            from tqdm.dask import TqdmCallback

            with TqdmCallback(desc="Combining files"):
                _LOG.info(
                    "Writing netcdf file using Dask delayed writer.",
                    extra={"file_name": file_path},
                )
                maybe_write_job.compute()
