from __future__ import annotations

import logging
import warnings
from importlib.metadata import version
from math import prod
from typing import TYPE_CHECKING, Literal

from packaging import version as p_version
from typing_extensions import deprecated

from qcodes.dataset.linked_datasets.links import links_to_str
from qcodes.utils import QCoDeSDeprecationWarning

from ..descriptions.versioning import serialization as serial
from .export_to_pandas import (
    _data_to_dataframe,
    _generate_pandas_index,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping
    from pathlib import Path

    import numpy.typing as npt
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


def _load_to_xarray_dataset_dict_no_metadata(
    dataset: DataSetProtocol,
    datadict: Mapping[str, Mapping[str, npt.NDArray]],
    *,
    use_multi_index: Literal["auto", "always", "never"] = "auto",
) -> dict[str, xr.Dataset]:
    if use_multi_index not in ("auto", "always", "never"):
        raise ValueError(
            f"Invalid value for use_multi_index. Expected one of 'auto', 'always', 'never' but got {use_multi_index}"
        )

    xr_dataset_dict: dict[str, xr.Dataset] = {}

    for name, sub_dict in datadict.items():
        shape_is_consistent = (
            dataset.description.shapes is not None
            and name in dataset.description.shapes
            and sub_dict[name].shape == dataset.description.shapes[name]
        )

        if shape_is_consistent and use_multi_index != "always":
            _LOG.info("Exporting %s to xarray using direct method", name)
            xr_dataset_dict[name] = _xarray_data_set_direct(dataset, name, sub_dict)
        else:
            _LOG.info("Exporting %s to xarray via pandas index", name)
            index = _generate_pandas_index(
                sub_dict, dataset.description.interdeps, top_level_param_name=name
            )
            index_is_unique = (
                len(index.unique()) == len(index) if index is not None else False
            )

            if index is None:
                xr_dataset: xr.Dataset = _data_to_dataframe(
                    data=sub_dict,
                    index=index,
                    interdeps=dataset.description.interdeps,
                    dependent_parameter=name,
                ).to_xarray()
                xr_dataset_dict[name] = xr_dataset
            elif index_is_unique:
                df = _data_to_dataframe(
                    sub_dict,
                    index,
                    interdeps=dataset.description.interdeps,
                    dependent_parameter=name,
                )
                xr_dataset_dict[name] = _xarray_data_set_from_pandas_multi_index(
                    dataset, use_multi_index, name, df, index
                )
            else:
                df = _data_to_dataframe(
                    sub_dict,
                    index,
                    interdeps=dataset.description.interdeps,
                    dependent_parameter=name,
                )
                xr_dataset_dict[name] = df.reset_index().to_xarray()

    return xr_dataset_dict


def _xarray_data_set_from_pandas_multi_index(
    dataset: DataSetProtocol,
    use_multi_index: Literal["auto", "always", "never"],
    name: str,
    df: pd.DataFrame,
    index: pd.Index | pd.MultiIndex,
) -> xr.Dataset:
    import pandas as pd
    import xarray as xr

    calc_index = _calculate_index_shape(index)
    index_prod = prod(calc_index.values())
    # if the product of the len of individual index dims == len(total_index)
    # we are on a grid

    on_grid = index_prod == len(index)

    export_with_multi_index = (
        not on_grid and dataset.description.shapes is None and use_multi_index == "auto"
    ) or use_multi_index == "always"

    if export_with_multi_index:
        assert isinstance(df.index, pd.MultiIndex)
        _LOG.info(
            "Exporting %s to xarray using a MultiIndex since on_grid=%s, shape=%s, use_multi_index=%s",
            name,
            on_grid,
            dataset.description.shapes,
            use_multi_index,
        )

        coords = xr.Coordinates.from_pandas_multiindex(df.index, "multi_index")
        xr_dataset = xr.DataArray(df[name], coords=coords).to_dataset(name=name)
    else:
        xr_dataset = df.to_xarray()

    return xr_dataset


def _xarray_data_set_direct(
    dataset: DataSetProtocol, name: str, sub_dict: Mapping[str, npt.NDArray]
) -> xr.Dataset:
    import xarray as xr

    meas_paramspec = dataset.description.interdeps.graph.nodes[name]["value"]
    _, deps, inferred = dataset.description.interdeps.all_parameters_in_tree_by_group(
        meas_paramspec
    )
    # Build coordinate axes from direct dependencies preserving their order
    dep_axis: dict[str, npt.NDArray] = {}
    for axis, dep in enumerate(deps):
        dep_array = sub_dict[dep.name]
        dep_axis[dep.name] = dep_array[
            tuple(slice(None) if i == axis else 0 for i in range(dep_array.ndim))
        ]

    extra_coords: dict[str, tuple[tuple[str, ...], npt.NDArray]] = {}
    extra_data_vars: dict[str, tuple[tuple[str, ...], npt.NDArray]] = {}
    for inf in inferred:
        # skip parameters already used as primary coordinate axes
        if inf.name in dep_axis:
            continue
        # add only if data for this parameter is available
        if inf.name not in sub_dict:
            continue

        inf_related = dataset.description.interdeps.find_all_parameters_in_tree(inf)

        related_deps = inf_related.intersection(set(deps))
        related_top_level = inf_related.intersection({meas_paramspec})

        if len(related_top_level) > 0:
            # If inferred param is related to the top-level measurement parameter,
            # add it as a data variable with the full dependency dimensions
            inf_data_full = sub_dict[inf.name]
            inf_dims_full = tuple(dep_axis.keys())
            extra_data_vars[inf.name] = (inf_dims_full, inf_data_full)
        else:
            # Otherwise, add as a coordinate along the related dependency axes only
            inf_data = sub_dict[inf.name][
                tuple(slice(None) if dep in related_deps else 0 for dep in deps)
            ]
            inf_coords = [dep.name for dep in deps if dep in related_deps]

            extra_coords[inf.name] = (tuple(inf_coords), inf_data)

    # Compose coordinates dict including dependency axes and extra inferred coords
    coords: dict[str, tuple[tuple[str, ...], npt.NDArray] | npt.NDArray]
    coords = {**dep_axis, **extra_coords}

    # Compose data variables dict including measured var and any inferred data vars
    data_vars: dict[str, tuple[tuple[str, ...], npt.NDArray]] = {
        name: (tuple(dep_axis.keys()), sub_dict[name])
    }
    data_vars.update(extra_data_vars)

    ds = xr.Dataset(data_vars, coords=coords)
    return ds


@deprecated(
    "load_to_xarray_dataarray_dict is deprecated, use load_to_xarray_dataarray_dict instead",
    category=QCoDeSDeprecationWarning,
)
def load_to_xarray_dataarray_dict(
    dataset: DataSetProtocol,
    datadict: Mapping[str, Mapping[str, npt.NDArray]],
    *,
    use_multi_index: Literal["auto", "always", "never"] = "auto",
) -> dict[str, xr.DataArray]:
    xr_datasets = _load_to_xarray_dataset_dict_no_metadata(
        dataset, datadict, use_multi_index=use_multi_index
    )
    data_arrays: dict[str, xr.DataArray] = {}

    for dataname, xr_dataset in xr_datasets.items():
        data_array = xr_dataset[dataname]
        _add_param_spec_to_xarray_coords(dataset, data_array)
        paramspec_dict = _paramspec_dict_with_extras(dataset, str(dataname))
        data_array.attrs.update(paramspec_dict.items())
        _add_metadata_to_xarray(dataset, data_array)
        data_arrays[dataname] = data_array

    return data_arrays


def load_to_xarray_dataset_dict(
    dataset: DataSetProtocol,
    datadict: Mapping[str, Mapping[str, npt.NDArray]],
    *,
    use_multi_index: Literal["auto", "always", "never"] = "auto",
) -> dict[str, xr.Dataset]:
    xr_datasets = _load_to_xarray_dataset_dict_no_metadata(
        dataset, datadict, use_multi_index=use_multi_index
    )

    for xr_dataset in xr_datasets.values():
        _add_param_spec_to_xarray_coords(dataset, xr_dataset)
        _add_param_spec_to_xarray_data_vars(dataset, xr_dataset)
        _add_metadata_to_xarray(dataset, xr_dataset)

    return xr_datasets


def _add_metadata_to_xarray(
    dataset: DataSetProtocol, xr_dataset: xr.Dataset | xr.DataArray
) -> None:
    xr_dataset.attrs.update(
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
    # Use -1 as sentinel value for None timestamps since NetCDF doesn't support None
    xr_dataset.attrs["run_timestamp_raw"] = (
        dataset.run_timestamp_raw if dataset.run_timestamp_raw is not None else -1
    )
    xr_dataset.attrs["completed_timestamp_raw"] = (
        dataset.completed_timestamp_raw
        if dataset.completed_timestamp_raw is not None
        else -1
    )
    if len(dataset.metadata) > 0:
        for metadata_tag, metadata in dataset.metadata.items():
            xr_dataset.attrs[metadata_tag] = metadata


def load_to_xarray_dataset(
    dataset: DataSetProtocol,
    data: ParameterData,
    *,
    use_multi_index: Literal["auto", "always", "never"] = "auto",
) -> xr.Dataset:
    import xarray as xr

    xr_dataset_dict = _load_to_xarray_dataset_dict_no_metadata(
        dataset, data, use_multi_index=use_multi_index
    )

    xr_dataset = xr.merge(xr_dataset_dict.values(), compat="equals", join="outer")

    _add_param_spec_to_xarray_coords(dataset, xr_dataset)
    _add_param_spec_to_xarray_data_vars(dataset, xr_dataset)
    _add_metadata_to_xarray(dataset, xr_dataset)

    return xr_dataset


def _add_param_spec_to_xarray_coords(
    dataset: DataSetProtocol, xr_dataset: xr.Dataset | xr.DataArray
) -> None:
    for coord in xr_dataset.coords:
        if coord not in ("index", "multi_index"):
            paramspec_dict = _paramspec_dict_with_extras(dataset, str(coord))
            xr_dataset.coords[str(coord)].attrs.update(paramspec_dict.items())


def _add_param_spec_to_xarray_data_vars(
    dataset: DataSetProtocol, xr_dataset: xr.Dataset
) -> None:
    for data_var in xr_dataset.data_vars:
        paramspec_dict = _paramspec_dict_with_extras(dataset, str(data_var))
        xr_dataset.data_vars[str(data_var)].attrs.update(paramspec_dict.items())


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
    import cf_xarray as cf_xr
    from pandas import MultiIndex

    has_multi_index = any(
        isinstance(xarray_dataset.indexes[index_name], MultiIndex)
        for index_name in xarray_dataset.indexes
    )

    if has_multi_index:
        # as of xarray 2023.8.0 there is no native support
        # for multi index so use cf_xarray for that
        internal_ds = cf_xr.coding.encode_multi_index_as_compress(
            xarray_dataset,
        )
    else:
        internal_ds = xarray_dataset

    data_var_kinds = [
        internal_ds.data_vars[data_var].dtype.kind for data_var in internal_ds.data_vars
    ]
    coord_kinds = [internal_ds.coords[coord].dtype.kind for coord in internal_ds.coords]
    dataset_has_complex_vals = "c" in data_var_kinds or "c" in coord_kinds
    # these are the versions of xarray / h5netcdf respectively required to support complex
    # values without fallback to invalid features. Once these are the min versions supported
    # we can drop the fallback code here including the warning suppression.
    xarray_too_old = p_version.Version(version("xarray")) < p_version.Version(
        "2024.10.0"
    )
    h5netcdf_too_old = p_version.Version(version("h5netcdf")) < p_version.Version(
        "1.4.0"
    )

    allow_invalid_netcdf = dataset_has_complex_vals and (
        xarray_too_old or h5netcdf_too_old
    )

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
                    extra={"file_name": str(file_path)},
                )
                maybe_write_job.compute()
