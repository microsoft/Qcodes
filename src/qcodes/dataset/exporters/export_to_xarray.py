from __future__ import annotations

import itertools
import logging
import os
import shutil
import tempfile
import uuid
import warnings
from importlib.metadata import version
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from packaging import version as p_version

import qcodes
from qcodes.dataset.linked_datasets.links import links_to_str

from ..descriptions.versioning import serialization as serial
from .export_to_pandas import (
    _data_to_dataframe,
    _generate_pandas_index,
)
from .netcdf_snapshot import (
    _SNAPSHOT_POINTER,
    SNAPSHOT_GROUP,
    SNAPSHOT_LOCATION,
    _should_write_snapshot_to_group,
    _snapshot_pointer_location,
    _write_snapshot_to_group,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    import numpy.typing as npt
    import pandas as pd
    import xarray as xr

    from qcodes.dataset.data_set_protocol import DataSetProtocol, ParameterData

_LOGGER = logging.getLogger(__name__)

# Target size in bytes of a single HDF5 chunk when compression is enabled.
# The chunk size matters a lot for how well the data compresses since each chunk
# is compressed independently. The auto chunking performed by h5py picks chunks
# that are significantly smaller than this which results in noticeably worse
# compression ratios for typical QCoDeS data.
_TARGET_CHUNK_SIZE_BYTES = 4 * 1024**2


def _variable_chunksizes(
    variable: xr.DataArray, target_bytes: int = _TARGET_CHUNK_SIZE_BYTES
) -> tuple[int, ...] | None:
    """
    Calculate the HDF5 chunk shape to use for a given variable.

    Chunks are approximately ``target_bytes`` large and never larger, unless a
    single element is. Dimensions are chunked from the leading dimension
    onwards: leading dimensions get a chunk size of 1 for as long as the
    remaining trailing dimensions are larger than the target. The first
    dimension for which the trailing dimensions fit is sized to fill the
    target and all dimensions after it are kept whole. This keeps chunks well
    below the 4 GiB HDF5 chunk size limit even for very large rows.

    Returns None for scalars and for variables with a zero sized dimension
    since those cannot be chunked.
    """
    shape = variable.shape
    if len(shape) == 0 or 0 in shape:
        return None

    itemsize = max(variable.dtype.itemsize, 1)
    chunks: list[int] = []
    for i, size in enumerate(shape):
        trailing_bytes = prod(shape[i + 1 :]) * itemsize
        if trailing_bytes > target_bytes:
            chunks.append(1)
            continue
        chunks.append(max(1, min(size, target_bytes // trailing_bytes)))
        chunks.extend(shape[i + 1 :])
        break
    return tuple(chunks)


def _netcdf_compression_encoding(
    dataset: xr.Dataset, complevel: int
) -> dict[str, dict[str, Any]]:
    """
    Build a per variable netcdf encoding dict enabling deflate compression.

    The shuffle filter is only enabled for non complex data. QCoDeS stores
    complex numbers as an HDF5 compound type of two floats and byte shuffling
    such a compound type destroys the byte patterns that deflate relies on,
    roughly halving the compression ratio.

    Args:
        dataset: The dataset that is about to be written.
        complevel: Deflate compression level between 1 and 9.

    Returns:
        A mapping from variable name to netcdf encoding options. Variables that
        cannot be compressed, such as variable length strings, are omitted.

    """
    encoding: dict[str, dict[str, Any]] = {}

    for name in itertools.chain(dataset.data_vars, dataset.coords):
        variable = dataset[name]
        # Compression filters can only be applied to numeric and boolean data.
        # Variable length strings and object arrays are written without filters.
        if variable.dtype.kind not in "buifc":
            continue

        variable_encoding: dict[str, Any] = {
            "zlib": True,
            "complevel": complevel,
            "shuffle": variable.dtype.kind != "c",
        }
        chunksizes = _variable_chunksizes(variable)
        if chunksizes is not None:
            variable_encoding["chunksizes"] = chunksizes
        encoding[str(name)] = variable_encoding

    return encoding


def _rechunk_to_match_encoding(
    dataset: xr.Dataset, encoding: Mapping[str, Mapping[str, Any]]
) -> xr.Dataset:
    """
    Align the dask chunks of a dataset with the HDF5 chunks it will be written to.

    Writing a dask backed dataset is only performed one dask block at a time. If
    a block covers less than a full HDF5 chunk, that chunk has to be read,
    decompressed, updated and recompressed again for every block, which is very
    slow.

    Each variable is rechunked using its own HDF5 chunk layout. Variables that
    share a dimension may have different chunk sizes along it, for example when
    the dimensions are ordered differently, so a single dataset wide chunk size
    per dimension would leave dask blocks much smaller than the HDF5 chunks.
    Variables that are not dask backed are left unmodified.
    """
    rechunked: dict[str, xr.Variable] = {}
    for name, variable_encoding in encoding.items():
        chunksizes = variable_encoding.get("chunksizes")
        variable = dataset.variables[name]
        if chunksizes is None or variable.chunks is None:
            continue
        rechunked[name] = variable.chunk(
            dict(zip(variable.dims, chunksizes, strict=True))
        )

    if not rechunked:
        return dataset
    dataset = dataset.copy(deep=False)
    for name, variable in rechunked.items():
        dataset.variables[name].data = variable.data
    return dataset


# Encoding keys that control compression or chunking of a netcdf variable.
# original_shape is included since xarray discards the requested chunksizes when
# it differs from the shape of the variable, which happens for a dataset combined
# from multiple files with open_mfdataset as in the chunked export.
_COMPRESSION_ENCODING_KEYS = frozenset(
    {
        "blosc",
        "bzip2",
        "chunksizes",
        "complevel",
        "compression",
        "compression_opts",
        "contiguous",
        "original_shape",
        "preferred_chunks",
        "shuffle",
        "szip",
        "zlib",
        "zstd",
    }
)


def _without_inherited_compression(dataset: xr.Dataset) -> xr.Dataset:
    """
    Return a shallow copy of a dataset without compression in its encoding.

    A dataset loaded from a compressed netcdf file keeps the compression settings
    in the encoding of each variable and xarray reuses them when writing the
    dataset again. Removing them ensures that only the compression requested for
    this export is applied, and in particular that writing without compression
    really produces an uncompressed file. Other encoding settings such as the
    dtype and fill value are preserved and the input dataset is not modified.
    """
    dataset = dataset.copy(deep=False)
    for variable in dataset.variables.values():
        variable.encoding = {
            key: value
            for key, value in variable.encoding.items()
            if key not in _COMPRESSION_ENCODING_KEYS
        }
    return dataset


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


def _add_inferred_data_vars(
    dataset: DataSetProtocol,
    name: str,
    sub_dict: Mapping[str, npt.NDArray],
    xr_dataset: xr.Dataset,
    index: pd.Index | pd.MultiIndex | None,
) -> xr.Dataset:
    """Add inferred parameters as data variables to an xarray dataset.

    Parameters that are inferred from the top-level measurement parameter
    and present in sub_dict but not yet in the dataset are added as data
    variables along the existing dimensions.
    """
    from pandas import Series

    interdeps = dataset.description.interdeps
    meas_paramspec = interdeps.graph.nodes[name]["value"]
    _, deps, inferred = interdeps.all_parameters_in_tree_by_group(meas_paramspec)

    dep_names = {dep.name for dep in deps}
    dims = tuple(d for d in xr_dataset.dims)

    # ``Series.to_xarray()`` expands the index into one dimension per index
    # level. That is only compatible with the target dataset if the dataset
    # dimensions are exactly the levels of the index. It is not the case when
    # the dataset uses a single ``multi_index`` dimension or the flat index
    # created by ``DataFrame.reset_index()``. A non unique MultiIndex cannot be
    # converted at all. In those cases the data is already in index order so it
    # can be reshaped directly.
    index_is_expandable = (
        index is not None and index.is_unique and set(dims) == set(index.names)
    )

    for inf in inferred:
        if inf.name in dep_names:
            continue
        if inf.name in xr_dataset:
            continue
        if inf.name not in sub_dict:
            continue

        inf_data = sub_dict[inf.name]
        if inf_data.dtype == np.dtype("O"):
            try:
                flat = np.concatenate(inf_data)
            except ValueError:
                flat = inf_data.ravel()
        else:
            flat = inf_data.ravel()

        # Only add if the flattened data can be reshaped to the dataset
        # dimensions. This is more robust than checking individual parent
        # sizes because an inferred parameter may have multiple parents
        # with different sizes.
        expected_shape = tuple(xr_dataset.sizes[d] for d in dims)
        expected_size = prod(expected_shape)
        if flat.shape[0] == expected_size:
            if index is not None and index_is_expandable:
                # If an index is provided, we should align the inferred data with the index.
                # This is necessary because data may be reordered when transforming from a pandas DataFrame to an xarray Dataset.
                # Passing an index allows the original data ordering to be preserved on reconstruction.
                indexed_data = Series(flat, index=index, name=inf.name).to_xarray()
                xr_dataset[inf.name] = indexed_data
            else:
                xr_dataset[inf.name] = (dims, flat.reshape(expected_shape))
        else:
            _LOGGER.warning(
                "Cannot add inferred parameter '%s' to xarray dataset for '%s' "
                "(run_id=%s): data size %d does not match the dataset "
                "dimensions %s (size %d). This is likely a user error in the "
                "measurement setup.",
                inf.name,
                name,
                dataset.run_id,
                flat.shape[0],
                dict(zip(dims, expected_shape)),
                expected_size,
            )

    return xr_dataset


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
            _LOGGER.info("Exporting %s to xarray using direct method", name)
            xr_dataset_dict[name] = _xarray_data_set_direct(dataset, name, sub_dict)
        else:
            _LOGGER.info("Exporting %s to xarray via pandas index", name)
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
                xr_dataset_dict[name] = _add_inferred_data_vars(
                    dataset, name, sub_dict, xr_dataset, index
                )
            elif index_is_unique:
                df = _data_to_dataframe(
                    sub_dict,
                    index,
                    interdeps=dataset.description.interdeps,
                    dependent_parameter=name,
                )
                xr_dataset = _xarray_data_set_from_pandas_multi_index(
                    dataset, use_multi_index, name, df, index
                )
                xr_dataset_dict[name] = _add_inferred_data_vars(
                    dataset, name, sub_dict, xr_dataset, index
                )
            else:
                df = _data_to_dataframe(
                    sub_dict,
                    index,
                    interdeps=dataset.description.interdeps,
                    dependent_parameter=name,
                )
                xr_dataset = df.reset_index().to_xarray()
                xr_dataset_dict[name] = _add_inferred_data_vars(
                    dataset, name, sub_dict, xr_dataset, index
                )

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
        _LOGGER.info(
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
    dataset: DataSetProtocol,
    name: str,
    sub_dict: Mapping[str, npt.NDArray],
) -> xr.Dataset:
    import xarray as xr

    meas_paramspec = dataset.description.interdeps.graph.nodes[name]["value"]
    _, deps, inferred = dataset.description.interdeps.all_parameters_in_tree_by_group(
        meas_paramspec
    )

    shape = sub_dict[name].shape
    expected_size = prod(shape)

    if not deps:
        dimensions = tuple(f"{name}_dim_{axis}" for axis in range(len(shape)))

        def reshape_without_dependencies(data: npt.NDArray) -> npt.NDArray:
            if data.size != expected_size:
                raise ValueError(
                    f"Parameter contains {data.size} values, "
                    f"but {expected_size} were expected"
                )
            return data.reshape(shape)

        independent_data_vars: dict[str, tuple[tuple[str, ...], npt.NDArray]] = {
            name: (dimensions, reshape_without_dependencies(sub_dict[name]))
        }
        for inf in inferred:
            if inf.name in sub_dict:
                independent_data_vars[inf.name] = (
                    dimensions,
                    reshape_without_dependencies(sub_dict[inf.name]),
                )

        return xr.Dataset(independent_data_vars)

    if len(deps) != len(shape):
        raise ValueError(
            f"Parameter {name!r} has shape {shape}, but has {len(deps)} dependencies"
        )

    dep_axis: dict[str, npt.NDArray] = {}
    destination = np.zeros(expected_size, dtype=np.intp)

    for dimension, dep in enumerate(deps):
        dep_data = sub_dict[dep.name].ravel()

        if dep_data.size != expected_size:
            raise ValueError(
                f"Dependency {dep.name!r} contains {dep_data.size} values, "
                f"but {expected_size} were expected"
            )

        values, first_indices, sorted_codes = np.unique(
            dep_data,
            return_index=True,
            return_inverse=True,
            equal_nan=True,
        )

        if values.size != shape[dimension]:
            raise ValueError(
                f"Dependency {dep.name!r} does not define an axis of length "
                f"{shape[dimension]}: found {values.size} unique values"
            )

        # Restore first-seen order because np.unique sorts its output.
        first_seen_order = np.argsort(first_indices)
        dep_axis[dep.name] = values[first_seen_order]

        code_remapping = np.empty(values.size, dtype=np.intp)
        code_remapping[first_seen_order] = np.arange(values.size)
        codes = code_remapping[sorted_codes]

        # Encode the multidimensional coordinate using C-order mixed radix.
        destination = destination * shape[dimension] + codes

    if not np.array_equal(
        np.sort(destination),
        np.arange(expected_size, dtype=np.intp),
    ):
        raise ValueError(
            f"Dependencies for {name!r} do not form a complete Cartesian "
            "grid without duplicate points"
        )

    permutation = np.argsort(destination)

    def reorder(data: npt.NDArray) -> npt.NDArray:
        if data.size != expected_size:
            raise ValueError(
                f"Parameter contains {data.size} values, "
                f"but {expected_size} were expected"
            )
        return data.ravel()[permutation].reshape(shape)

    reordered_data = {
        parameter_name: reorder(parameter_data)
        for parameter_name, parameter_data in sub_dict.items()
    }

    extra_coords: dict[str, tuple[tuple[str, ...], npt.NDArray]] = {}
    extra_data_vars: dict[str, tuple[tuple[str, ...], npt.NDArray]] = {}

    for inf in inferred:
        if inf.name not in reordered_data:
            continue

        inf_related = dataset.description.interdeps.find_all_parameters_in_tree(inf)
        related_deps = inf_related.intersection(set(deps))
        related_top_level = inf_related.intersection({meas_paramspec})

        if related_top_level:
            extra_data_vars[inf.name] = (
                tuple(dep_axis),
                reordered_data[inf.name],
            )
        else:
            inf_data = reordered_data[inf.name][
                tuple(slice(None) if dep in related_deps else 0 for dep in deps)
            ]
            inf_dims = tuple(dep.name for dep in deps if dep in related_deps)
            extra_coords[inf.name] = (inf_dims, inf_data)

    coords: dict[
        str,
        tuple[tuple[str, ...], npt.NDArray] | npt.NDArray,
    ] = {**dep_axis, **extra_coords}

    data_vars: dict[str, tuple[tuple[str, ...], npt.NDArray]] = {
        name: (tuple(dep_axis), reordered_data[name]),
        **extra_data_vars,
    }

    return xr.Dataset(data_vars, coords=coords)


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

    # When shapes are inconsistent (e.g. incomplete measurements), different
    # code paths may represent the same setpoint parameter as a coordinate in
    # some sub-datasets and a data variable in others.  xr.merge refuses to
    # combine them when the role is ambiguous, so we normalise first by
    # promoting any such variable to a coordinate everywhere.
    all_coords: set[Hashable] = set()
    all_data_vars: set[Hashable] = set()
    for xr_ds in xr_dataset_dict.values():
        all_coords.update(xr_ds.coords)
        all_data_vars.update(xr_ds.data_vars)
    conflicting = all_coords & all_data_vars
    if conflicting:
        for key in xr_dataset_dict:
            vars_to_promote = [
                v for v in conflicting if v in xr_dataset_dict[key].data_vars
            ]
            if vars_to_promote:
                xr_dataset_dict[key] = xr_dataset_dict[key].set_coords(vars_to_promote)

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


def _commit_new_file(staged: Path, destination: Path) -> None:
    """
    Move a staged file to a destination that did not exist when staging began.

    A hard link is used since it fails rather than overwrites if another writer
    created the destination in the meantime. In that case the destination is
    replaced with the permission bits of the file created by the other writer.
    """
    try:
        os.link(staged, destination)
    except FileExistsError:
        shutil.copymode(destination, staged)
        os.replace(staged, destination)
    except OSError:
        # the file system does not support hard links
        os.replace(staged, destination)
    else:
        staged.unlink()


def xarray_to_h5netcdf_with_complex_numbers(
    xarray_dataset: xr.Dataset,
    file_path: str | Path,
    compute: bool = True,
    compression_level: int | None = None,
    snapshot_in_group: bool = False,
) -> None:
    """
    Write an xarray dataset to a netcdf file using the h5netcdf engine.

    Args:
        xarray_dataset: The dataset to write.
        file_path: Path of the netcdf file to write.
        compute: If False the write is returned as a Dask delayed job which is
            computed with a progress bar rather than written eagerly.
        compression_level: Deflate compression level between 0 and 9 where 0
            means no compression. If None the level is read from
            ``qcodes.config.dataset.export_netcdf_compression_level``.
        snapshot_in_group: If True the snapshot is stored as a variable in the
            ``qcodes_metadata`` group, compressed with ``compression_level``,
            and the ``snapshot`` attribute is replaced by a pointer to it.

    """
    import cf_xarray as cf_xr
    from pandas import MultiIndex

    if compression_level is None:
        compression_level = int(qcodes.config.dataset.export_netcdf_compression_level)

    snapshot_attr = xarray_dataset.attrs.get("snapshot")
    if _snapshot_pointer_location(snapshot_attr) is not None:
        # The pointer refers to a group in the file the dataset was read from,
        # which is not copied, so writing it would leave a dangling reference.
        raise ValueError(
            "The snapshot attribute of this dataset is a pointer to the "
            f"{SNAPSHOT_LOCATION!r} group of the NetCDF file it was loaded from. "
            "Load the file with qcodes.dataset.load_from_netcdf or set the "
            "snapshot attribute to the JSON string "
            "json.dumps(qcodes.dataset.snapshot_from_netcdf(path)) before "
            "exporting it."
        )
    group_snapshot: str | None = None
    if snapshot_in_group and _should_write_snapshot_to_group(snapshot_attr):
        if SNAPSHOT_GROUP in xarray_dataset.variables or (
            SNAPSHOT_GROUP in xarray_dataset.dims
        ):
            _LOGGER.warning(
                "Cannot store the snapshot in the %r group since the dataset "
                "has a variable or dimension with that name. Storing it as the "
                "snapshot attribute instead.",
                SNAPSHOT_GROUP,
                extra={"file_name": str(file_path)},
            )
        else:
            group_snapshot = snapshot_attr
            xarray_dataset = xarray_dataset.assign_attrs(snapshot=_SNAPSHOT_POINTER)

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

    # The compression options are merged into the encoding of each variable
    # rather than passed as the encoding argument of to_netcdf, since that
    # replaces the existing encoding of a variable and would drop settings such
    # as dtype, scale_factor and _FillValue of a dataset loaded from file.
    internal_ds = _without_inherited_compression(internal_ds)
    if compression_level > 0:
        encoding = _netcdf_compression_encoding(internal_ds, compression_level)
        for name, variable_encoding in encoding.items():
            internal_ds.variables[name].encoding.update(variable_encoding)
        internal_ds = _rechunk_to_match_encoding(internal_ds, encoding)

    # A grouped snapshot is appended after the data has been written, so such
    # an export is staged in a temporary file owned by this call and only moved
    # into place once both writes have succeeded. A failure therefore never
    # removes or corrupts an existing file or another writer's file. ``~`` is
    # expanded since not all backends do so, and a symlinked destination is
    # resolved so that its target rather than the link is replaced.
    destination = Path(file_path).expanduser()
    if destination.is_symlink():
        destination = destination.resolve()
    stage = group_snapshot is not None
    replaces_existing = stage and destination.exists()
    if replaces_existing:
        # mkstemp creates the file readable only by the owner, so the staged
        # data is never more accessible than the file it replaces.
        fd, staged_name = tempfile.mkstemp(
            dir=destination.parent, prefix=".qcodes-", suffix=".nc"
        )
        os.close(fd)
        write_path = Path(staged_name)
    elif stage:
        # a new file is created by the writer with the default permissions
        write_path = destination.with_name(f".qcodes-{uuid.uuid4().hex}.nc")
    else:
        write_path = destination
    try:
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
                path=write_path,
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
                    _LOGGER.info(
                        "Writing netcdf file using Dask delayed writer.",
                        extra={"file_name": str(file_path)},
                    )
                    maybe_write_job.compute()
        if group_snapshot is not None:
            _write_snapshot_to_group(
                write_path,
                group_snapshot,
                compression_level=compression_level,
                chunk_size_bytes=_TARGET_CHUNK_SIZE_BYTES,
            )
            if replaces_existing:
                shutil.copymode(destination, write_path)
                os.replace(write_path, destination)
            else:
                _commit_new_file(write_path, destination)
    except BaseException:
        if stage:
            write_path.unlink(missing_ok=True)
        raise
