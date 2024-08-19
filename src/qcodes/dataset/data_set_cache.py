from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

import numpy as np

from qcodes.dataset.exporters.export_info import ExportInfo
from qcodes.dataset.sqlite.queries import completed, load_new_data_for_rundescriber

from .exporters.export_to_pandas import (
    load_to_concatenated_dataframe,
    load_to_dataframe_dict,
)
from .exporters.export_to_xarray import (
    load_to_xarray_dataarray_dict,
    load_to_xarray_dataset,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd
    import xarray as xr

    from qcodes.dataset.descriptions.rundescriber import RunDescriber
    from qcodes.dataset.sqlite.connection import ConnectionPlus

    # used in forward refs that cannot be detected
    from .data_set import DataSet  # noqa F401
    from .data_set_in_memory import DataSetInMem
    from .data_set_protocol import DataSetProtocol, ParameterData

DatasetType_co = TypeVar("DatasetType_co", bound="DataSetProtocol", covariant=True)

log = logging.getLogger(__name__)


class DataSetCache(Generic[DatasetType_co]):
    """
    The DataSetCache contains a in memory representation of the
    data in this dataset as well a a method to progressively read data
    from the db as it is written and methods to append data as it is received
    without writing it to disk. The cache can either be loaded from the db
    or produced as an in memory cache. It is not possible to combine these
    two ways of producing a dataset cache. The cache is available in the
    same formats as :py:class:`.DataSet.get_parameter_data` and
    :py:class:`.DataSet.to_pandas_dataframe_dict`
    """

    def __init__(self, dataset: DatasetType_co):
        self._dataset = dataset
        self._data: ParameterData = {}
        #: number of rows read per parameter tree (by the name of the dependent parameter)
        self._read_status: dict[str, int] = {}
        #: number of rows written per parameter tree (by the name of the dependent parameter)
        self._write_status: dict[str, int | None] = {}
        self._loaded_from_completed_ds = False
        self._live: bool | None = None

    @property
    def rundescriber(self) -> RunDescriber:
        return self._dataset.description

    @property
    def live(self) -> bool | None:
        """
        If true this cache has been produced by appending data as measured.
        If false the data has been read from disk.
        If None, then the cache does not yet have any data.
        """
        return self._live

    def data(self) -> ParameterData:
        """
        Loads data from the database on disk if needed and returns
        the cached data. The cached data is in almost the same format as
        :py:class:`.DataSet.get_parameter_data`. However if a shape is provided
        as part of the dataset metadata and fewer datapoints than expected are
        returned the missing values will be replaced by `NaN` or zeroes
        depending on the datatype.

        Returns:
            The cached dataset.
        """
        if not self.live:
            self.load_data_from_db()

        return self._data

    def prepare(self) -> None:
        """
        Set up the internal datastructure of the cache.
        Must be called after the dataset has been setup with
        interdependencies but before data is added to the dataset.
        """

        if self._data == {}:
            self._data = self.rundescriber.interdeps._empty_data_dict()
        else:
            raise RuntimeError("Cannot prepare a cache that is not empty")

    def load_data_from_db(self) -> None:
        """
        Load the data from an on-disk format in case the cache is not live

        Should be implemented in a specific subclass that knows how to read data
        from disk
        """

    def add_data(self, new_data: Mapping[str, Mapping[str, np.ndarray]]) -> None:
        if self.live is False:
            raise RuntimeError(
                "Cannot append live data to a dataset that has "
                "been fully or partially loaded from a database."
            )

        expanded_data = {}
        for param_name, single_param_dict in new_data.items():
            expanded_data[param_name] = _expand_single_param_dict(single_param_dict)

        (self._write_status, self._data) = (
            append_shaped_parameter_data_to_existing_arrays(
                self.rundescriber,
                self._write_status,
                self._data,
                new_data=expanded_data,
            )
        )

        if not all(status is None for status in self._write_status.values()):
            self._live = True

    def to_pandas_dataframe_dict(self) -> dict[str, pd.DataFrame]:
        """
        Convert the cached dataset to Pandas dataframes. The returned dataframes
        are in the same format :py:class:`.DataSet.to_pandas_dataframe_dict`.

        Returns:
            A dict from parameter name to Pandas Dataframes. Each dataframe
            represents one parameter tree.
        """
        data = self.data()
        return load_to_dataframe_dict(data)

    def to_pandas_dataframe(self) -> pd.DataFrame:
        """
        Convert the cached dataset to Pandas dataframes. The returned dataframes
        are in the same format :py:class:`.DataSet.to_pandas_dataframe_dict`.

        Returns:
            A dict from parameter name to Pandas Dataframes. Each dataframe
            represents one parameter tree.
        """
        data = self.data()
        return load_to_concatenated_dataframe(data)

    def to_xarray_dataarray_dict(
        self, *, use_multi_index: Literal["auto", "always", "never"] = "auto"
    ) -> dict[str, xr.DataArray]:  # noqa: F821
        """
        Returns the values stored in the :class:`.dataset.data_set.DataSet` as a dict of
        :py:class:`xr.DataArray` s
        Each element in the dict is indexed by the names of the dependent parameters.

        Returns:
            Dictionary from requested parameter names to :py:class:`xr.DataArray` s
            with the requested parameter(s) as a column(s) and coordinates
            formed by the dependencies.

        """
        data = self.data()
        return load_to_xarray_dataarray_dict(
            self._dataset, data, use_multi_index=use_multi_index
        )

    def to_xarray_dataset(
        self, *, use_multi_index: Literal["auto", "always", "never"] = "auto"
    ) -> xr.Dataset:
        """
        Returns the values stored in the :class:`.dataset.data_set.DataSet` as a
        :py:class:`xr.Dataset` object.

        Note that if the dataset contains data for multiple parameters that do
        not share the same setpoints it is recommended to use
        :py:class:`.to_xarray_dataarray_dict`

        Returns:
            :py:class:`xr.Dataset` with the requested parameter(s) data as
            :py:class:`xr.DataArray` s and coordinates formed by the dependencies.

        """
        data = self.data()
        return load_to_xarray_dataset(
            self._dataset, data, use_multi_index=use_multi_index
        )


def load_new_data_from_db_and_append(
    conn: ConnectionPlus,
    table_name: str,
    rundescriber: RunDescriber,
    write_status: Mapping[str, int | None],
    read_status: Mapping[str, int],
    existing_data: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[dict[str, int | None], dict[str, int], dict[str, dict[str, np.ndarray]]]:
    """
    Append any new data in the db to an already existing datadict and return the merged
    data.

    Args:
        conn: The connection to the sqlite database
        table_name: The name of the table the data is stored in
        rundescriber: The rundescriber that describes the run
        write_status: Mapping from dependent parameter name to number of rows
          written to the cache previously.
        read_status: Mapping from dependent parameter name to number of rows
          read from the db previously.
        existing_data: Mapping from dependent parameter name to mapping
          from parameter name to numpy arrays that the data should be
          inserted into.
          appended to.

    Returns:
        Updated write and read status, and the updated ``data``

    """
    new_data, updated_read_status = load_new_data_for_rundescriber(
        conn, table_name, rundescriber, read_status
    )

    (updated_write_status, merged_data) = (
        append_shaped_parameter_data_to_existing_arrays(
            rundescriber, write_status, existing_data, new_data
        )
    )
    return updated_write_status, updated_read_status, merged_data


def append_shaped_parameter_data_to_existing_arrays(
    rundescriber: RunDescriber,
    write_status: Mapping[str, int | None],
    existing_data: Mapping[str, Mapping[str, np.ndarray]],
    new_data: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[dict[str, int | None], dict[str, dict[str, np.ndarray]]]:
    """
    Append datadict to an already existing datadict and return the merged
    data.

    Args:
        rundescriber: The rundescriber that describes the run
        write_status: Mapping from dependent parameter name to number of rows
          written to the cache previously.
        new_data: Mapping from dependent parameter name to mapping
          from parameter name to numpy arrays that the data should be
          appended to.
        existing_data: Mapping from dependent parameter name to mapping
          from parameter name to numpy arrays of new data.

    Returns:
        Updated write and read status, and the updated ``data``
    """
    parameters = tuple(ps.name for ps in rundescriber.interdeps.non_dependencies)
    merged_data = {}

    updated_write_status = dict(write_status)

    for meas_parameter in parameters:
        existing_data_1_tree = existing_data.get(meas_parameter, {})

        new_data_1_tree = new_data.get(meas_parameter, {})

        shapes = rundescriber.shapes
        if shapes is not None:
            shape = shapes.get(meas_parameter, None)
        else:
            shape = None

        (merged_data[meas_parameter], updated_write_status[meas_parameter]) = (
            _merge_data(
                existing_data_1_tree,
                new_data_1_tree,
                shape,
                single_tree_write_status=write_status.get(meas_parameter),
                meas_parameter=meas_parameter,
            )
        )
    return updated_write_status, merged_data


def _merge_data(
    existing_data: Mapping[str, np.ndarray],
    new_data: Mapping[str, np.ndarray],
    shape: tuple[int, ...] | None,
    single_tree_write_status: int | None,
    meas_parameter: str,
) -> tuple[dict[str, np.ndarray], int | None]:
    subtree_merged_data = {}
    subtree_parameters = existing_data.keys()

    if not set(new_data.keys()).issubset(set(existing_data.keys())):
        raise RuntimeError(
            "Trying to add unexpected key to cache."
            "The following keys were unexpected: "
            f"{set(new_data.keys() - existing_data.keys())}"
        )

    new_write_status: int | None
    single_param_merged_data, new_write_status = _merge_data_single_param(
        existing_data.get(meas_parameter),
        new_data.get(meas_parameter),
        shape,
        single_tree_write_status,
    )
    if single_param_merged_data is not None:
        subtree_merged_data[meas_parameter] = single_param_merged_data

    for subtree_param in subtree_parameters:
        if subtree_param != meas_parameter:
            single_param_merged_data, new_write_status = _merge_data_single_param(
                existing_data.get(subtree_param),
                new_data.get(subtree_param),
                shape,
                single_tree_write_status,
            )
            if single_param_merged_data is not None:
                subtree_merged_data[subtree_param] = single_param_merged_data

    return subtree_merged_data, new_write_status


def _merge_data_single_param(
    existing_values: np.ndarray | None,
    new_values: np.ndarray | None,
    shape: tuple[int, ...] | None,
    single_tree_write_status: int | None,
) -> tuple[np.ndarray | None, int | None]:
    merged_data: np.ndarray | None
    if (
        existing_values is not None and existing_values.size != 0
    ) and new_values is not None:
        (merged_data, new_write_status) = _insert_into_data_dict(
            existing_values, new_values, single_tree_write_status, shape=shape
        )
    elif new_values is not None:
        (merged_data, new_write_status) = _create_new_data_dict(new_values, shape)
    elif existing_values is not None:
        merged_data = existing_values
        new_write_status = single_tree_write_status
    else:
        merged_data = None
        new_write_status = None
    return merged_data, new_write_status


def _create_new_data_dict(
    new_values: np.ndarray, shape: tuple[int, ...] | None
) -> tuple[np.ndarray, int]:
    if shape is None:
        return new_values, new_values.size
    elif new_values.size > 0:
        n_values = new_values.size
        data = np.zeros(shape, dtype=new_values.dtype)

        if new_values.dtype.kind == "f":
            data[:] = np.nan
        elif new_values.dtype.kind == "c":
            data[:] = np.nan + 1j * np.nan

        data.ravel()[0:n_values] = new_values.ravel()
        return data, n_values
    else:
        return new_values, new_values.size


def _insert_into_data_dict(
    existing_values: np.ndarray,
    new_values: np.ndarray,
    write_status: int | None,
    shape: tuple[int, ...] | None,
) -> tuple[np.ndarray, int | None]:
    if new_values.size == 0:
        return existing_values, write_status

    if shape is None or write_status is None:
        try:
            data = np.append(existing_values, new_values, axis=0)
        except ValueError:
            # we cannot append into a ragged array so make that manually
            n_existing = existing_values.shape[0]
            n_new = new_values.shape[0]
            n_rows = n_existing + n_new
            data = np.ndarray((n_rows,), dtype=object)

            for i in range(n_existing):
                data[i] = np.atleast_1d(existing_values[i])
            for i, j in enumerate(range(n_existing, n_existing + n_new)):
                data[j] = np.atleast_1d(new_values[i])
        return data, None
    else:
        if existing_values.dtype.kind in ("U", "S"):
            # string type arrays may be too small for the new data
            # read so rescale if needed.
            if new_values.dtype.itemsize > existing_values.dtype.itemsize:
                existing_values = existing_values.astype(new_values.dtype)
        n_values = new_values.size
        new_write_status = write_status + n_values
        if new_write_status > existing_values.size:
            log.warning(
                f"Incorrect shape of dataset: Dataset is expected to "
                f"contain {existing_values.size} points but trying to "
                f"add an amount of data that makes it contain {new_write_status} points. Cache will "
                f"be flattened into a 1D array"
            )
            return (
                np.append(existing_values.flatten(), new_values.flatten(), axis=0),
                new_write_status,
            )
        else:
            existing_values.ravel()[write_status:new_write_status] = new_values.ravel()
            return existing_values, new_write_status


def _expand_single_param_dict(
    single_param_dict: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    sizes = {name: array.size for name, array in single_param_dict.items()}
    maxsize = max(sizes.values())
    max_names = tuple(name for name, size in sizes.items() if size == maxsize)

    expanded_param_dict = {}
    for name, array in single_param_dict.items():
        if name in max_names:
            expanded_param_dict[name] = array
        else:
            assert array.size == 1
            expanded_param_dict[name] = np.full_like(
                single_param_dict[max_names[0]], array.ravel()[0], dtype=array.dtype
            )

    return expanded_param_dict


class DataSetCacheInMem(DataSetCache["DataSetInMem"]):
    pass


class DataSetCacheDeferred(DataSetCacheInMem):
    def __init__(self, dataset: DataSetInMem, loaded_data: Path | str):
        super().__init__(dataset)
        self._xr_dataset_path = Path(loaded_data)

    def load_data_from_db(self) -> None:
        if self._data == {}:
            loaded_data = self._load_xr_dataset()
            self._data = self._dataset._from_xarray_dataset_to_qcodes_raw_data(
                loaded_data
            )

    def _load_xr_dataset(self) -> xr.Dataset:
        import cf_xarray as cfxr
        import xarray as xr

        loaded_data = xr.load_dataset(self._xr_dataset_path, engine="h5netcdf")
        loaded_data = cfxr.coding.decode_compress_to_multi_index(loaded_data)
        export_info = ExportInfo.from_str(loaded_data.attrs.get("export_info", ""))
        export_info.export_paths["nc"] = str(self._xr_dataset_path)
        loaded_data.attrs["export_info"] = export_info.to_str()
        return loaded_data

    def to_xarray_dataset(
        self, *, use_multi_index: Literal["auto", "always", "never"] = "auto"
    ) -> xr.Dataset:
        loaded_data = self._load_xr_dataset()
        if use_multi_index == "always":
            ds = loaded_data.stack()
        elif use_multi_index == "never":
            ds = loaded_data.unstack()
        else:
            ds = loaded_data
        return ds


class DataSetCacheWithDBBackend(DataSetCache["DataSet"]):
    def load_data_from_db(self) -> None:
        """
        Loads data from the dataset into the cache.
        If new data has been added to the dataset since the last time
        this method was called, calling this method again would load
        that new portion of the data and append to the already loaded data.
        If the dataset is marked completed and data has already been loaded
        no load will be performed.
        """
        if self.live:
            raise RuntimeError(
                "Cannot load data into this cache from the "
                "database because this dataset is being built "
                "in-memory."
            )

        if self._loaded_from_completed_ds:
            return
        # Only updated the completed property if necessary to avoid the warning emitted by
        # mark_run_completed if the run is already marked completed.
        is_completed = completed(self._dataset.conn, self._dataset.run_id)
        if self._dataset.completed != is_completed:
            self._dataset.completed = is_completed
        if self._dataset.completed:
            self._loaded_from_completed_ds = True
        if self._data == {}:
            self.prepare()
        (
            self._write_status,
            self._read_status,
            self._data,
        ) = load_new_data_from_db_and_append(
            self._dataset.conn,
            self._dataset.table_name,
            self.rundescriber,
            self._write_status,
            self._read_status,
            self._data,
        )
        data_not_read = all(
            status is None or status == 0 for status in self._write_status.values()
        )
        if not data_not_read:
            self._live = False
