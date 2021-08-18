from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sized
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from qcodes.dataset.data_set_protocol import DataSetProtocol
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.param_spec import ParamSpec, ParamSpecBase
from qcodes.dataset.descriptions.rundescriber import RunDescriber
from qcodes.dataset.descriptions.versioning.converters import new_to_old
from qcodes.dataset.descriptions.versioning.rundescribertypes import Shapes
from qcodes.dataset.export_config import (
    DataExportType,
    get_data_export_path,
    get_data_export_prefix,
    get_data_export_type,
)
from qcodes.dataset.guids import generate_guid
from qcodes.dataset.linked_datasets.links import Link, links_to_str
from qcodes.instrument.parameter import _BaseParameter
from qcodes.utils.helpers import NumpyJSONEncoder

from .data_set import SPECS, CompletedError
from .data_set_cache import DataSetCacheInMem
from .descriptions.versioning import serialization as serial
from .exporters.export_to_csv import dataframe_to_csv
from .linked_datasets.links import str_to_links

if TYPE_CHECKING:
    import xarray as xr

log = logging.getLogger(__name__)


class DataSetInMem(DataSetProtocol, Sized):
    def __init__(
        self,
        run_id: int,
        counter: int,
        name: str,
        exp_id: int,
        exp_name: str,
        sample_name: str,
        guid: str,
        path_to_db: str,
        run_timestamp_raw: Optional[float],
        completed_timestamp_raw: Optional[float],
        metadata: Optional[Mapping[str, Any]] = None,
        rundescriber: Optional[RunDescriber] = None,
        parent_dataset_links: Optional[Sequence[Link]] = None,
    ) -> None:
        """Note that the constructor is considered private.

        A ``DataSetInMem``
        should be constructed either using ``create_new_run`` or ``load_from_netcdf``
        """

        self._run_id = run_id
        self._counter = counter
        self._name = name
        self._exp_id = exp_id
        self._exp_name = exp_name
        self._sample_name = sample_name
        self._guid = guid
        self._cache = DataSetCacheInMem(self)
        self._run_timestamp_raw = run_timestamp_raw
        self._completed_timestamp_raw = completed_timestamp_raw
        self._path_to_db = path_to_db
        if metadata is None:
            self._metadata = {}
        else:
            self._metadata = dict(metadata)
        if rundescriber is None:
            interdeps = InterDependencies_()
            rundescriber = RunDescriber(interdeps, shapes=None)

        self._rundescriber = rundescriber
        if parent_dataset_links is not None:
            self._parent_dataset_links = list(parent_dataset_links)
        else:
            self._parent_dataset_links = []

    @classmethod
    def create_new_run(
        cls,
        name: str,
        path_to_db: Optional[Union[Path, str]] = None,
        exp_id: Optional[int] = None,
    ) -> DataSetInMem:

        from qcodes.dataset.sqlite.database import conn_from_dbpath_or_conn
        from qcodes.dataset.sqlite.queries import (
            create_run,
            get_experiment_name_from_experiment_id,
            get_sample_name_from_experiment_id,
        )

        from .experiment_settings import get_default_experiment_id

        if path_to_db is not None:
            path_to_db = str(path_to_db)

        conn = conn_from_dbpath_or_conn(conn=None, path_to_db=path_to_db)
        try:
            if exp_id is None:
                exp_id = get_default_experiment_id(conn)
            name = name or "dataset"
            sample_name = get_sample_name_from_experiment_id(conn, exp_id)
            exp_name = get_experiment_name_from_experiment_id(conn, exp_id)
            guid = generate_guid()

            run_counter, run_id, __ = create_run(
                conn, exp_id, name, guid=guid, parameters=None, create_run_table=False
            )

            ds = cls(
                run_id=run_id,
                counter=run_counter,
                name=name,
                exp_id=exp_id,
                exp_name=exp_name,
                sample_name=sample_name,
                guid=guid,
                path_to_db=conn.path_to_dbfile,
                run_timestamp_raw=None,
                completed_timestamp_raw=None,
                metadata=None,
            )
        finally:
            conn.close()

        return ds

    @classmethod
    def load_from_netcdf(cls, path: Union[Path, str]) -> DataSetInMem:
        import xarray as xr

        loaded_data = xr.load_dataset(path)
        # todo do we want to create this run in the sqlites run table if not
        # exists. e.g. load by guid and then if that is not there create one.

        parent_dataset_links = str_to_links(
            loaded_data.attrs.get("parent_dataset_links", "[]")
        )

        ds = cls(
            run_id=loaded_data.captured_run_id,
            counter=loaded_data.captured_counter,
            name=loaded_data.ds_name,
            exp_id=0,
            exp_name=loaded_data.exp_name,
            sample_name=loaded_data.sample_name,
            guid=loaded_data.guid,
            path_to_db="",  # todo what should this be?
            run_timestamp_raw=loaded_data.run_timestamp_raw,
            completed_timestamp_raw=loaded_data.completed_timestamp_raw,
            metadata={"snapshot": loaded_data.snapshot},
            rundescriber=serial.from_json_to_current(loaded_data.run_description),
            parent_dataset_links=parent_dataset_links,
        )
        ds._cache = DataSetCacheInMem(ds)
        ds._cache._data = cls._from_xarray_dataset_to_qcodes(loaded_data)

        return ds

    @staticmethod
    def _from_xarray_dataset_to_qcodes(
        xr_data: xr.Dataset,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        output: Dict[str, Dict[str, np.ndarray]] = {}
        for datavar in xr_data.data_vars:
            output[str(datavar)] = {}
            data = xr_data[datavar]
            output[str(datavar)][str(datavar)] = data.data
            coords_unexpanded = []
            for coord in data.coords:
                coords_unexpanded.append(xr_data[coord].data)
            coords_arrays = np.meshgrid(*coords_unexpanded)
            for coord_name, coord_array in zip(data.coords, coords_arrays):
                output[str(datavar)][str(coord_name)] = coord_array
        return output

    def prepare(
        self,
        *,
        snapshot: Mapping[Any, Any],
        interdeps: InterDependencies_,
        shapes: Shapes = None,
        parent_datasets: Sequence[Mapping[Any, Any]] = (),
        write_in_background: bool = False,
    ) -> None:

        self.add_snapshot(json.dumps({"station": snapshot}, cls=NumpyJSONEncoder))

        if interdeps == InterDependencies_():
            raise RuntimeError("No parameters supplied")

        self._set_interdependencies(interdeps, shapes)
        links = [Link(head=self.guid, **pdict) for pdict in parent_datasets]
        self._set_parent_dataset_links(links)

        if self.pristine:
            self._perform_start_actions()

    @property
    def pristine(self) -> bool:
        """Is this :class:`.DataSet` pristine?

        A pristine :class:`.DataSet` has not yet been started,
        meaning that parameters can still be added and removed, but results
        can not be added.
        """
        return self._run_timestamp_raw is None and self._completed_timestamp_raw is None

    @property
    def running(self) -> bool:
        """
        Is this :class:`.DataSet` currently running?

        A running :class:`.DataSet` has been started,
        but not yet completed.
        """
        return (
            self._run_timestamp_raw is not None
            and self._completed_timestamp_raw is None
        )

    @property
    def completed(self) -> bool:
        """
        Is this :class:`.DataSet` completed?

        A completed :class:`.DataSet` may not be modified in
        any way.
        """
        return self._completed_timestamp_raw is not None

    def mark_completed(self) -> None:
        """
        Mark :class:`.DataSet` as complete and thus read only and notify the subscribers
        """
        if self.completed:
            return
        if self.pristine:
            raise RuntimeError(
                "Can not mark DataSet as complete before it "
                "has been marked as started."
            )

        self._complete(True)

    @property
    def run_id(self) -> int:
        return self._run_id

    @property
    def captured_run_id(self) -> int:
        return self._run_id

    @property
    def captured_counter(self) -> int:
        return self._counter

    @property
    def guid(self) -> str:
        return self._guid

    @property
    def number_of_results(self) -> int:
        # todo how to replace
        return 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def exp_name(self) -> str:
        return self._exp_name

    @property
    def sample_name(self) -> str:
        return self._sample_name

    def run_timestamp(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> Optional[str]:
        """
        Returns run timestamp in a human-readable format

        The run timestamp is the moment when the measurement for this run
        started. If the run has not yet been started, this function returns
        None.

        Consult with :func:`time.strftime` for information about the format.
        """
        if self.run_timestamp_raw is None:
            return None
        else:
            return time.strftime(fmt, time.localtime(self.run_timestamp_raw))

    @property
    def run_timestamp_raw(self) -> Optional[float]:
        """
        Returns run timestamp as number of seconds since the Epoch

        The run timestamp is the moment when the measurement for this run
        started.
        """
        return self._run_timestamp_raw

    def completed_timestamp(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> Optional[str]:
        """
        Returns timestamp when measurement run was completed
        in a human-readable format

        If the run (or the dataset) is not completed, then returns None.

        Consult with ``time.strftime`` for information about the format.
        """
        completed_timestamp_raw = self.completed_timestamp_raw

        if completed_timestamp_raw:
            completed_timestamp: Optional[str] = time.strftime(
                fmt, time.localtime(completed_timestamp_raw)
            )
        else:
            completed_timestamp = None

        return completed_timestamp

    @property
    def completed_timestamp_raw(self) -> Optional[float]:
        """
        Returns timestamp when measurement run was completed
        as number of seconds since the Epoch

        If the run (or the dataset) is not completed, then returns None.
        """
        return self._completed_timestamp_raw

    @property
    def snapshot(self) -> Optional[Dict[str, Any]]:
        """Snapshot of the run as dictionary (or None)."""
        snapshot_json = self._snapshot_raw
        if snapshot_json is not None:
            return json.loads(snapshot_json)
        else:
            return None

    def add_snapshot(self, snapshot: str, overwrite: bool = False) -> None:
        """
        Adds a snapshot to this run.

        Args:
            snapshot: the raw JSON dump of the snapshot
            overwrite: force overwrite an existing snapshot
        """
        if self.snapshot is None or overwrite:
            self.add_metadata("snapshot", snapshot)
        elif self.snapshot is not None and not overwrite:
            log.warning(
                "This dataset already has a snapshot. Use overwrite"
                "=True to overwrite that"
            )

    @property
    def _snapshot_raw(self) -> Optional[str]:
        """Snapshot of the run as a JSON-formatted string (or None)."""
        return self._metadata.get("snapshot")

    def add_metadata(self, tag: str, metadata: Any) -> None:
        """
        Adds metadata to the :class:`.DataSet`.

        The metadata is stored under the provided tag.
        Note that None is not allowed as a metadata value.

        Args:
            tag: represents the key in the metadata dictionary
            metadata: actual metadata
        """
        from qcodes.dataset.sqlite.connection import atomic
        from qcodes.dataset.sqlite.database import conn_from_dbpath_or_conn
        from qcodes.dataset.sqlite.queries import add_meta_data

        self._metadata[tag] = metadata
        conn = conn_from_dbpath_or_conn(conn=None, path_to_db=self._path_to_db)

        try:
            with atomic(conn) as conn:
                add_meta_data(conn, self.run_id, {tag: metadata})
        finally:
            conn.close()

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def paramspecs(self) -> Dict[str, ParamSpec]:
        return {ps.name: ps for ps in self._get_paramspecs()}

    @property
    def description(self) -> RunDescriber:
        return self._rundescriber

    @property
    def parent_dataset_links(self) -> List[Link]:
        """
        Return a list of Link objects. Each Link object describes a link from
        this dataset to one of its parent datasets
        """
        return self._parent_dataset_links

    def export(
        self,
        export_type: Optional[Union[DataExportType, str]] = None,
        path: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> None:
        """
        Export data to disk with file name {prefix}{run_id}.{ext}.

        Values for the export type, path and prefix can also be set in the "dataset"
        section of qcodes config.

        Args:
            export_type: Data export type, e.g. "netcdf" or ``DataExportType.NETCDF``,
                defaults to a value set in qcodes config
            path: Export path, defaults to value set in config
            prefix: File prefix, e.g. ``qcodes_``, defaults to value set in config.

        Raises:
            ValueError: If the export data type is not specified, raise an error
        """
        export_type = get_data_export_type(export_type)

        if export_type is None:
            raise ValueError(
                "No data export type specified. Please set the export data type "
                "by using ``qcodes.dataset.export_config.set_data_export_type`` or "
                "give an explicit export_type when calling ``dataset.export`` manually."
            )

        self._export_path = self._export_data(
            export_type=export_type, path=path, prefix=prefix
        )

    @property
    def cache(self) -> DataSetCacheInMem:
        return self._cache

    def _enqueue_results(self, result_dict: Mapping[ParamSpecBase, np.ndarray]) -> None:
        """
        Enqueue the results into self._results

        Before we can enqueue the results, all values of the results dict
        must have the same length. We enqueue each parameter tree separately,
        effectively mimicking making one call to add_results per parameter
        tree.

        Deal with 'numeric' type parameters. If a 'numeric' top level parameter
        has non-scalar shape, it must be unrolled into a list of dicts of
        single values (database).
        """
        self._raise_if_not_writable()
        interdeps = self._rundescriber.interdeps

        toplevel_params = set(interdeps.dependencies).intersection(set(result_dict))
        new_results: Dict[str, Dict[str, np.ndarray]] = {}
        for toplevel_param in toplevel_params:
            inff_params = set(interdeps.inferences.get(toplevel_param, ()))
            deps_params = set(interdeps.dependencies.get(toplevel_param, ()))
            all_params = inff_params.union(deps_params).union({toplevel_param})

            new_results[toplevel_param.name] = {}
            new_results[toplevel_param.name][
                toplevel_param.name
            ] = self._reshape_array_for_cache(
                toplevel_param, result_dict[toplevel_param]
            )
            for param in all_params:
                if param is not toplevel_param:
                    new_results[toplevel_param.name][
                        param.name
                    ] = self._reshape_array_for_cache(param, result_dict[param])

        # Finally, handle standalone parameters

        standalones = set(interdeps.standalones).intersection(set(result_dict))

        if standalones:
            for st in standalones:
                new_results[st.name] = {
                    st.name: self._reshape_array_for_cache(st, result_dict[st])
                }

        self.cache.add_data(new_results)

    def _flush_data_to_database(self, block: bool = False) -> None:
        pass

    # not part of the protocol specified api

    @property
    def counter(self) -> int:
        return self._counter

    def _set_parent_dataset_links(self, links: List[Link]) -> None:
        """
        Assign one or more links to parent datasets to this dataset. It is an
        error to assign links to a non-pristine dataset

        Args:
            links: The links to assign to this dataset
        """
        # todo write to db
        if not self.pristine:
            raise RuntimeError(
                "Can not set parent dataset links on a dataset "
                "that has been started."
            )

        if not all(isinstance(link, Link) for link in links):
            raise ValueError("Invalid input. Did not receive a list of Links")

        for link in links:
            if link.head != self.guid:
                raise ValueError(
                    "Invalid input. All links must point to this dataset. "
                    "Got link(s) with head(s) pointing to another dataset."
                )

        self._parent_dataset_links = links

    def _set_interdependencies(
        self, interdeps: InterDependencies_, shapes: Shapes = None
    ) -> None:
        """
        Set the interdependencies object (which holds all added
        parameters and their relationships) of this dataset and
        optionally the shapes object that holds information about
        the shape of the data to be measured.
        """
        # todo write to db
        if not isinstance(interdeps, InterDependencies_):
            raise TypeError(
                "Wrong input type. Expected InterDepencies_, " f"got {type(interdeps)}"
            )

        if not self.pristine:
            mssg = "Can not set interdependencies on a DataSet that has been started."
            raise RuntimeError(mssg)
        self._rundescriber = RunDescriber(interdeps, shapes=shapes)

    def _get_paramspecs(self) -> SPECS:
        old_interdeps = new_to_old(self.description.interdeps)
        return list(old_interdeps.paramspecs)

    def _complete(self, value: bool) -> None:
        from qcodes.dataset.sqlite.database import conn_from_dbpath_or_conn
        from qcodes.dataset.sqlite.queries import mark_run_complete

        conn = conn_from_dbpath_or_conn(conn=None, path_to_db=self._path_to_db)
        if value:
            self._completed_timestamp_raw = time.time()
            mark_run_complete(conn, self.run_id, self._completed_timestamp_raw)

    def _perform_start_actions(self) -> None:
        """
        Perform the actions that must take place once the run has been started
        """
        from qcodes.dataset.sqlite.database import conn_from_dbpath_or_conn
        from qcodes.dataset.sqlite.queries import (
            set_run_timestamp,
            update_parent_datasets,
            update_run_description,
        )

        conn = conn_from_dbpath_or_conn(conn=None, path_to_db=self._path_to_db)

        try:
            # paramspecs = new_to_old(self.description.interdeps).paramspecs
            #
            # # for spec in paramspecs:
            # #     add_parameter(conn, self.table_name, spec)
            # todo do we want to keep adding parameters to the runs table?

            desc_str = serial.to_json_for_storage(self.description)

            update_run_description(conn, self.run_id, desc_str)
            self._run_timestamp_raw = time.time()
            set_run_timestamp(conn, self.run_id, self._run_timestamp_raw)

            pdl_str = links_to_str(self._parent_dataset_links)
            update_parent_datasets(conn, self.run_id, pdl_str)
        finally:
            conn.close()

    def _raise_if_not_writable(self) -> None:
        if self.pristine:
            raise RuntimeError(
                "This DataSet has not been marked as started. "
                "Please mark the DataSet as started before "
                "adding results to it."
            )
        if self.completed:
            raise CompletedError(
                "This DataSet is complete, no further " "results can be added to it."
            )

    @staticmethod
    def _validate_parameters(
        *params: Union[str, ParamSpec, _BaseParameter]
    ) -> List[str]:
        """
        Validate that the provided parameters have a name and return those
        names as a list.
        The Parameters may be a mix of strings, ParamSpecs or ordinary
        QCoDeS parameters.
        """

        valid_param_names = []
        for maybe_param in params:
            if isinstance(maybe_param, str):
                valid_param_names.append(maybe_param)
            else:
                try:
                    maybe_param_name = maybe_param.name
                except Exception as e:
                    raise ValueError("This parameter does not have  a name") from e
                valid_param_names.append(maybe_param_name)
        return valid_param_names

    def __len__(self) -> int:
        # todo what should this be
        return 0

    def __repr__(self) -> str:
        # todo what to put in
        return "dataset"

    @staticmethod
    def _reshape_array_for_cache(
        param: ParamSpecBase, param_data: np.ndarray
    ) -> np.ndarray:
        """
        Shape cache data so it matches data read from database.
        This means:

        - Add an extra singleton dim to array data
        - flatten non array data into a linear array.
        """
        param_data = np.atleast_1d(param_data)
        if param.type == "array":
            new_data = np.reshape(param_data, (1,) + param_data.shape)
        else:
            new_data = param_data.ravel()
        return new_data

    def _export_file_name(self, prefix: str, export_type: DataExportType) -> str:
        """Get export file name."""
        extension = export_type.value
        return f"{prefix}{self.run_id}.{extension}"

    def _export_as_netcdf(self, path: str, file_name: str) -> str:
        """Export data as netcdf to a given path with file prefix"""
        file_path = os.path.join(path, file_name)
        xarr_dataset = self.cache.to_xarray_dataset()
        data_var_kinds = [
            xarr_dataset.data_vars[data_var].dtype.kind
            for data_var in xarr_dataset.data_vars
        ]
        coord_kinds = [
            xarr_dataset.coords[coord].dtype.kind for coord in xarr_dataset.coords
        ]
        if "c" in data_var_kinds or "c" in coord_kinds:
            # see http://xarray.pydata.org/en/stable/howdoi.html
            # for how to export complex numbers
            xarr_dataset.to_netcdf(
                path=file_path, engine="h5netcdf", invalid_netcdf=True
            )
        else:
            xarr_dataset.to_netcdf(path=file_path, engine="h5netcdf")
        return path

    def _export_as_csv(self, path: str, file_name: str) -> str:
        """Export data as csv to a given path with file prefix."""
        dfdict = self.cache.to_pandas_dataframe_dict()
        dataframe_to_csv(
            dfdict=dfdict,
            path=path,
            single_file=True,
            single_file_name=file_name,
        )
        return os.path.join(path, file_name)

    def _export_data(
        self,
        export_type: DataExportType,
        path: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> Optional[str]:
        """Export data to disk with file name {prefix}{run_id}.{ext}.

        Values for the export type, path and prefix can also be set in the qcodes
        "dataset" config.

        Args:
            export_type: Data export type, e.g. DataExportType.NETCDF
            path: Export path, defaults to value set in config
            prefix: File prefix, e.g. "qcodes_", defaults to value set in config.

        Returns:
            str: Path file was saved to, returns None if no file was saved.
        """
        # Set defaults to values in config if the value was not set
        # (defaults to None)
        path = path if path is not None else get_data_export_path()
        prefix = prefix if prefix is not None else get_data_export_prefix()

        if DataExportType.NETCDF == export_type:
            file_name = self._export_file_name(
                prefix=prefix, export_type=DataExportType.NETCDF
            )
            return self._export_as_netcdf(path=path, file_name=file_name)

        elif DataExportType.CSV == export_type:
            file_name = self._export_file_name(
                prefix=prefix, export_type=DataExportType.CSV
            )
            return self._export_as_csv(path=path, file_name=file_name)

        else:
            return None

    @property
    def export_path(self) -> Optional[str]:
        return self._export_path
