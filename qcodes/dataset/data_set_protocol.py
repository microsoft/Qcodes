from __future__ import annotations

import os
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Sized,
    Tuple,
    Union,
)

import numpy as np
from typing_extensions import Protocol, runtime_checkable

from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.param_spec import ParamSpec, ParamSpecBase
from qcodes.dataset.descriptions.rundescriber import RunDescriber
from qcodes.dataset.descriptions.versioning.rundescribertypes import Shapes
from qcodes.dataset.export_config import (
    DataExportType,
    get_data_export_path,
    get_data_export_prefix,
    get_data_export_type,
)
from qcodes.dataset.linked_datasets.links import Link
from qcodes.instrument.parameter import _BaseParameter

from .descriptions.versioning.converters import new_to_old
from .exporters.export_info import ExportInfo
from .exporters.export_to_csv import dataframe_to_csv

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

    from .data_set_cache import DataSetCache

array_like_types = (tuple, list, np.ndarray)
scalar_res_types = Union[str, complex, np.integer, np.floating, np.complexfloating]
values_type = Union[scalar_res_types, np.ndarray, Sequence[scalar_res_types]]
res_type = Tuple[Union[_BaseParameter, str], values_type]
setpoints_type = Sequence[Union[str, _BaseParameter]]
SPECS = List[ParamSpec]
# Transition period type: SpecsOrInterDeps. We will allow both as input to
# the DataSet constructor for a while, then deprecate SPECS and finally remove
# the ParamSpec class
SpecsOrInterDeps = Union[SPECS, InterDependencies_]
ParameterData = Dict[str, Dict[str, np.ndarray]]


class CompletedError(RuntimeError):
    pass


@runtime_checkable
class DataSetProtocol(Protocol, Sized):

    # the "persistent traits" are the attributes/properties of the DataSet
    # that are NOT tied to the representation of the DataSet in any particular
    # database
    persistent_traits: Tuple[str, ...] = (
        "name",
        "guid",
        "number_of_results",
        "exp_name",
        "sample_name",
        "completed",
        "snapshot",
        "run_timestamp_raw",
        "description",
        "completed_timestamp_raw",
        "metadata",
        "parent_dataset_links",
        "captured_run_id",
        "captured_counter",
    )

    def prepare(
        self,
        *,
        snapshot: Mapping[Any, Any],
        interdeps: InterDependencies_,
        shapes: Shapes = None,
        parent_datasets: Sequence[Mapping[Any, Any]] = (),
        write_in_background: bool = False,
    ) -> None:
        pass

    @property
    def pristine(self) -> bool:
        pass

    @property
    def running(self) -> bool:
        pass

    @property
    def completed(self) -> bool:
        pass

    def mark_completed(self) -> None:
        pass

    # dataset attributes

    @property
    def run_id(self) -> int:
        pass

    @property
    def captured_run_id(self) -> int:
        pass

    @property
    def counter(self) -> int:
        pass

    @property
    def captured_counter(self) -> int:
        pass

    @property
    def guid(self) -> str:
        pass

    @property
    def number_of_results(self) -> int:
        pass

    @property
    def name(self) -> str:
        pass

    @property
    def exp_name(self) -> str:
        pass

    @property
    def exp_id(self) -> int:
        pass

    @property
    def sample_name(self) -> str:
        pass

    def run_timestamp(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> Optional[str]:
        pass

    @property
    def run_timestamp_raw(self) -> Optional[float]:
        pass

    def completed_timestamp(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> Optional[str]:
        pass

    @property
    def completed_timestamp_raw(self) -> Optional[float]:
        pass

    # snapshot and metadata
    @property
    def snapshot(self) -> Optional[Dict[str, Any]]:
        pass

    def add_snapshot(self, snapshot: str, overwrite: bool = False) -> None:
        pass

    @property
    def _snapshot_raw(self) -> Optional[str]:
        pass

    def add_metadata(self, tag: str, metadata: Any) -> None:
        pass

    @property
    def metadata(self) -> Dict[str, Any]:
        pass

    @property
    def path_to_db(self) -> Optional[str]:
        pass

    # dataset description and links
    @property
    def paramspecs(self) -> Dict[str, ParamSpec]:
        pass

    @property
    def description(self) -> RunDescriber:
        pass

    @property
    def parent_dataset_links(self) -> List[Link]:
        pass

    # data related members

    def export(
        self,
        export_type: Optional[Union[DataExportType, str]] = None,
        path: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> None:
        pass

    @property
    def export_info(self) -> ExportInfo:
        pass

    @property
    def cache(self) -> DataSetCache[DataSetProtocol]:
        pass

    def get_parameters(self) -> SPECS:
        # used by plottr
        pass

    # private members called by various other parts or the api

    def _enqueue_results(self, result_dict: Mapping[ParamSpecBase, np.ndarray]) -> None:
        pass

    def _flush_data_to_database(self, block: bool = False) -> None:
        pass

    @property
    def _parameters(self) -> Optional[str]:
        pass


class BaseDataSet(DataSetProtocol):

    # shared methods between all implementations of the dataset

    def the_same_dataset_as(self, other: DataSetProtocol) -> bool:
        """
        Check if two datasets correspond to the same run by comparing
        all their persistent traits. Note that this method
        does not compare the data itself.

        This function raises if the GUIDs match but anything else doesn't

        Args:
            other: the dataset to compare self to
        """
        if not isinstance(other, DataSetProtocol):
            return False

        guids_match = self.guid == other.guid

        # note that the guid is in itself a persistent trait of the DataSet.
        # We therefore do not need to handle the case of guids not equal
        # but all persistent traits equal, as this is not possible.
        # Thus, if all persistent traits are the same we can safely return True
        for attr in self.persistent_traits:
            if getattr(self, attr) != getattr(other, attr):
                if guids_match:
                    raise RuntimeError(
                        "Critical inconsistency detected! "
                        "The two datasets have the same GUID, "
                        f'but their "{attr}" differ.'
                    )
                return False

        return True

    def get_parameters(self) -> SPECS:
        old_interdeps = new_to_old(self.description.interdeps)
        return list(old_interdeps.paramspecs)

    def export(
        self,
        export_type: Optional[Union[DataExportType, str]] = None,
        path: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> None:
        """Export data to disk with file name {prefix}{run_id}.{ext}.
        Values for the export type, path and prefix can also be set in the "dataset"
        section of qcodes config.

        Args:
            export_type: Data export type, e.g. "netcdf" or ``DataExportType.NETCDF``,
                defaults to a value set in qcodes config
            path: Export path, defaults to value set in config
            prefix: File prefix, e.g. ``qcodes_``, defaults to value set in config.

        Raises:
            ValueError: If the export data type is not specified or unknown, raise an error
        """
        parsed_export_type = get_data_export_type(export_type)

        if parsed_export_type is None and export_type is None:
            raise ValueError(
                "No data export type specified. Please set the export data type "
                "by using ``qcodes.dataset.export_config.set_data_export_type`` or "
                "give an explicit export_type when calling ``dataset.export`` manually."
            )
        elif parsed_export_type is None:
            raise ValueError(
                f"Export type {export_type} is unknown. Export type should be a member of the `DataExportType` enum"
            )

        export_path = self._export_data(
            export_type=parsed_export_type, path=path, prefix=prefix
        )
        export_info = self.export_info
        if export_path is not None:
            export_info.export_paths[parsed_export_type.value] = os.path.abspath(
                export_path
            )

        self._set_export_info(export_info)

    def _set_export_info(self, export_info: ExportInfo) -> None:
        self.add_metadata("export_info", export_info.to_str())
        self._export_info = export_info

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

    def _export_file_name(self, prefix: str, export_type: DataExportType) -> str:
        """Get export file name"""
        extension = export_type.value
        return f"{prefix}{self.run_id}.{extension}"

    def _export_as_netcdf(self, path: str, file_name: str) -> str:
        """Export data as netcdf to a given path with file prefix"""
        file_path = os.path.join(path, file_name)
        xarr_dataset = self._get_data_as_xr_ds_for_export()
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
        return file_path

    def _export_as_csv(self, path: str, file_name: str) -> str:
        """Export data as csv to a given path with file prefix."""
        dfdict = self._get_data_as_pd_dict_for_export()
        dataframe_to_csv(
            dfdict=dfdict,
            path=path,
            single_file=True,
            single_file_name=file_name,
        )
        return os.path.join(path, file_name)

    def _get_data_as_pd_dict_for_export(self) -> Dict[str, pd.DataFrame]:
        """
        Abstract over the fact that preferred source of data may be cache or
        loaded from the db depending on the implementation
        """
        raise NotImplementedError()

    def _get_data_as_xr_ds_for_export(self) -> xr.Dataset:
        """
        Abstract over the fact that preferred source of data may be cache or
        loaded from the db depending on the implementation.
        """
        raise NotImplementedError()
