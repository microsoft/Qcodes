from __future__ import annotations

import os
import warnings
from collections.abc import Mapping, Sized
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
from typing_extensions import TypeAlias

from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.param_spec import ParamSpec, ParamSpecBase
from qcodes.dataset.descriptions.rundescriber import RunDescriber
from qcodes.dataset.descriptions.versioning.rundescribertypes import Shapes
from qcodes.dataset.export_config import (
    DataExportType,
    get_data_export_name_elements,
    get_data_export_path,
    get_data_export_prefix,
    get_data_export_type,
)
from qcodes.dataset.linked_datasets.links import Link

from .descriptions.versioning.converters import new_to_old
from .exporters.export_info import ExportInfo
from .exporters.export_to_csv import dataframe_to_csv
from .exporters.export_to_xarray import xarray_to_h5netcdf_with_complex_numbers
from .sqlite.queries import raw_time_to_str_time

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

    from qcodes.parameters import ParameterBase

    from .data_set_cache import DataSetCache

# even with from __future__ import annotations
# type aliases must use the old format until we drop 3.8/3.9
array_like_types = (tuple, list, np.ndarray)
scalar_res_types: TypeAlias = Union[
    str, complex, np.integer, np.floating, np.complexfloating
]
values_type: TypeAlias = Union[scalar_res_types, np.ndarray, Sequence[scalar_res_types]]
res_type: TypeAlias = Tuple[Union["ParameterBase", str], values_type]
setpoints_type: TypeAlias = Sequence[Union[str, "ParameterBase"]]
SPECS: TypeAlias = List[ParamSpec]
# Transition period type: SpecsOrInterDeps. We will allow both as input to
# the DataSet constructor for a while, then deprecate SPECS and finally remove
# the ParamSpec class
SpecsOrInterDeps: TypeAlias = Union[SPECS, InterDependencies_]
ParameterData: TypeAlias = Dict[str, Dict[str, np.ndarray]]


class CompletedError(RuntimeError):
    pass


@runtime_checkable
class DataSetProtocol(Protocol, Sized):

    # the "persistent traits" are the attributes/properties of the DataSet
    # that are NOT tied to the representation of the DataSet in any particular
    # database
    persistent_traits: tuple[str, ...] = (
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
        shapes: Shapes | None = None,
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

    def run_timestamp(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> str | None:
        pass

    @property
    def run_timestamp_raw(self) -> float | None:
        pass

    def completed_timestamp(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> str | None:
        pass

    @property
    def completed_timestamp_raw(self) -> float | None:
        pass

    # snapshot and metadata
    @property
    def snapshot(self) -> dict[str, Any] | None:
        pass

    def add_snapshot(self, snapshot: str, overwrite: bool = False) -> None:
        pass

    @property
    def _snapshot_raw(self) -> str | None:
        pass

    def add_metadata(self, tag: str, metadata: Any) -> None:
        pass

    @property
    def metadata(self) -> dict[str, Any]:
        pass

    @property
    def path_to_db(self) -> str | None:
        pass

    # dataset description and links
    @property
    def paramspecs(self) -> dict[str, ParamSpec]:
        pass

    @property
    def description(self) -> RunDescriber:
        pass

    @property
    def parent_dataset_links(self) -> list[Link]:
        pass

    # data related members

    def export(
        self,
        export_type: DataExportType | str | None = None,
        path: str | None = None,
        prefix: str | None = None,
    ) -> None:
        pass

    @property
    def export_info(self) -> ExportInfo:
        pass

    @property
    def cache(self) -> DataSetCache[DataSetProtocol]:
        pass

    def get_parameter_data(
        self,
        *params: str | ParamSpec | ParameterBase,
        start: int | None = None,
        end: int | None = None,
        callback: Callable[[float], None] | None = None,
    ) -> ParameterData:
        pass

    def get_parameters(self) -> SPECS:
        # used by plottr
        pass

    @property
    def dependent_parameters(self) -> tuple[ParamSpecBase, ...]:
        pass

    # exporters to other in memory formats

    def to_xarray_dataarray_dict(
        self,
        *params: str | ParamSpec | ParameterBase,
        start: int | None = None,
        end: int | None = None,
    ) -> dict[str, xr.DataArray]:
        pass

    def to_xarray_dataset(
        self,
        *params: str | ParamSpec | ParameterBase,
        start: int | None = None,
        end: int | None = None,
    ) -> xr.Dataset:
        pass

    def to_pandas_dataframe_dict(
        self,
        *params: str | ParamSpec | ParameterBase,
        start: int | None = None,
        end: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        pass

    def to_pandas_dataframe(
        self,
        *params: str | ParamSpec | ParameterBase,
        start: int | None = None,
        end: int | None = None,
    ) -> pd.DataFrame:
        pass

    # private members called by various other parts or the api

    def _enqueue_results(self, result_dict: Mapping[ParamSpecBase, np.ndarray]) -> None:
        pass

    def _flush_data_to_database(self, block: bool = False) -> None:
        pass

    @property
    def _parameters(self) -> str | None:
        pass

    def _set_export_info(self, export_info: ExportInfo) -> None:
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
        export_type: DataExportType | str | None = None,
        path: str | None = None,
        prefix: str | None = None,
    ) -> None:
        """Export data to disk with file name `{prefix}{name_elements}.{ext}`.
        Name elements are names of dataset object attributes that are taken
        from the dataset and inserted into the name of the export file, for
        example if name elements are ``["captured_run_id", "guid"]``, then
        the file name will be `{prefix}{captured_run_id}_{guid}.{ext}`.
        Values for the export type, path, export_name_elements and prefix can
        also be set in the "dataset" section of qcodes config.

        Args:
            export_type: Data export type, e.g. "netcdf" or ``DataExportType.NETCDF``,
                defaults to a value set in qcodes config
            path: Export path, defaults to value set in config
            prefix: File prefix, e.g. ``qcodes_``, defaults to value set in config.

        Raises:
            ValueError: If the export data type is not specified or unknown,
                raise an error
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
                f"Export type {export_type} is unknown. Export type "
                f"should be a member of the `DataExportType` enum"
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

    def _export_data(
        self,
        export_type: DataExportType,
        path: str | None = None,
        prefix: str | None = None,
    ) -> str | None:
        """Export data to disk with file name `{prefix}{name_elements}.{ext}`.
        Name elements are names of dataset object attributes that are taken
        from the dataset and inserted into the name of the export file, for
        example if name elements are ``["captured_run_id", "guid"]``, then
        the file name will be `{prefix}{captured_run_id}_{guid}.{ext}`.
        Values for the export type, path, export_name_elements and prefix can
        also be set in the "dataset" section of qcodes config.

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
        name_elements = get_data_export_name_elements()
        post_fix = "_".join([str(getattr(self, name)) for name in name_elements])
        return f"{prefix}{post_fix}.{extension}"

    def _export_as_netcdf(self, path: str, file_name: str) -> str:
        """Export data as netcdf to a given path with file prefix"""
        file_path = os.path.join(path, file_name)
        xarr_dataset = self.to_xarray_dataset()
        xarray_to_h5netcdf_with_complex_numbers(xarr_dataset, file_path)
        return file_path

    def _export_as_csv(self, path: str, file_name: str) -> str:
        """Export data as csv to a given path with file prefix."""
        dfdict = self.to_pandas_dataframe_dict()
        dataframe_to_csv(
            dfdict=dfdict,
            path=path,
            single_file=True,
            single_file_name=file_name,
        )
        return os.path.join(path, file_name)

    def _add_metadata_to_netcdf_if_nc_exported(self, tag: str, data: Any) -> None:
        export_paths = self.export_info.export_paths
        nc_file = export_paths.get(DataExportType.NETCDF.value, None)
        if nc_file is not None:
            import h5netcdf

            try:
                with h5netcdf.File(
                    nc_file, mode="r+", decode_vlen_strings=False
                ) as h5nc_file:
                    h5nc_file.attrs[tag] = data
            except (
                FileNotFoundError,
                OSError,
            ):  # older versions of h5py may throw a OSError here
                warnings.warn(
                    f"Could not add metadata to the exported NetCDF file, "
                    f"was the file moved? GUID {self.guid}, NetCDF file {nc_file}"
                )

    @staticmethod
    def _validate_parameters(*params: str | ParamSpec | ParameterBase) -> list[str]:
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

    def run_timestamp(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> str | None:
        """
        Returns run timestamp in a human-readable format

        The run timestamp is the moment when the measurement for this run
        started. If the run has not yet been started, this function returns
        None.

        Consult with :func:`time.strftime` for information about the format.
        """
        return raw_time_to_str_time(self.run_timestamp_raw, fmt)

    def completed_timestamp(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> str | None:
        """
        Returns timestamp when measurement run was completed
        in a human-readable format

        If the run (or the dataset) is not completed, then returns None.

        Consult with ``time.strftime`` for information about the format.
        """
        return raw_time_to_str_time(self.completed_timestamp_raw, fmt)

    @property
    def dependent_parameters(self) -> tuple[ParamSpecBase, ...]:
        """
        Return all the parameters that explicitly depend on other parameters
        """
        return tuple(self.description.interdeps.dependencies.keys())


class DataSetType(str, Enum):

    DataSet = "DataSet"
    DataSetInMem = "DataSetInMem"
