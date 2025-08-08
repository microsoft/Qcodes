from __future__ import annotations

import logging
import os
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from warnings import warn

import numpy as np
from opentelemetry import trace
from tqdm.auto import tqdm

from qcodes.dataset.data_set import DataSet, load_by_id
from qcodes.dataset.data_set_in_memory import load_from_netcdf
from qcodes.dataset.dataset_helpers import _add_run_to_runs_table
from qcodes.dataset.experiment_container import _create_exp_if_needed
from qcodes.dataset.export_config import get_data_export_path
from qcodes.dataset.sqlite.connection import AtomicConnection, atomic
from qcodes.dataset.sqlite.database import (
    connect,
    get_db_version_and_newest_available_version,
)
from qcodes.dataset.sqlite.queries import (
    _populate_results_table,
    get_exp_ids_from_run_ids,
    get_experiment_attributes_by_exp_id,
    get_runid_from_guid,
    get_runs,
    is_run_id_in_database,
)

if TYPE_CHECKING:
    from qcodes.dataset.data_set_protocol import DataSetProtocol

_LOG = logging.getLogger(__name__)
_TRACER = trace.get_tracer(__name__)


@_TRACER.start_as_current_span(f"{__name__}.extract_runs_into_db")
def extract_runs_into_db(
    source_db_path: str | Path,
    target_db_path: str | Path,
    *run_ids: int,
    upgrade_source_db: bool = False,
    upgrade_target_db: bool = False,
) -> None:
    """
    Extract a selection of runs into another DB file. All runs must come from
    the same experiment. They will be added to an experiment with the same name
    and ``sample_name`` in the target db. If such an experiment does not exist, it
    will be created.

    Args:
        source_db_path: Path to the source DB file
        target_db_path: Path to the target DB file. The target DB file will be
          created if it does not exist.
        run_ids: The ``run_id``'s of the runs to copy into the target DB file
        upgrade_source_db: If the source DB is found to be in a version that is
          not the newest, should it be upgraded?
        upgrade_target_db: If the target DB is found to be in a version that is
          not the newest, should it be upgraded?

    """
    # Check for versions
    (s_v, new_v) = get_db_version_and_newest_available_version(source_db_path)
    if s_v < new_v and not upgrade_source_db:
        warn(
            f"Source DB version is {s_v}, but this function needs it to be"
            f" in version {new_v}. Run this function again with "
            "upgrade_source_db=True to auto-upgrade the source DB file."
        )
        return

    if os.path.exists(target_db_path):
        (t_v, new_v) = get_db_version_and_newest_available_version(target_db_path)
        if t_v < new_v and not upgrade_target_db:
            warn(
                f"Target DB version is {t_v}, but this function needs it to "
                f"be in version {new_v}. Run this function again with "
                "upgrade_target_db=True to auto-upgrade the target DB file."
            )
            return

    source_conn = connect(source_db_path)

    # Validate that all runs are in the source database
    do_runs_exist = is_run_id_in_database(source_conn, *run_ids)
    if False in do_runs_exist.values():
        source_conn.close()
        non_existing_ids = [rid for rid in run_ids if not do_runs_exist[rid]]
        err_mssg = (
            "Error: not all run_ids exist in the source database. "
            "The following run(s) is/are not present: "
            f"{non_existing_ids}"
        )
        raise ValueError(err_mssg)

    # Validate that all runs are from the same experiment

    source_exp_ids = np.unique(get_exp_ids_from_run_ids(source_conn, run_ids))
    if len(source_exp_ids) != 1:
        source_conn.close()
        raise ValueError(
            "Did not receive runs from a single experiment. "
            f"Got runs from experiments {source_exp_ids}"
        )

    # Fetch the attributes of the runs' experiment
    # hopefully, this is enough to uniquely identify the experiment
    exp_attrs = get_experiment_attributes_by_exp_id(source_conn, source_exp_ids[0])

    # Massage the target DB file to accommodate the runs
    # (create new experiment if needed)

    target_conn = connect(target_db_path)

    # this function raises if the target DB file has several experiments
    # matching both the name and sample_name

    try:
        with atomic(target_conn) as target_conn:
            target_exp_id = _create_exp_if_needed(
                target_conn,
                exp_attrs["name"],
                exp_attrs["sample_name"],
                exp_attrs["format_string"],
                exp_attrs["start_time"],
                exp_attrs["end_time"],
            )

            # Finally insert the runs
            for run_id in run_ids:
                _extract_single_dataset_into_db(
                    DataSet(run_id=run_id, conn=source_conn), target_conn, target_exp_id
                )
    finally:
        source_conn.close()
        target_conn.close()


def _extract_single_dataset_into_db(
    dataset: DataSet, target_conn: AtomicConnection, target_exp_id: int
) -> None:
    """
    NB: This function should only be called from within
    meth:`extract_runs_into_db`

    Insert the given dataset into the specified database file as the latest
    run.

    Trying to insert a run already in the DB is a NOOP.

    Args:
        dataset: A dataset representing the run to be copied
        target_conn: connection to the DB. Must be atomically guarded
        target_exp_id: The ``exp_id`` of the (target DB) experiment in which to
          insert the run

    """

    if not dataset.completed:
        raise ValueError(
            "Dataset not completed. An incomplete dataset "
            "can not be copied. The incomplete dataset has "
            f"GUID: {dataset.guid} and run_id: {dataset.run_id}"
        )

    source_conn = dataset.conn

    run_id = get_runid_from_guid(target_conn, dataset.guid)

    if run_id is not None:
        return

    _, _, target_table_name = _add_run_to_runs_table(
        dataset, target_conn, target_exp_id
    )
    assert target_table_name is not None
    _populate_results_table(
        source_conn, target_conn, dataset.table_name, target_table_name
    )


@_TRACER.start_as_current_span(f"{__name__}.export_datasets_and_create_metadata_db")
def export_datasets_and_create_metadata_db(
    source_db_path: str | Path,
    target_db_path: str | Path,
    export_path: str | Path | None = None,
) -> dict[int, Literal["exported", "copied_as_is", "failed"]]:
    """
    Export all datasets from a source database to NetCDF files and create
    a new database file containing only metadata (no raw data) for those exported
    datasets. Datasets that cannot be exported to NetCDF will be transferred
    as-is to the new database file.

    This function is useful for reducing the size of database files by offloading
    raw data to NetCDF files while preserving all metadata information in a database file.

    Args:
        source_db_path: Path to the source database file
        target_db_path: Path to the target database file that will be created. Error is raised if it already exist.
        export_path: Optional path where NetCDF files should be exported. If None,
            uses the default export path from QCoDeS configuration.

    Returns:
        A dictionary mapping run_id to status ('exported', 'copied_as_is', or 'failed')

    """
    span = trace.get_current_span()
    span.set_attribute("source_db_path", str(source_db_path))
    span.set_attribute("target_db_path", str(target_db_path))
    span.set_attribute("export_path", str(export_path))

    source_db_path = Path(source_db_path)

    if not source_db_path.exists():
        raise FileNotFoundError(f"Source database file not found: {source_db_path}")

    with closing(connect(source_db_path)) as source_con:
        run_ids = sorted(get_runs(source_con))
        _LOG.debug(f"Found {len(run_ids)} datasets to process")
        if not run_ids:
            _LOG.warning(
                f"No datasets found in source database {source_db_path}, nothing to export"
            )
            return {}

    target_db_path = Path(target_db_path)

    if target_db_path.exists():
        raise FileExistsError(
            f"Target database file already exists: {target_db_path}. "
            "Please choose a different path or remove the existing file."
        )

    span.set_attribute("export_path_from_qcodes_config", str(get_data_export_path()))

    if export_path is None:
        export_path = get_data_export_path()
    else:
        export_path = Path(export_path)

    try:
        export_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create export directory {export_path}") from e

    _LOG.info(
        f"Starting NetCDF export process from {source_db_path} to {export_path}, "
        f"and creating metadata-only database file {target_db_path}."
    )

    # Process datasets by experiment to preserve structure
    result_status = {}
    processed_experiments = {}  # Map source exp_id to target exp_id

    with (
        closing(connect(source_db_path)) as source_conn,
        closing(connect(target_db_path)) as target_conn,
    ):
        for run_id in tqdm(run_ids):
            try:
                dataset = load_by_id(run_id, conn=source_conn)
                exp_id = dataset.exp_id

                # Create experiment in target DB if not already done
                if exp_id not in processed_experiments:
                    exp_attrs = get_experiment_attributes_by_exp_id(source_conn, exp_id)

                    with atomic(target_conn) as atomic_target_conn:
                        target_exp_id = _create_exp_if_needed(
                            atomic_target_conn,
                            exp_attrs["name"],
                            exp_attrs["sample_name"],
                            exp_attrs["format_string"],
                            exp_attrs["start_time"],
                            exp_attrs["end_time"],
                        )

                    processed_experiments[exp_id] = target_exp_id
                    _LOG.info(
                        f"Created experiment `{exp_attrs['name']}` on `{exp_attrs['sample_name']}` with ID {target_exp_id} in target database"
                    )
                else:
                    target_exp_id = processed_experiments[exp_id]

                # Try to export dataset to NetCDF and create metadata-only version
                status = _process_single_dataset(
                    dataset, source_conn, target_conn, export_path, target_exp_id
                )
                result_status[run_id] = status

            except Exception:
                _LOG.exception(f"Failed to process dataset {run_id}")
                result_status[run_id] = "failed"

    _LOG.info("Exporting complete.")
    return result_status


@_TRACER.start_as_current_span(f"{__name__}._process_single_dataset")
def _process_single_dataset(
    dataset: DataSetProtocol,
    source_conn: AtomicConnection,
    target_conn: AtomicConnection,
    export_path: Path,
    target_exp_id: int,
) -> Literal["exported", "copied_as_is", "failed"]:
    """
    Export a dataset to NetCDF and add its metadata
    to target database file, or, if it fails, copy directily
    into the target database file.

    Returns:
        Status string indicating what was done with the dataset

    """
    span = trace.get_current_span()
    span.set_attribute("guid", dataset.guid)
    span.set_attribute("given_export_path", str(export_path))

    run_id = dataset.run_id
    span.set_attribute("run_id", run_id)

    netcdf_export_path = None

    existing_netcdf_path = dataset.export_info.export_paths.get("nc")
    span.set_attribute("dataset_netcdf_export_path", str(existing_netcdf_path))
    if existing_netcdf_path is not None:
        existing_path = Path(existing_netcdf_path)
        # Check if the existing export path matches the desired export path
        if existing_path.exists() and existing_path.parent == export_path:
            _LOG.debug(
                f"Dataset {run_id} already exported to NetCDF at {existing_netcdf_path}"
            )
            netcdf_export_path = existing_netcdf_path
        else:
            _LOG.info(
                f"Dataset {run_id} was exported to different location, re-exporting to {export_path}"
            )
    else:
        _LOG.debug(f"Attempting to export dataset {run_id} to NetCDF")

    if netcdf_export_path is None:
        try:
            dataset.export("netcdf", path=export_path)
            netcdf_export_path = dataset.export_info.export_paths.get("nc")
            if netcdf_export_path is None:
                raise RuntimeError(
                    f"Failed to get NetCDF export path for dataset {run_id}. "
                    "Export appears to have succeeded but no path was recorded."
                )
        except Exception:
            _LOG.exception(
                f"Failed to export dataset {run_id} to NetCDF, copying as-is"
            )
            return _copy_dataset_as_is(dataset, source_conn, target_conn, target_exp_id)

    _LOG.debug(f"Dataset {run_id} available as NetCDF at {netcdf_export_path}")

    netcdf_dataset = load_from_netcdf(
        netcdf_export_path, path_to_db=target_conn.path_to_dbfile
    )
    netcdf_dataset.write_metadata_to_db()

    _LOG.info(
        f"Successfully wrote dataset metadata of {run_id} to {target_conn.path_to_dbfile}"
    )

    return "exported"


def _copy_dataset_as_is(
    dataset: DataSetProtocol,
    source_conn: AtomicConnection,
    target_conn: AtomicConnection,
    target_exp_id: int,
) -> Literal["copied_as_is", "failed"]:
    try:
        dataset_obj = DataSet(run_id=dataset.run_id, conn=source_conn)
        with atomic(target_conn) as target_conn_atomic:
            _extract_single_dataset_into_db(
                dataset_obj, target_conn_atomic, target_exp_id
            )
        _LOG.debug(f"Successfully copied dataset {dataset.run_id} as-is")
        return "copied_as_is"
    except Exception:
        _LOG.exception(f"Failed to copy dataset {dataset.run_id} as-is")
        return "failed"
