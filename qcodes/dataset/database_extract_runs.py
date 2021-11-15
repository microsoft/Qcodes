import os
from warnings import warn

import numpy as np

from qcodes.dataset.data_set import DataSet
from qcodes.dataset.dataset_helpers import _add_run_to_runs_table
from qcodes.dataset.experiment_container import _create_exp_if_needed
from qcodes.dataset.sqlite.connection import ConnectionPlus, atomic
from qcodes.dataset.sqlite.database import (
    connect,
    get_db_version_and_newest_available_version,
)
from qcodes.dataset.sqlite.queries import (
    _populate_results_table,
    get_exp_ids_from_run_ids,
    get_experiment_attributes_by_exp_id,
    get_runid_from_guid,
    is_run_id_in_database,
)


def extract_runs_into_db(source_db_path: str,
                         target_db_path: str, *run_ids: int,
                         upgrade_source_db: bool = False,
                         upgrade_target_db: bool = False) -> None:
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
        warn(f'Source DB version is {s_v}, but this function needs it to be'
             f' in version {new_v}. Run this function again with '
             'upgrade_source_db=True to auto-upgrade the source DB file.')
        return

    if os.path.exists(target_db_path):
        (t_v, new_v) = get_db_version_and_newest_available_version(target_db_path)
        if t_v < new_v and not upgrade_target_db:
            warn(f'Target DB version is {t_v}, but this function needs it to '
                 f'be in version {new_v}. Run this function again with '
                 'upgrade_target_db=True to auto-upgrade the target DB file.')
            return

    source_conn = connect(source_db_path)

    # Validate that all runs are in the source database
    do_runs_exist = is_run_id_in_database(source_conn, *run_ids)
    if False in do_runs_exist.values():
        source_conn.close()
        non_existing_ids = [rid for rid in run_ids if not do_runs_exist[rid]]
        err_mssg = ("Error: not all run_ids exist in the source database. "
                    "The following run(s) is/are not present: "
                    f"{non_existing_ids}")
        raise ValueError(err_mssg)

    # Validate that all runs are from the same experiment

    source_exp_ids = np.unique(get_exp_ids_from_run_ids(source_conn, run_ids))
    if len(source_exp_ids) != 1:
        source_conn.close()
        raise ValueError('Did not receive runs from a single experiment. '
                         f'Got runs from experiments {source_exp_ids}')

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

            target_exp_id = _create_exp_if_needed(target_conn,
                                                  exp_attrs['name'],
                                                  exp_attrs['sample_name'],
                                                  exp_attrs['format_string'],
                                                  exp_attrs['start_time'],
                                                  exp_attrs['end_time'])

            # Finally insert the runs
            for run_id in run_ids:
                _extract_single_dataset_into_db(DataSet(run_id=run_id,
                                                        conn=source_conn),
                                                target_conn,
                                                target_exp_id)
    finally:
        source_conn.close()
        target_conn.close()


def _extract_single_dataset_into_db(dataset: DataSet,
                                    target_conn: ConnectionPlus,
                                    target_exp_id: int) -> None:
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
        raise ValueError('Dataset not completed. An incomplete dataset '
                         'can not be copied. The incomplete dataset has '
                         f'GUID: {dataset.guid} and run_id: {dataset.run_id}')

    source_conn = dataset.conn

    run_id = get_runid_from_guid(target_conn, dataset.guid)

    if run_id is not None:
        return

    target_table_name = _add_run_to_runs_table(dataset, target_conn, target_exp_id)
    assert target_table_name is not None
    _populate_results_table(
        source_conn, target_conn, dataset.table_name, target_table_name
    )
