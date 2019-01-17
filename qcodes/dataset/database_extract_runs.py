from typing import Union, Optional
from warnings import warn
import os

import numpy as np

from qcodes.dataset.data_set import DataSet
from qcodes.dataset.experiment_container import load_or_create_experiment
from qcodes.dataset.sqlite_base import (add_meta_data,
                                        atomic,
                                        connect,
                                        create_run,
                                        format_table_name,
                                        get_db_version_and_newest_available_version,
                                        get_exp_ids_from_run_ids,
                                        get_last_experiment,
                                        get_matching_exp_ids,
                                        get_runid_from_guid,
                                        insert_column,
                                        is_run_id_in_database,
                                        mark_run_complete,
                                        new_experiment,
                                        select_many_where,
                                        ConnectionPlus,
                                        sql_placeholder_string)


def extract_runs_into_db(source_db_path: str,
                         target_db_path: str, *run_ids: int,
                         upgrade_source_db: bool=False,
                         upgrade_target_db: bool=False) -> None:
    """
    Extract a selection of runs into another DB file. All runs must come from
    the same experiment. They will be added to an experiment with the same name
    and sample_name in the target db. If such an experiment does not exist, it
    will be created.

    Args:
        source_db_path: Path to the source DB file
        target_db_path: Path to the target DB file. The target DB file will be
          created if it does not exist.
        run_ids: The run_ids of the runs to copy into the target DB file
        upgrade_source_db: If the source DB is found to be in a version that is
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
    do_runs_exist = is_run_id_in_database(source_conn, run_ids)
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

    exp_attr_names = ['name', 'sample_name', 'start_time', 'end_time',
                      'format_string']

    exp_attr_vals = select_many_where(source_conn,
                                      'experiments',
                                      *exp_attr_names,
                                      where_column='exp_id',
                                      where_value=source_exp_ids[0])

    exp_attrs = dict(zip(exp_attr_names, exp_attr_vals))

    # Massage the target DB file to accomodate the runs
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


def _create_exp_if_needed(target_conn: ConnectionPlus,
                          exp_name: str,
                          sample_name: str,
                          fmt_str: str,
                          start_time: float,
                          end_time: Union[float, None]) -> int:
    """
    Look up in the database whether an experiment already exists and create
    it if it doesn't. Note that experiments do not have GUIDs, so this method
    is not guaranteed to work. Matching names and times is the best we can do.
    """

    matching_exp_ids = get_matching_exp_ids(target_conn,
                                            name=exp_name,
                                            sample_name=sample_name,
                                            format_string=fmt_str,
                                            start_time=start_time,
                                            end_time=end_time)

    if len(matching_exp_ids) > 1:
        exp_id = matching_exp_ids[0]
        warn(f'{len(matching_exp_ids)} experiments found in target DB that '
             'match name, sample_name, fmt_str, start_time, and end_time. '
             f'Inserting into the experiment with exp_id={exp_id}.')
        return exp_id
    if len(matching_exp_ids) == 1:
        return matching_exp_ids[0]
    else:
        lastrowid = new_experiment(target_conn,
                                   name=exp_name,
                                   sample_name=sample_name,
                                   format_string=fmt_str,
                                   start_time=start_time,
                                   end_time=end_time)
        return lastrowid


def _extract_single_dataset_into_db(dataset: DataSet,
                                    target_conn: ConnectionPlus,
                                    target_exp_id: int) -> None:
    """
    NB: This function should only be called from within
    :meth:extract_runs_into_db

    Insert the given dataset into the specified database file as the latest
    run.

    Trying to insert a run already in the DB is a NOOP.

    Args:
        dataset: A dataset representing the run to be copied
        target_conn: connection to the DB. Must be atomically guarded
        target_exp_id: The exp_id of the (target DB) experiment in which to
          insert the run
    """

    if not dataset.completed:
        raise ValueError('Dataset not completed. An incomplete dataset '
                         'can not be copied. The incomplete dataset has '
                         f'GUID: {dataset.guid} and run_id: {dataset.run_id}')

    source_conn = dataset.conn

    run_id = get_runid_from_guid(target_conn, dataset.guid)

    if run_id != -1:
        return

    parspecs = dataset.paramspecs.values()
    metadata = dataset.metadata
    snapshot_raw = dataset.snapshot_raw

    _, target_run_id, target_table_name = create_run(target_conn,
                                                     target_exp_id,
                                                     name=dataset.name,
                                                     guid=dataset.guid,
                                                     parameters=list(parspecs),
                                                     metadata=metadata)
    _populate_results_table(source_conn,
                            target_conn,
                            dataset.table_name,
                            target_table_name)
    mark_run_complete(target_conn, target_run_id)
    _rewrite_timestamps(target_conn,
                        target_run_id,
                        dataset.run_timestamp_raw,
                        dataset.completed_timestamp_raw)

    if snapshot_raw is not None:
        add_meta_data(target_conn, target_run_id, {'snapshot': snapshot_raw})


def _populate_results_table(source_conn: ConnectionPlus,
                            target_conn: ConnectionPlus,
                            source_table_name: str,
                            target_table_name: str) -> None:
    """
    Copy over all the entries of the results table
    """
    get_data_query = f"""
                     SELECT *
                     FROM "{source_table_name}"
                     """

    source_cursor = source_conn.cursor()
    target_cursor = target_conn.cursor()

    for row in source_cursor.execute(get_data_query):
        column_names = ','.join(row.keys()[1:])  # the first key is "id"
        values = tuple(val for val in row[1:])
        value_placeholders = sql_placeholder_string(len(values))
        insert_data_query = f"""
                             INSERT INTO "{target_table_name}"
                             ({column_names})
                             values {value_placeholders}
                             """
        target_cursor.execute(insert_data_query, values)


def _rewrite_timestamps(target_conn: ConnectionPlus, target_run_id: int,
                        correct_run_timestamp: float,
                        correct_completed_timestamp: Optional[float]) -> None:
    """
    Update the timestamp to match the original one
    """
    query = """
            UPDATE runs
            SET run_timestamp = ?
            WHERE run_id = ?
            """
    cursor = target_conn.cursor()
    cursor.execute(query, (correct_run_timestamp, target_run_id))

    query = """
            UPDATE runs
            SET completed_timestamp = ?
            WHERE run_id = ?
            """
    cursor = target_conn.cursor()
    cursor.execute(query, (correct_completed_timestamp, target_run_id))
