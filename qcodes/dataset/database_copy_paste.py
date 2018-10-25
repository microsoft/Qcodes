import numpy as np

from qcodes.dataset.data_set import DataSet
from qcodes.dataset.experiment_container import load_or_create_experiment
from qcodes.dataset.sqlite_base import (atomic,
                                        connect,
                                        get_last_experiment,
                                        insert_column,
                                        SomeConnection)


def sql_placeholder_string(n: int) -> str:
    """
    Return an SQL placeholder string of length n.
    """
    return '(' + ','.join('?'*n) + ')'


def copy_runs_into_db(source_db_path: str,
                      target_db_path: str, *run_ids) -> None:
    """
    Copy a selection of runs into another DB file. All runs must come from the
    same experiment. They will be added to an experiment with the same name
    and sample_name in the target db. If such an experiment does not exist,
    it will be created.

    Args:
        source_db_path: Path to the source DB file
        target_db_path: Path to the target DB file
        run_ids: The run_ids of the runs to copy into the target DB file
    """

    # Validate that all runs are from the same experiment

    sql_placeholders = sql_placeholder_string(len(run_ids))
    exp_id_query = f"""
                    SELECT exp_id
                    FROM runs
                    WHERE run_id IN {sql_placeholders}
                    """
    source_conn = connect(source_db_path)
    cursor = source_conn.cursor()
    cursor.execute(exp_id_query, run_ids)
    rows = cursor.fetchall()
    source_exp_ids = np.unique([exp_id for row in rows for exp_id in row])
    if len(source_exp_ids) != 1:
        raise ValueError('Did not receive runs from a single experiment. '
                         f'Got runs from experiments {source_exp_ids}')

    # Fetch the name and sample name of the runs' experiment

    names_query = """
                  SELECT name, sample_name
                  FROM experiments
                  WHERE exp_id = ?
                  """
    cursor = source_conn.cursor()
    cursor.execute(names_query, (source_exp_ids[0],))
    row = cursor.fetchall()[0]
    (source_exp_name, source_sample_name) = (row['name'], row['sample_name'])

    # Massage the target DB file to accomodate the runs
    # (create new experiment if needed)

    target_conn = connect(target_db_path)

    # this function raises if the target DB file has several experiments
    # matching both the name and sample_name

    load_or_create_experiment(source_exp_name, source_sample_name,
                              conn=target_conn)

    # Finally insert the runs
    for run_id in run_ids:
        copy_single_dataset_into_db(DataSet(run_id=run_id, conn=source_conn),
                                    target_db_path)


def copy_single_dataset_into_db(dataset: DataSet, path_to_db: str) -> None:
    """
    Insert the given dataset into the specified database file as the latest
    run. The database file must exist and its latest experiment must have name
    and sample_name matching that of the dataset's parent experiment

    Args:
        dataset: A dataset representing the run to be copied
        path_to_db: The path to the target DB into which the run should be
          inserted
    """

    if not dataset.completed:
        raise ValueError('Dataset not completed. An incomplete dataset '
                         'can not be copied.')

    source_conn = dataset.conn
    target_conn = connect(path_to_db)

    already_in_query = """
                       SELECT run_id
                       FROM runs
                       WHERE guid = ?
                       """
    cursor = target_conn.cursor()
    cursor.execute(already_in_query, (dataset.guid,))
    res = cursor.fetchall()
    if len(res) > 0:
        return

    exp_id = get_last_experiment(target_conn)

    with atomic(target_conn) as target_conn:
        _copy_runs_table_entries(source_conn,
                                 target_conn,
                                 dataset.run_id,
                                 exp_id)


def _copy_runs_table_entries(source_conn: SomeConnection,
                             target_conn: SomeConnection,
                             source_run_id: int,
                             target_exp_id: int) -> None:
    """
    Copy an entire runs table row from one DB and paste it all
    (expect the primary key) into another DB. The two DBs may be the same.

    This function should be executed with an atomically guarded target_conn
    as a part of a larger atomic transaction
    """
    runs_row_query = """
                    SELECT *
                    FROM runs
                    WHERE run_id = ?
                    """
    cursor = source_conn.cursor()
    cursor.execute(runs_row_query, (source_run_id,))
    source_runs_row = cursor.fetchall()[0]
    source_colnames = set(source_runs_row.keys())

    # There might not be any runs in the target DB, hence we ask PRAGMA
    cursor = target_conn.cursor()
    cursor.execute("PRAGMA table_info(runs)", ())
    tab_info = cursor.fetchall()
    target_colnames = set([r['name'] for r in tab_info])

    for colname in source_colnames.difference(target_colnames):
        insert_column(target_conn, 'runs', colname)

    # the first entry is "run_id"
    sql_colnames = str(tuple(source_runs_row.keys()[1:])).replace("'", "")
    sql_placeholders = sql_placeholder_string(len(source_runs_row.keys())-1)

    sql_insert_values = f"""
                         INSERT INTO runs
                         {sql_colnames}
                         VALUES
                         {sql_placeholders}
                         """
    # the first two entries in source_runs_row are run_id and exp_id
    values = tuple([target_exp_id] + [val for val in source_runs_row[2:]])

    cursor = target_conn.cursor()
    cursor.execute(sql_insert_values, values)