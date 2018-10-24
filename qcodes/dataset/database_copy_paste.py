from qcodes.dataset.data_set import DataSet
from qcodes.dataset.sqlite_base import (atomic,
                                        connect,
                                        get_last_experiment,
                                        insert_column,
                                        SomeConnection)


def copy_dataset_into_db(dataset: DataSet, path_to_db: str) -> None:
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
    sql_placeholders = '(' + ','.join('?'*len(sql_colnames)) + ')'

    sql_insert_values = f"""
                         INSERT INTO runs
                         {sql_colnames}
                         VALUES
                         {sql_placeholders}
                         """
    # the first two entries in source_runs_row are run_id and exp_id
    values = tuple([exp_id] + [val for val in source_runs_row[2:]])

    cursor = target_conn.cursor()
    cursor.execute(sql_insert_values, values)