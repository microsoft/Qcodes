from typing import Union
from warnings import warn

import numpy as np

from qcodes.dataset.data_set import DataSet
from qcodes.dataset.experiment_container import load_or_create_experiment
from qcodes.dataset.sqlite_base import (atomic,
                                        connect,
                                        format_table_name,
                                        get_exp_ids_from_run_ids,
                                        get_last_experiment,
                                        get_matching_exp_ids,
                                        get_runid_from_guid,
                                        insert_column,
                                        is_run_id_in_database,
                                        new_experiment,
                                        select_many_where,
                                        SomeConnection,
                                        sql_placeholder_string)


def extract_runs_into_db(source_db_path: str,
                         target_db_path: str, *run_ids: int) -> None:
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
    """
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


def _create_exp_if_needed(target_conn: SomeConnection,
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
                                    target_conn: SomeConnection,
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
                         'can not be copied.')

    source_conn = dataset.conn

    run_id = get_runid_from_guid(target_conn, dataset.guid)

    if run_id != -1:
        return

    parspecs = dataset.paramspecs.values()
    data_column_names_and_types = ",".join([p.sql_repr() for p in parspecs])

    target_run_id = _copy_runs_table_entries(source_conn,
                                             target_conn,
                                             dataset.run_id,
                                             target_exp_id)
    _update_run_counter(target_conn, target_exp_id)
    _copy_layouts_and_dependencies(source_conn,
                                   target_conn,
                                   dataset.run_id,
                                   target_run_id)
    target_table_name = _copy_results_table(source_conn,
                                            target_conn,
                                            dataset.run_id,
                                            target_run_id,
                                            data_column_names_and_types)
    _update_result_table_name(target_conn, target_table_name, target_run_id)


def _copy_runs_table_entries(source_conn: SomeConnection,
                             target_conn: SomeConnection,
                             source_run_id: int,
                             target_exp_id: int) -> int:
    """
    Copy an entire runs table row from one DB and paste it all
    (expect the primary key) into another DB. The two DBs may not be the same.
    Note that this function does not create a new results table

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

    return cursor.lastrowid


def _update_run_counter(target_conn: SomeConnection, target_exp_id) -> None:
    """
    Update the run_counter in the target DB experiments table
    """
    update_sql = """
                 UPDATE experiments
                 SET run_counter = run_counter + 1
                 WHERE exp_id = ?
                 """
    cursor = target_conn.cursor()
    cursor.execute(update_sql, (target_exp_id,))


def _copy_layouts_and_dependencies(source_conn: SomeConnection,
                                   target_conn: SomeConnection,
                                   source_run_id: int,
                                   target_run_id: int) -> None:
    """
    Copy over the layouts and dependencies tables. Note that the layout_ids
    are not preserved in the target DB, but of course their relationships are
    (e.g. layout_id 10 that depends on layout_id 9 might be inserted as
    layout_id 2 that depends on layout_id 1)
    """
    layout_query = """
                   SELECT layout_id, run_id, "parameter", label, unit, inferred_from
                   FROM layouts
                   WHERE run_id = ?
                   """
    cursor = source_conn.cursor()
    cursor.execute(layout_query, (source_run_id,))
    rows = cursor.fetchall()

    layout_insert = """
                    INSERT INTO layouts
                    (run_id, parameter, label, unit, inferred_from)
                    VALUES (?,?,?,?,?)
                    """

    colnames = ('run_id', 'parameter', 'label', 'unit', 'inferred_from')
    cursor = target_conn.cursor()
    source_layout_ids = []
    target_layout_ids = []
    for row in rows:
        values = ((target_run_id,) +
                  tuple(row[colname] for colname in colnames[1:]))
        cursor.execute(layout_insert, values)
        source_layout_ids.append(row['layout_id'])
        target_layout_ids.append(cursor.lastrowid)

    # for the dependencies, we need a map from source layout_id to
    # target layout_id
    layout_id_map = dict(zip(source_layout_ids, target_layout_ids))

    placeholders = sql_placeholder_string(len(source_layout_ids))

    deps_query = f"""
                 SELECT dependent, independent, axis_num
                 FROM dependencies
                 WHERE dependent IN {placeholders}
                 OR independent IN {placeholders}
                 """

    cursor = source_conn.cursor()
    cursor.execute(deps_query, tuple(source_layout_ids*2))
    rows = cursor.fetchall()

    deps_insert = """
                  INSERT INTO dependencies
                  (dependent, independent, axis_num)
                  VALUES (?,?,?)
                  """
    cursor = target_conn.cursor()

    for row in rows:
        values = (layout_id_map[row['dependent']],
                  layout_id_map[row['independent']],
                  row['axis_num'])
        cursor.execute(deps_insert, values)


def _copy_results_table(source_conn: SomeConnection,
                        target_conn: SomeConnection,
                        source_run_id: int,
                        target_run_id: int,
                        column_names_and_types: str) -> str:
    """
    Copy the contents of the results table. Creates a new results_table with
    a name appropriate for the target DB and updates the rows of that table

    Returns the name of the new results table
    """
    table_name_query = """
                       SELECT result_table_name, name
                       FROM runs
                       WHERE run_id = ?
                       """
    cursor = source_conn.cursor()
    cursor.execute(table_name_query, (source_run_id,))
    row = cursor.fetchall()[0]
    table_name = row['result_table_name']
    run_name = row['name']

    format_string_query = """
                          SELECT format_string, run_counter, experiments.exp_id
                          FROM experiments
                          INNER JOIN runs
                          ON runs.exp_id = experiments.exp_id
                          WHERE runs.run_id = ?
                          """
    cursor = target_conn.cursor()
    cursor.execute(format_string_query, (target_run_id,))
    row = cursor.fetchall()[0]
    format_string = row['format_string']
    run_counter = row['run_counter']
    exp_id = int(row['exp_id'])

    get_data_query = f"""
                     SELECT *
                     FROM "{table_name}"
                     """
    cursor = source_conn.cursor()
    cursor.execute(get_data_query)
    data_rows = cursor.fetchall()
    if len(data_rows) > 0:
        data_columns = data_rows[0].keys()
        data_columns.remove('id')
    else:
        data_columns = []

    target_table_name = format_table_name(format_string,
                                          run_name,
                                          exp_id,
                                          run_counter)

    if column_names_and_types != '':
        make_table = f"""
                    CREATE TABLE "{target_table_name}" (
                        id INTEGER PRIMARY KEY,
                        {column_names_and_types}
                    )
                    """
    else:
        make_table = f"""
            CREATE TABLE "{target_table_name}" (
                id INTEGER PRIMARY KEY
            )
            """

    cursor = target_conn.cursor()
    cursor.execute(make_table)

    # according to one of our reports, multiple single-row inserts
    # are okay if there's only one commit

    column_names = ','.join(data_columns)
    value_placeholders = sql_placeholder_string(len(data_columns))
    insert_data = f"""
                   INSERT INTO "{target_table_name}"
                   ({column_names})
                   values {value_placeholders}
                   """

    for row in data_rows:
        # the first row entry is the ID, which is automatically inserted
        cursor.execute(insert_data, tuple(v for v in row[1:]))

    return target_table_name


def _update_result_table_name(target_conn: SomeConnection,
                              target_table_name: str,
                              target_run_id: int) -> None:
    sql = """
          UPDATE runs
          SET result_table_name = ?
          WHERE run_id = ?
          """
    cursor = target_conn.cursor()
    cursor.execute(sql, (target_table_name, target_run_id))
