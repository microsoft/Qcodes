from contextlib import contextmanager
import logging
import sqlite3
import time
from numbers import Number
from numpy import ndarray
import numpy as np
import io
from typing import Any, List, Optional, Tuple, Union, Dict

from qcodes.dataset.param_spec import ParamSpec

# represent the type of  data we can/want map to sqlite column
VALUES = List[Union[str, Number, List, ndarray, bool]]

_experiment_table_schema = """
CREATE  TABLE IF NOT EXISTS experiments (
    -- this will autoncrement by default if
    -- no value is specified on insert
    exp_id INTEGER PRIMARY KEY,
    name TEXT,
    sample_name TEXT,
    start_time INTEGER,
    end_time INTEGER,
    -- this is the last counter registered
    -- 1 based
    run_counter INTEGER,
    -- this is the formatter strin used to cosntruct
    -- the run name
    format_string TEXT
-- TODO: maybe I had a good reason for this doulbe primary key
--    PRIMARY KEY (exp_id, start_time, sample_name)
);
"""

_runs_table_schema = """
CREATE TABLE IF NOT EXISTS runs (
    -- this will autoncrement by default if
    -- no value is specified on insert
    run_id INTEGER PRIMARY KEY,
    exp_id INTEGER,
    -- friendly name for the run
    name TEXT,
    -- the name of the table which stores
    -- the actual results
    result_table_name TEXT,
    -- this is the run counter in its experiment 0 based
    result_counter INTEGER,
    ---
    run_timestamp INTEGER,
    completed_timestamp INTEGER,
    is_completed BOOL,
    parameters TEXT,
    -- metadata fields are added dynamically
    FOREIGN KEY(exp_id)
    REFERENCES
        experiments(exp_id)
);
"""


# utility function to allow sqlite/numpy type

def _adapt_array(arr: ndarray) -> sqlite3.Binary:
    """
    See this:
    https://stackoverflow.com/questions/3425320/sqlite3-programmingerror-you-must-not-use-8-bit-bytestrings-unless-you-use-a-te
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def _convert_array(text: bytes) -> ndarray:
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def one(curr: sqlite3.Cursor, column: str) -> Any:
    """Get the value of one column from one row
    Args:
        curr: cursor to operate on
        column: name of the column

    Returns:
        the value
    """
    res = curr.fetchall()
    if len(res) > 1:
        raise RuntimeError("Expected only one row")
    else:
        return res[0][column]


def many(curr: sqlite3.Cursor, *columns: str) -> List[Any]:
    """Get the values of many columns from one row
    Args:
        curr: cursor to operate on
        columns: names of the columns

    Returns:
        list of  values
    """
    res = curr.fetchall()
    if len(res) > 1:
        raise RuntimeError("Expected only one row")
    else:
        return [res[0][c] for c in columns]


def many_many(curr: sqlite3.Cursor, *columns: str) -> List[List[Any]]:
    """Get all values of many columns
    Args:
        curr: cursor to operate on
        columns: names of the columns

    Returns:
        list of lists of values
    """
    res = curr.fetchall()
    results = []
    for r in res:
        results.append([r[c] for c in columns])
    return results


def connect(name: str, debug: bool = False) -> sqlite3.Connection:
    """Connect or create  database. If debug the queries will be echoed back.
    This function takes care of registering the numpy/sqlite type
    converters that we need.


    Args:
        name: name or path to the sqlite file
        debug: whether or not to turn on tracing

    Returns:
        conn: connection object to the database

    """
    # register numpy->binary(TEXT) adapter
    sqlite3.register_adapter(np.ndarray, _adapt_array)
    # register binary(TEXT) -> numpy converter
    # for some reasons mypy complains about this
    sqlite3.register_converter("array", _convert_array)
    conn = sqlite3.connect(name, detect_types=sqlite3.PARSE_DECLTYPES)
    # sqlite3 options
    conn.row_factory = sqlite3.Row

    if debug:
        conn.set_trace_callback(print)
    return conn


def transaction(conn: sqlite3.Connection,
                sql: str, *args: Any) -> sqlite3.Cursor:
    """Perform a transaction.
    The transaction needs to be committed or rolled back.


    Args:
        conn: database connection
        sql: formatted string
        *args: arguments to use for parameter substitution

    Returns:
        sqlite cursor

    """
    c = conn.cursor()
    if len(args) > 0:
        c.execute(sql, args)
    else:
        c.execute(sql)
    return c


def atomicTransaction(conn: sqlite3.Connection,
                      sql: str, *args: Any) -> sqlite3.Cursor:
    """Perform an **atomic** transaction.
    The transaction is committed if there are no exceptions else the
    transaction is rolled back.


    Args:
        conn: database connection
        sql: formatted string
        *args: arguments to use for parameter substitution

    Returns:
        sqlite cursor

    """
    try:
        c = transaction(conn, sql, *args)
    except Exception as e:
        logging.exception("Could not execute transaction, rolling back")
        conn.rollback()
        raise e

    conn.commit()
    return c


@contextmanager
def atomic(conn: sqlite3.Connection):
    """
    Guard a series of transactions as atomic.
    If one fails the transaction is rolled back and no more transactions
    are performed.

    Args:
        - conn: connection to guard
    """
    try:
        yield
    except Exception as e:
        conn.rollback()
        raise RuntimeError("Rolling back due to unhandled exception") from e
    else:
        conn.commit()


def init_db(conn: sqlite3.Connection)->None:
    with atomic(conn):
        transaction(conn, _experiment_table_schema)
        transaction(conn, _runs_table_schema)


def insert_column(conn: sqlite3.Connection, table: str, name: str,
                  type: Optional[str] = None) -> None:
    """Insert new column to a table

    Args:
        conn: database connection
        table: destination for the insertion
        name: column name
        type: sqlite type of the column
    """
    if type:
        transaction(conn,
                    f'ALTER TABLE "{table}" ADD COLUMN "{name}" {type}')
    else:
        transaction(conn,
                    f'ALTER TABLE "{table}" ADD COLUMN "{name}"')


def select_one_where(conn: sqlite3.Connection, table: str, column: str,
                     where_column: str, where_value: Any) -> Any:
    query = f"""
    SELECT {column}
    FROM
        {table}
    WHERE
        {where_column} = ?
    """
    cur = atomicTransaction(conn, query, where_value)
    res = one(cur, column)
    return res


def select_many_where(conn: sqlite3.Connection, table: str, *columns: str,
                      where_column: str, where_value: Any) -> Any:
    _columns = ",".join(columns)
    query = f"""
    SELECT {_columns}
    FROM
        {table}
    WHERE
        {where_column} = ?
    """
    cur = atomicTransaction(conn, query, where_value)
    res = many(cur, *columns)
    return res


def _massage_dict(metadata: Dict[str, Any]) -> Tuple[str, List[Any]]:
    """
    {key:value, key2:value} -> ["key=?, key2=?", [value, value]]
    """
    template = []
    values = []
    for key, value in metadata.items():
        template.append(f"{key} = ?")
        values.append(value)
    return ','.join(template), values


def update_where(conn: sqlite3.Connection, table: str,
                 where_column: str, where_value: Any, **updates) -> None:
    _updates, values = _massage_dict(updates)
    query = f"""
    UPDATE
        '{table}'
    SET
        {_updates}
    WHERE
        {where_column} = ?
    """
    transaction(conn, query, *values, where_value)


def insert_values(conn: sqlite3.Connection,
                  formatted_name: str,
                  columns: List[str],
                  values: VALUES,
                  ) -> int:
    """
    Inserts values for the specified columns.
    Will pad with null if not all parameters are specified.
    NOTE this need to be committed before closing the connection.
    """
    _columns = ",".join(columns)
    _values = ",".join(["?"] * len(columns))
    query = f"""INSERT INTO "{formatted_name}"
        ({_columns})
    VALUES
        ({_values})
    """
    c = transaction(conn, query, *values)
    return c.lastrowid


def insert_many_values(conn: sqlite3.Connection,
                       formatted_name: str,
                       columns: List[str],
                       values: List[VALUES],
                       ) -> int:
    """
    Inserts many values for the specified columns.
    Will pad with null if not all columns are specified.

    NOTE this need to be committed before closing the connection.
    """
    _columns = ",".join(columns)
    # TODO: none of the code below is not form PRADA SS-2017
    # [a, b] -> (?,?), (?,?)
    # [[1,1], [2,2]]
    _values = "(" + ",".join(["?"] * len(columns)) + ")"
    # NOTE: assume that all the values have same length
    _values_x_params = ",".join([_values] * len(values[0]))
    query = f"""INSERT INTO "{formatted_name}"
        ({_columns})
    VALUES
        {_values_x_params}
    """
    # we need to make values a flat list from a list of list
    flattened_values = [item for sublist in values for item in sublist]
    c = transaction(conn, query, *flattened_values)
    return c.lastrowid


def modify_values(conn: sqlite3.Connection,
                  formatted_name: str,
                  index: int,
                  columns: List[str],
                  values: VALUES,
                  ) -> int:
    """
    Modify values for the specified columns.
    If a column is in the table but not in the columns list is
    left untouched.
    If a column is mapped to None, it will be a null value.
    """
    name_val_template = []
    for name, value in zip(columns, values):
        name_val_template.append(f"{name}=?")
    name_val_templates = ",".join(name_val_template)
    query = f"""
    UPDATE "{formatted_name}"
    SET
        {name_val_templates}
    WHERE
        rowid = {index+1}
    """
    c = transaction(conn, query, *values)
    return c.rowcount


def modify_many_values(conn: sqlite3.Connection,
                       formatted_name: str,
                       start_index: int,
                       columns: List[str],
                       values: List[VALUES],
                       ) -> None:
    """
    Modify many values for the specified columns.
    If a column is in the table but not in the column list is
    left untouched.
    If a column is mapped to None, it will be a null value.
    """
    _len = length(conn, formatted_name)
    len_requested = start_index + len(values)
    available = _len - start_index
    if len_requested > _len:
        reason = f""""Modify operation Out of bounds.
        Trying to modify {len(values)} results,
        but therere are only {available} results.
        """
        raise ValueError(reason)
    for value in values:
        modify_values(conn, formatted_name, start_index, columns, value)
        start_index += 1


def length(conn: sqlite3.Connection,
           formatted_name: str
           ) -> int:
    """
    Return the lenght of the table

    Args:
        - conn: the connection to the sqlite database
        - formatted_name: name of the table

    Returns:
        -the lenght of the table
    """
    query = f"select MAX(id) from '{formatted_name}'"
    c = atomicTransaction(conn, query)
    _len = c.fetchall()[0][0]
    if _len is None:
        return 0
    else:
        return _len


def get_data(conn: sqlite3.Connection,
             table_name: str,
             columns: List[str],
             start: int = None,
             end: int = None,
             ) -> List[List[Any]]:
    """
    Get data from the columns of a table.
    Allows to specfiy a range.

    Args:
        conn: database connection
        table_name: name of the table
        columns: list of columns
        start: start of range (1 indedex)
        end: start of range (1 indedex)

    Returns:
        the data requested
    """
    _columns = ",".join(columns)
    if start and end:
        query = f"""
        SELECT {_columns}
        FROM "{table_name}"
        WHERE rowid
            > {start} and
              rowid
            <= {end}
        """
    elif start:
        query = f"""
        SELECT {_columns}
        FROM "{table_name}"
        WHERE rowid
            >= {start}
        """
    elif end:
        query = f"""
        SELECT {_columns}
        FROM "{table_name}"
        WHERE rowid
            <= {end}
        """
    else:
        query = f"""
        SELECT {_columns}
        FROM "{table_name}"
        """
    c = transaction(conn, query)
    res = many_many(c, *columns)
    return res

# Higher level Wrappers


def new_experiment(conn: sqlite3.Connection,
                   name: str,
                   sample_name: str,
                   format_string: Optional[str] = "{}-{}-{}"
                   ) -> int:
    """ Add new experiment to container

    Args:
        conn: database connection
        name: the name of the experiment
        sample_name: the name of the current sample
        format_string: basic format string for table-name
            must contain 3 placeholders.
    Returns:
        id: row-id of the created experiment
    """
    query = """
    INSERT INTO experiments
        (name, sample_name, start_time, format_string, run_counter)
    VALUES
        (?,?,?,?,?)
    """
    curr = atomicTransaction(conn, query, name, sample_name,
                             time.time(), format_string, 0)
    return curr.lastrowid


def mark_run(conn: sqlite3.Connection, run_id: int, complete: bool):
    """ Mark run complete

    Args:
        conn: database connection
        run_id: id of the run to mark complete
        complete: wether the run is completed or not
    """
    query = """
    UPDATE
        runs
    SET
        completed_timestamp=?,
        is_completed=?
    WHERE run_id=?;
    """
    atomicTransaction(conn, query, time.time(), complete, run_id)


def completed(conn: sqlite3.Connection, run_id)->bool:
    """ Check if the run scomplete

    Args:
        conn: database connection
        run_id: id of the run to check
    """
    return bool(select_one_where(conn, "runs", "is_completed",
                                  "run_id", run_id))


def finish_experiment(conn: sqlite3.Connection, exp_id: int):
    """ Finish experiment

    Args:
        conn: database connection
        name: the name of the experiment
    """
    query = """
    UPDATE experiments SET end_time=? WHERE exp_id=?;
    """
    atomicTransaction(conn, query, time.time(), exp_id)


def get_run_counter(conn: sqlite3.Connection, exp_id: int) -> int:
    """ Get the experiment run counter

    Args:
        conn: the connection to the sqlite database
        exp_id: experiment identifier

    Returns:
        the exepriment run counter

    """
    return select_one_where(conn, "experiments", "run_counter",
                            where_column="exp_id",
                            where_value=exp_id)


def get_experiments(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """ Get a list of experiments
     Args:
         conn: database connection

     Returns:
         list of rows
     """
    sql = """
    SELECT * FROM experiments
    """
    c = transaction(conn, sql)
    return c.fetchall()


def get_last_experiment(conn: sqlite3.Connection) -> int:
    """
    Return last started experiment id
    """
    query = "SELECT MAX(exp_id) FROM experiments"
    c = atomicTransaction(conn, query)
    return c.fetchall()[0][0]


def get_runs(conn: sqlite3.Connection)->List[sqlite3.Row]:
    """ Get a list of experiments, if exp_id is specified
    we use it as filter

     Args:
         conn: database connection

     Returns:
         list of rows
     """
    sql = """
    SELECT * FROM runs
    """
    c = transaction(conn, sql)
    return c.fetchall()


def get_last_run(conn: sqlite3.Connection, exp_id: int) -> str:
    query = """
    SELECT run_id, max(run_timestamp), exp_id
    FROM runs
    WHERE exp_id = ?;
    """
    c = transaction(conn, query, exp_id)
    return one(c, 'run_id')


def data_sets(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """ Get a list of datasets
    Args:
        conn: database connection

    Returns:
        list of rows
    """
    sql = """
    SELECT * FROM runs
    """
    c = transaction(conn, sql)
    return c.fetchall()


def _insert_run(conn: sqlite3.Connection, exp_id: int, name: str,
                parameters: Optional[List[ParamSpec]] = None,
                ):
    # get run counter and formatter from experiments
    run_counter, format_string = select_many_where(conn,
                                                    "experiments",
                                                    "run_counter",
                                                    "format_string",
                                                   where_column="exp_id",
                                                   where_value=exp_id)
    run_counter += 1
    formatted_name = format_string.format(name, exp_id, run_counter)
    table = "runs"
    if parameters:
        query = f"""
        INSERT INTO {table}
            (name,exp_id,result_table_name,result_counter,run_timestamp,parameters,is_completed)
        VALUES
            (?,?,?,?,?,?,?)
        """
        curr = transaction(conn, query,
                           name,
                           exp_id,
                           formatted_name,
                           run_counter,
                           time.time(),
                           ",".join([p.name for p in parameters]),
                           False
                           )
    else:
        query = f"""
        INSERT INTO {table}
            (name,exp_id,result_table_name,result_counter,run_timestamp,is_completed)
        VALUES
            (?,?,?,?,?,?)
        """
        curr = transaction(conn, query,
                           name,
                           exp_id,
                           formatted_name,
                           run_counter,
                           time.time(),
                           False
                           )
    return run_counter, formatted_name, curr.lastrowid


def _update_experiment_run_counter(conn: sqlite3.Connection, exp_id: int,
                                   run_counter: int) -> None:
    query = """
    UPDATE experiments
    SET run_counter = ?
    WHERE exp_id = ?
    """
    transaction(conn, query, run_counter, exp_id)


def get_parameters(conn: sqlite3.Connection,
                   formatted_name: str) -> List[ParamSpec]:
    """
    Get the list of param specs for run

    Args:
        - conn: the connection to the sqlite database
        - formatted_name: name of the table

    Returns:
        - A list of param specs for this table
    """
    # TODO: FIX mapping of types
    c = conn.execute(f"""pragma table_info('{formatted_name}')""")
    params: List[ParamSpec] = []
    for row in c.fetchall():
        if row['name'] == 'id':
            continue
        else:
            params.append(ParamSpec(row['name'], row['type']))
    return params


def add_parameter_(conn: sqlite3.Connection,
                   formatted_name: str,
                   *parameter: ParamSpec):
    """ Add parameters to the dataset
    NOTE: two parameters with the same name are not allowed
    Args:
        - conn: the connection to the sqlite database
        - formatted_name: name of the table
        - parameter: the paraemters to add
    """
    p_names = []
    for p in parameter:
        insert_column(conn, formatted_name, p.name, p.type)
        p_names.append(p.name)
    # get old parameters column from run table
    sql = f"""
    SELECT parameters FROM runs
    WHERE result_table_name=?
    """
    c = transaction(conn, sql, formatted_name)
    old_parameters = one(c, 'parameters')
    if old_parameters:
        new_parameters = ",".join([old_parameters] + p_names)
    else:
        new_parameters = ",".join(p_names)
    sql = "UPDATE runs SET parameters=? WHERE result_table_name=?"
    transaction(conn, sql, new_parameters, formatted_name)


def add_parameter(conn: sqlite3.Connection,
                  formatted_name: str,
                  *parameter: ParamSpec):
    """ Add parameters to the dataset
    NOTE: two parameters with the same name are not allowed
    Args:
        - conn: the connection to the sqlite database
        - formatted_name: name of the table
        - parameter: the paraemters to add
    """
    with atomic(conn):
        add_parameter_(conn, formatted_name, *parameter)


def _create_run_table(conn: sqlite3.Connection,
                      formatted_name: str,
                      parameters: Optional[List[ParamSpec]] = None,
                      values: Optional[VALUES] = None
                      ) -> None:
    """Create run table with formatted_name as name

    NOTE this need to be committed before closing the connection.

    Args:
        conn: database connection
        formatted_name: the name of the table to create
    """
    if parameters and values:
        _parameters = ",".join([p.sql_repr() for p in parameters])
        query = f"""
        CREATE TABLE "{formatted_name}" (
            id INTEGER PRIMARY KEY,
            {_parameters}
        );
        """
        transaction(conn, query)
        # now insert values
        insert_values(conn, formatted_name,
                      [p.name for p in parameters], values)
    elif parameters:
        _parameters = ",".join([p.sql_repr() for p in parameters])
        query = f"""
        CREATE TABLE "{formatted_name}" (
            id INTEGER PRIMARY KEY,
            {_parameters}
        );
        """
        transaction(conn, query)
    else:
        query = f"""
        CREATE TABLE "{formatted_name}" (
            id INTEGER PRIMARY KEY
        );
        """
        transaction(conn, query)


def create_run(conn: sqlite3.Connection, exp_id: int, name: str,
               parameters: List[ParamSpec],
               values:  List[Any] = None,
               metadata: Optional[Dict[str, Any]] = None) -> Tuple[int, int, str]:
    """ Create a single run for the experiment.


    This will register the run in the runs table, the counter in the
    experiments table and create a new table with the formatted name.

    Args:
        - conn: the connection to the sqlite database
        - exp_id: the experiment id we want to create the run into
        - name: a friendly name for this run
        - parameters: optional list of parameters this run has
        - values:  optional list of values for the parameters
        - metadata: optional metadata dictionary

    Returns:
        - run_counter: the id of the newly created run (not unique)
        - run_id: the row id of the newly created run
        - formatted_name: the name of the newly created table
    """
    with atomic(conn):
        run_counter, formatted_name, run_id = _insert_run(conn,
                                                          exp_id,
                                                          name,
                                                          parameters)
        if metadata:
            add_meta_data(conn, run_id, metadata)
        _update_experiment_run_counter(conn, exp_id, run_counter)
        _create_run_table(conn, formatted_name, parameters, values)
    return run_counter, run_id, formatted_name


def get_metadata(conn: sqlite3.Connection, tag: str, table_name: str):
    """ Get metadata under the tag from table
    """
    return select_one_where(conn, "runs", tag,
                             "result_table_name", table_name)


def insert_meta_data(conn: sqlite3.Connection, row_id: int, table_name: str,
                     metadata: Dict[str, Any]) -> None:
    """
    Insert new metadata column and add values

    Args:
        - conn: the connection to the sqlite database
        - row_id: the row to add the metadata at
        - table_name: the table to add to, defaults to runs
        - metadata: the metadata to add
    """
    for key, value in metadata.items():
        insert_column(conn, table_name, key)
    update_meta_data(conn, row_id, table_name, metadata)


def update_meta_data(conn: sqlite3.Connection, row_id: int, table_name: str,
                     metadata: Dict[str, Any]) -> None:
    """
    Updates metadata (they must exist already)

    Args:
        - conn: the connection to the sqlite database
        - row_id: the row to add the metadata at
        - table_name: the table to add to, defaults to runs
        - metadata: the metadata to add
    """
    update_where(conn, table_name, 'rowid', row_id, **metadata)


def add_meta_data(conn: sqlite3.Connection,
                  row_id: int,
                  metadata: Dict[str, Any],
                  table_name: Optional[str] = "runs") -> None:
    """
    Add medata data (updates if exists, create otherwise).

    Args:
        - conn: the connection to the sqlite database
        - row_id: the row to add the metadata at
        - metadata: the metadata to add
        - table_name: the table to add to, defaults to runs
    """
    try:
        insert_meta_data(conn, row_id, table_name, metadata)
    except sqlite3.OperationalError as e:
        # this means that the column already exists
        # so just insert the new value
        if str(e).startswith("duplicate"):
            update_meta_data(conn, row_id, table_name, metadata)
        else:
            raise e
