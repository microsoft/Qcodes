import sys
from contextlib import contextmanager
import logging
import sqlite3
import time
import io
from typing import Any, List, Optional, Tuple, Union, Dict, cast, Callable
import itertools

from numbers import Number
from numpy import ndarray
import numpy as np
from distutils.version import LooseVersion
import wrapt

import qcodes as qc
import unicodedata
from qcodes.dataset.param_spec import ParamSpec
from qcodes.dataset.guids import generate_guid, parse_guid

log = logging.getLogger(__name__)

# represent the type of  data we can/want map to sqlite column
VALUE = Union[str, Number, List, ndarray, bool]
VALUES = List[VALUE]

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
    -- this will autoincrement by default if
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

_layout_table_schema = """
CREATE TABLE IF NOT EXISTS layouts (
    layout_id INTEGER PRIMARY KEY,
    run_id INTEGER,
    -- name matching column name in result table
    parameter TEXT,
    label TEXT,
    unit TEXT,
    inferred_from TEXT,
    FOREIGN KEY(run_id)
    REFERENCES
        runs(run_id)
);
"""

_dependencies_table_schema = """
CREATE TABLE IF NOT EXISTS dependencies (
    dependent INTEGER,
    independent INTEGER,
    axis_num INTEGER
);
"""

_unicode_categories = ('Lu', 'Ll', 'Lt', 'Lm', 'Lo', 'Nd', 'Pc', 'Pd', 'Zs')
# utility function to allow sqlite/numpy type


class ConnectionPlus(wrapt.ObjectProxy):
    """
    A class to extend the sqlite3.Connection object with a single extra
    attribute. Since sqlite3.Connection has no __dict__, we can not directly
    add the attribute to a normal instance.

    The extra attribute is a bool describing whether the connection is
    currently in the middle of an atomic block of transactions, thus allowing
    us to nest atomic context managers
    """
    atomic_in_progress: bool = True


SomeConnection = Union[sqlite3.Connection, ConnectionPlus]


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


this_session_default_encoding = sys.getdefaultencoding()


def _convert_numeric(value: bytes) -> Union[float, int, str]:
    """
    This is a converter for sqlite3 'numeric' type class.

    This converter is capable of deducting whether a number is a float or an
    int.

    Note sqlite3 allows to save data to columns even if their type is not
    compatible with the table type class (for example, it is possible to save
    integers into 'text' columns). Due to this fact, and for the reasons of
    flexibility, the numeric converter is also made capable of handling
    strings. An obvious exception to this is 'nan' (case insensitive) which
    gets converted to `np.nan`.
    """
    try:
        # First, try to convert bytes to float
        numeric = float(value)
    except ValueError as e:
        # If an exception has been raised, we first need to find out
        # if the reason was the conversion to float, and, if so, we are sure
        # that we need to return a string
        if "could not convert string to float" in str(e):
            return str(value, encoding=this_session_default_encoding)
        else:
            # otherwise, the exception is forwarded up the stack
            raise e

    # If that worked, e.g. did not raise an exception, then we check if the
    # outcome is 'nan'
    if np.isnan(numeric):
        return numeric

    # If it is not 'nan', then we need to see if the value is really an
    # integer or with floating point digits
    numeric_int = int(numeric)
    if numeric != numeric_int:
        return numeric
    else:
        return numeric_int


def _adapt_float(fl: float) -> Union[float, str]:
    if np.isnan(fl):
        return "nan"
    return float(fl)


def one(curr: sqlite3.Cursor, column: Union[int, str]) -> Any:
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
    elif len(res) == 0:
        raise RuntimeError("Expected one row")
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


def connect(name: str, debug: bool = False,
            version: int=-1) -> sqlite3.Connection:
    """
    Connect or create  database. If debug the queries will be echoed back.
    This function takes care of registering the numpy/sqlite type
    converters that we need.


    Args:
        name: name or path to the sqlite file
        debug: whether or not to turn on tracing
        version: which version to create. We count from 0. -1 means 'latest'
          Should always be left at -1 except when testing.

    Returns:
        conn: connection object to the database

    """
    # register numpy->binary(TEXT) adapter
    # the typing here is ignored due to what we think is a flaw in typeshed
    # see https://github.com/python/typeshed/issues/2429
    sqlite3.register_adapter(np.ndarray, _adapt_array)  # type: ignore
    # register binary(TEXT) -> numpy converter
    # for some reasons mypy complains about this
    sqlite3.register_converter("array", _convert_array)
    conn = sqlite3.connect(name, detect_types=sqlite3.PARSE_DECLTYPES)
    # sqlite3 options
    conn.row_factory = sqlite3.Row

    # Make sure numpy ints and floats types are inserted properly
    for numpy_int in [
        np.int, np.int8, np.int16, np.int32, np.int64,
        np.uint, np.uint8, np.uint16, np.uint32, np.uint64
    ]:
        sqlite3.register_adapter(numpy_int, int)

    sqlite3.register_converter("numeric", _convert_numeric)

    for numpy_float in [np.float, np.float16, np.float32, np.float64]:
        sqlite3.register_adapter(numpy_float, _adapt_float)

    if debug:
        conn.set_trace_callback(print)

    init_db(conn)
    perform_db_upgrade(conn, version=version)
    return conn


def perform_db_upgrade(conn: SomeConnection, version: int=-1) -> None:
    """
    This is intended to perform all upgrades as needed to bring the
    db from version 0 to the most current version (or the version specified).
    All the perform_db_upgrade_X_to_Y functions must raise if they cannot
    upgrade and be a NOOP if the current version is higher than their target.

    Args:
        version: Which version to upgrade to. We count from 0. -1 means
          'newest version'
    """

    upgrade_actions = [perform_db_upgrade_0_to_1]
    newest_version = len(upgrade_actions)
    version = newest_version if version == -1 else version

    current_version = get_user_version(conn)
    if current_version < newest_version:
        log.info("Commencing database upgrade")
        for action in upgrade_actions[:version]:
            action(conn)


def perform_db_upgrade_0_to_1(conn: SomeConnection) -> None:
    """
    Perform the upgrade from version 0 to version 1

    Returns:
        A bool indicating whether everything went well. The next upgrade in
          the upgrade chain should run conditioned in this bool
    """
    log.info('Starting database upgrade version 0 -> 1')

    start_version = get_user_version(conn)
    if start_version != 0:
        log.warn('Can not upgrade, current database version is '
                 f'{start_version}, aborting.')
        return

    sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
    cur = atomic_transaction(conn, sql)
    n_run_tables = len(cur.fetchall())

    if n_run_tables == 1:
        with atomic(conn) as conn:
            sql = "ALTER TABLE runs ADD COLUMN guid TEXT"
            transaction(conn, sql)
            # now assign GUIDs to existing runs
            cur = transaction(conn, 'SELECT run_id FROM runs')
            run_ids = [r[0] for r in many_many(cur, 'run_id')]

            for run_id in run_ids:
                query = f"""
                        SELECT run_timestamp
                        FROM runs
                        WHERE run_id == {run_id}
                        """
                cur = transaction(conn, query)
                timestamp = one(cur, 'run_timestamp')
                timeint = int(np.round(timestamp*1000))
                sql = f"""
                        UPDATE runs
                        SET guid = ?
                        where run_id == {run_id}
                        """
                sampleint = 3736062718  # 'deafcafe'
                cur.execute(sql, (generate_guid(timeint=timeint,
                                                sampleint=sampleint),))
    else:
        raise RuntimeError(f"found {n_run_tables} runs tables expected 1")

    log.info('Succesfully upgraded database version 0 -> 1.')
    set_user_version(conn, 1)


def transaction(conn: SomeConnection,
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


def atomic_transaction(conn: SomeConnection,
                       sql: str, *args: Any) -> sqlite3.Cursor:
    """Perform an **atomic** transaction.
    The transaction is committed if there are no exceptions else the
    transaction is rolled back.
    NB: 'BEGIN' is by default only inserted before INSERT/UPDATE/DELETE/REPLACE
    but we want to guard any transaction that modifies the database (e.g. also
    ALTER). 'BEGIN' marks a place to commit from/roll back to

    Args:
        conn: database connection
        sql: formatted string
        *args: arguments to use for parameter substitution

    Returns:
        sqlite cursor

    """
    with atomic(conn) as conn:
        c = transaction(conn, sql, *args)
    return c


@contextmanager
def atomic(conn: SomeConnection):
    """
    Guard a series of transactions as atomic.
    If one fails the transaction is rolled back and no more transactions
    are performed.
    NB: 'BEGIN' is by default only inserted before INSERT/UPDATE/DELETE/REPLACE
    but we want to guard any transaction that modifies the database (e.g. also
    ALTER)

    Args:
        - conn: connection to guard
    """

    if not(hasattr(conn, 'atomic_in_progress')):
        conn = ConnectionPlus(conn)
        conn.atomic_in_progress = False
    else:
        conn = cast(ConnectionPlus, conn)

    is_outmost = not(conn.atomic_in_progress)
    conn.atomic_in_progress = True

    if conn.in_transaction and is_outmost:
        raise RuntimeError('SQLite connection has uncommited transactions. '
                           'Please commit those before starting an atomic '
                           'transaction.')

    try:
        if is_outmost:
            old_level = conn.isolation_level
            conn.isolation_level = None
            conn.cursor().execute('BEGIN')
        yield conn
    except Exception as e:
        conn.rollback()
        log.exception("Rolling back due to unhandled exception")
        raise RuntimeError("Rolling back due to unhandled exception") from e
    else:
        if is_outmost:
            conn.commit()
    finally:
        if is_outmost:
            conn.isolation_level = old_level


def init_db(conn: SomeConnection)->None:
    with atomic(conn) as conn:
        transaction(conn, _experiment_table_schema)
        transaction(conn, _runs_table_schema)
        transaction(conn, _layout_table_schema)
        transaction(conn, _dependencies_table_schema)


def insert_column(conn: SomeConnection, table: str, name: str,
                  paramtype: Optional[str] = None) -> None:
    """Insert new column to a table

    Args:
        conn: database connection
        table: destination for the insertion
        name: column name
        type: sqlite type of the column
    """
    # first check that the column is not already there
    # and do nothing if it is
    query = f'PRAGMA TABLE_INFO("{table}");'
    cur = atomic_transaction(conn, query)
    columns = many_many(cur, "name")
    if name in [col[0] for col in columns]:
        return

    with atomic(conn) as conn:
        if paramtype:
            transaction(conn,
                        f'ALTER TABLE "{table}" ADD COLUMN "{name}" '
                        f'{paramtype}')
        else:
            transaction(conn,
                        f'ALTER TABLE "{table}" ADD COLUMN "{name}"')


def select_one_where(conn: SomeConnection, table: str, column: str,
                     where_column: str, where_value: Any) -> Any:
    query = f"""
    SELECT {column}
    FROM
        {table}
    WHERE
        {where_column} = ?
    """
    cur = atomic_transaction(conn, query, where_value)
    res = one(cur, column)
    return res


def select_many_where(conn: SomeConnection, table: str, *columns: str,
                      where_column: str, where_value: Any) -> Any:
    _columns = ",".join(columns)
    query = f"""
    SELECT {_columns}
    FROM
        {table}
    WHERE
        {where_column} = ?
    """
    cur = atomic_transaction(conn, query, where_value)
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


def update_where(conn: SomeConnection, table: str,
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
    atomic_transaction(conn, query, *values, where_value)


def insert_values(conn: SomeConnection,
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

    c = atomic_transaction(conn, query, *values)
    return c.lastrowid


def insert_many_values(conn: SomeConnection,
                       formatted_name: str,
                       columns: List[str],
                       values: List[VALUES],
                       ) -> int:
    """
    Inserts many values for the specified columns.

    Example input:
    columns: ['xparam', 'yparam']
    values: [[x1, y1], [x2, y2], [x3, y3]]

    NOTE this need to be committed before closing the connection.
    """
    # We demand that all values have the same length
    lengths = [len(val) for val in values]
    if len(np.unique(lengths)) > 1:
        raise ValueError('Wrong input format for values. Must specify the '
                         'same number of values for all columns. Received'
                         f' lengths {lengths}.')
    no_of_rows = len(lengths)
    no_of_columns = lengths[0]

    # The TOTAL number of inserted values in one query
    # must be less than the SQLITE_MAX_VARIABLE_NUMBER

    # Version check cf.
    # "https://stackoverflow.com/questions/9527851/sqlite-error-
    #  too-many-terms-in-compound-select"
    version = qc.SQLiteSettings.settings['VERSION']

    # According to the SQLite changelog, the version number
    # to check against below
    # ought to be 3.7.11, but that fails on Travis
    if LooseVersion(str(version)) <= LooseVersion('3.8.2'):
        max_var = qc.SQLiteSettings.limits['MAX_COMPOUND_SELECT']
    else:
        max_var = qc.SQLiteSettings.limits['MAX_VARIABLE_NUMBER']
    rows_per_transaction = int(int(max_var)/no_of_columns)

    _columns = ",".join(columns)
    _values = "(" + ",".join(["?"] * len(values[0])) + ")"

    a, b = divmod(no_of_rows, rows_per_transaction)
    chunks = a*[rows_per_transaction] + [b]
    if chunks[-1] == 0:
        chunks.pop()

    start = 0
    stop = 0

    with atomic(conn) as conn:
        for ii, chunk in enumerate(chunks):
            _values_x_params = ",".join([_values] * chunk)

            query = f"""INSERT INTO "{formatted_name}"
                        ({_columns})
                        VALUES
                        {_values_x_params}
                     """
            stop += chunk
            # we need to make values a flat list from a list of list
            flattened_values = list(
                itertools.chain.from_iterable(values[start:stop]))

            c = transaction(conn, query, *flattened_values)

            if ii == 0:
                return_value = c.lastrowid
            start += chunk

    return return_value


def modify_values(conn: SomeConnection,
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
    for name in columns:
        name_val_template.append(f"{name}=?")
    name_val_templates = ",".join(name_val_template)
    query = f"""
    UPDATE "{formatted_name}"
    SET
        {name_val_templates}
    WHERE
        rowid = {index+1}
    """
    c = atomic_transaction(conn, query, *values)
    return c.rowcount


def modify_many_values(conn: SomeConnection,
                       formatted_name: str,
                       start_index: int,
                       columns: List[str],
                       list_of_values: List[VALUES],
                       ) -> None:
    """
    Modify many values for the specified columns.
    If a column is in the table but not in the column list is
    left untouched.
    If a column is mapped to None, it will be a null value.
    """
    _len = length(conn, formatted_name)
    len_requested = start_index + len(list_of_values[0])
    available = _len - start_index
    if len_requested > _len:
        reason = f""""Modify operation Out of bounds.
        Trying to modify {len(list_of_values)} results,
        but therere are only {available} results.
        """
        raise ValueError(reason)
    for values in list_of_values:
        modify_values(conn, formatted_name, start_index, columns, values)
        start_index += 1


def length(conn: SomeConnection,
           formatted_name: str
           ) -> int:
    """
    Return the lenght of the table

    Args:
        conn: the connection to the sqlite database
        formatted_name: name of the table

    Returns:
        the lenght of the table
    """
    query = f"select MAX(id) from '{formatted_name}'"
    c = atomic_transaction(conn, query)
    _len = c.fetchall()[0][0]
    if _len is None:
        return 0
    else:
        return _len


def get_data(conn: SomeConnection,
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
    c = atomic_transaction(conn, query)
    res = many_many(c, *columns)

    return res


def get_values(conn: SomeConnection,
               table_name: str,
               param_name: str) -> List[List[Any]]:
    """
    Get the not-null values of a parameter

    Args:
        conn: Connection to the database
        table_name: Name of the table that holds the data
        param_name: Name of the parameter to get the setpoints of

    Returns:
        The values
    """
    sql = f"""
    SELECT {param_name} FROM "{table_name}"
    WHERE {param_name} IS NOT NULL
    """
    c = atomic_transaction(conn, sql)
    res = many_many(c, param_name)

    return res


def get_setpoints(conn: SomeConnection,
                  table_name: str,
                  param_name: str) -> Dict[str, List[List[Any]]]:
    """
    Get the setpoints for a given dependent parameter

    Args:
        conn: Connection to the database
        table_name: Name of the table that holds the data
        param_name: Name of the parameter to get the setpoints of

    Returns:
        A list of returned setpoint values. Each setpoint return value
        is a list of lists of Any. The first list is a list of run points,
        the second list is a list of parameter values.
    """
    # TODO: We do this in no less than 5 table lookups, surely
    # this number can be reduced

    # get run_id
    sql = """
    SELECT run_id FROM runs WHERE result_table_name = ?
    """
    c = atomic_transaction(conn, sql, table_name)
    run_id = one(c, 'run_id')

    # get the parameter layout id
    sql = """
    SELECT layout_id FROM layouts
    WHERE parameter = ?
    and run_id = ?
    """
    c = atomic_transaction(conn, sql, param_name, run_id)
    layout_id = one(c, 'layout_id')

    # get the setpoint layout ids
    sql = """
    SELECT independent FROM dependencies
    WHERE dependent = ?
    """
    c = atomic_transaction(conn, sql, layout_id)
    indeps = many_many(c, 'independent')
    indeps = [idp[0] for idp in indeps]

    # get the setpoint names
    sql = f"""
    SELECT parameter FROM layouts WHERE layout_id
    IN {str(indeps).replace('[', '(').replace(']', ')')}
    """
    c = atomic_transaction(conn, sql)
    setpoint_names_temp = many_many(c, 'parameter')
    setpoint_names = [spn[0] for spn in setpoint_names_temp]
    setpoint_names = cast(List[str], setpoint_names)

    # get the actual setpoint data
    output: Dict[str, List[List[Any]]] = {}
    for sp_name in setpoint_names:
        sql = f"""
        SELECT {sp_name}
        FROM "{table_name}"
        WHERE {param_name} IS NOT NULL
        """
        c = atomic_transaction(conn, sql)
        sps = many_many(c, sp_name)
        output[sp_name] = sps

    return output


def get_layout(conn: SomeConnection,
               layout_id) -> Dict[str, str]:
    """
    Get the layout of a single parameter for plotting it

    Args:
        conn: The database connection
        run_id: The run_id as in the runs table

    Returns:
        A dict with name, label, and unit
    """
    sql = """
    SELECT parameter, label, unit FROM layouts WHERE layout_id=?
    """
    c = atomic_transaction(conn, sql, layout_id)
    t_res = many(c, 'parameter', 'label', 'unit')
    res = dict(zip(['name', 'label', 'unit'], t_res))
    return res


def get_layout_id(conn: SomeConnection,
                  parameter: Union[ParamSpec, str],
                  run_id: int) -> int:
    """
    Get the layout id of a parameter in a given run

    Args:
        conn: The database connection
        parameter: A ParamSpec or the name of the parameter
        run_id: The run_id of the run in question
    """
    # get the parameter layout id
    sql = """
    SELECT layout_id FROM layouts
    WHERE parameter = ?
    and run_id = ?
    """

    if isinstance(parameter, ParamSpec):
        name = parameter.name
    elif isinstance(parameter, str):
        name = parameter
    else:
        raise ValueError('Wrong parameter type, must be ParamSpec or str, '
                         f'received {type(parameter)}.')

    c = atomic_transaction(conn, sql, name, run_id)
    res = one(c, 'layout_id')

    return res


def get_dependents(conn: SomeConnection,
                   run_id: int) -> List[int]:
    """
    Get dependent layout_ids for a certain run_id, i.e. the layout_ids of all
    the dependent variables
    """
    sql = """
    SELECT layout_id FROM layouts
    WHERE run_id=? and layout_id in (SELECT dependent FROM dependencies)
    """
    c = atomic_transaction(conn, sql, run_id)
    res = [d[0] for d in many_many(c, 'layout_id')]
    return res


def get_dependencies(conn: SomeConnection,
                     layout_id: int) -> List[List[int]]:
    """
    Get the dependencies of a certain dependent variable (indexed by its
    layout_id)

    Args:
        conn: connection to the database
        layout_id: the layout_id of the dependent variable
    """
    sql = """
    SELECT independent, axis_num FROM dependencies WHERE dependent=?
    """
    c = atomic_transaction(conn, sql, layout_id)
    res = many_many(c, 'independent', 'axis_num')
    return res

# Higher level Wrappers


def new_experiment(conn: SomeConnection,
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
    curr = atomic_transaction(conn, query, name, sample_name,
                              time.time(), format_string, 0)
    return curr.lastrowid


# TODO(WilliamHPNielsen): we should remove the redundant
# is_completed
def mark_run_complete(conn: SomeConnection, run_id: int):
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
    atomic_transaction(conn, query, time.time(), True, run_id)


def completed(conn: SomeConnection, run_id)->bool:
    """ Check if the run scomplete

    Args:
        conn: database connection
        run_id: id of the run to check
    """
    return bool(select_one_where(conn, "runs", "is_completed",
                                 "run_id", run_id))


def get_completed_timestamp_from_run_id(
        conn: SomeConnection, run_id: int) -> float:
    """
    Retrieve the timestamp when the given measurement run was completed

    If the measurement run has not been marked as completed, then the returned
    value is None.

    Args:
        conn: database connection
        run_id: id of the run

    Returns:
        timestamp in seconds since the Epoch, or None
    """
    return select_one_where(conn, "runs", "completed_timestamp",
                            "run_id", run_id)


def get_guid_from_run_id(conn: SomeConnection, run_id: int) -> str:
    """
    Get the guid of the given run

    Args:
        conn: database connection
        run_id: id of the run
    """
    return select_one_where(conn, "runs", "guid", "run_id", run_id)


def finish_experiment(conn: SomeConnection, exp_id: int):
    """ Finish experiment

    Args:
        conn: database connection
        name: the name of the experiment
    """
    query = """
    UPDATE experiments SET end_time=? WHERE exp_id=?;
    """
    atomic_transaction(conn, query, time.time(), exp_id)


def get_run_counter(conn: SomeConnection, exp_id: int) -> int:
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


def get_experiments(conn: SomeConnection) -> List[sqlite3.Row]:
    """ Get a list of experiments
     Args:
         conn: database connection

     Returns:
         list of rows
     """
    sql = """
    SELECT * FROM experiments
    """
    c = atomic_transaction(conn, sql)

    return c.fetchall()


def get_last_experiment(conn: SomeConnection) -> int:
    """
    Return last started experiment id
    """
    query = "SELECT MAX(exp_id) FROM experiments"
    c = atomic_transaction(conn, query)
    return c.fetchall()[0][0]


def get_runs(conn: SomeConnection,
             exp_id: Optional[int] = None)->List[sqlite3.Row]:
    """ Get a list of runs.

    Args:
        conn: database connection

    Returns:
        list of rows
    """
    with atomic(conn) as conn:
        if exp_id:
            sql = """
            SELECT * FROM runs
            where exp_id = ?
            """
            c = transaction(conn, sql, exp_id)
        else:
            sql = """
            SELECT * FROM runs
            """
            c = transaction(conn, sql)

    return c.fetchall()


def get_last_run(conn: SomeConnection, exp_id: int) -> str:
    query = """
    SELECT run_id, max(run_timestamp), exp_id
    FROM runs
    WHERE exp_id = ?;
    """
    c = atomic_transaction(conn, query, exp_id)
    return one(c, 'run_id')


def data_sets(conn: SomeConnection) -> List[sqlite3.Row]:
    """ Get a list of datasets
    Args:
        conn: database connection

    Returns:
        list of rows
    """
    sql = """
    SELECT * FROM runs
    """
    c = atomic_transaction(conn, sql)
    return c.fetchall()


def _insert_run(conn: SomeConnection, exp_id: int, name: str,
                guid: str,
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
    with atomic(conn) as conn:

        if parameters:
            query = f"""
            INSERT INTO {table}
                (name,exp_id,guid,result_table_name,result_counter,run_timestamp,parameters,is_completed)
            VALUES
                (?,?,?,?,?,?,?,?)
            """
            curr = transaction(conn, query,
                               name,
                               exp_id,
                               guid,
                               formatted_name,
                               run_counter,
                               time.time(),
                               ",".join([p.name for p in parameters]),
                               False)

            _add_parameters_to_layout_and_deps(conn, formatted_name,
                                               *parameters)

        else:
            query = f"""
            INSERT INTO {table}
                (name,exp_id,guid,result_table_name,result_counter,run_timestamp,is_completed)
            VALUES
                (?,?,?,?,?,?,?)
            """
            curr = transaction(conn, query,
                               name,
                               exp_id,
                               guid,
                               formatted_name,
                               run_counter,
                               time.time(),
                               False)
    run_id = curr.lastrowid
    return run_counter, formatted_name, run_id


def _update_experiment_run_counter(conn: SomeConnection, exp_id: int,
                                   run_counter: int) -> None:
    query = """
    UPDATE experiments
    SET run_counter = ?
    WHERE exp_id = ?
    """
    atomic_transaction(conn, query, run_counter, exp_id)


def get_parameters(conn: SomeConnection,
                   run_id: int) -> List[ParamSpec]:
    """
    Get the list of param specs for run

    Args:
        conn: the connection to the sqlite database
        run_id: The id of the run

    Returns:
        A list of param specs for this run
    """

    sql = f"""
    SELECT parameter FROM layouts WHERE run_id={run_id}
    """
    c = conn.execute(sql)
    param_names_temp = many_many(c, 'parameter')
    param_names = [p[0] for p in param_names_temp]
    param_names = cast(List[str], param_names)

    parspecs = []

    for param_name in param_names:
        parspecs.append(get_paramspec(conn, run_id, param_name))

    return parspecs


def get_paramspec(conn: SomeConnection,
                  run_id: int,
                  param_name: str) -> ParamSpec:
    """
    Get the ParamSpec object for the given parameter name
    in the given run

    Args:
        conn: Connection to the database
        run_id: The run id
        param_name: The name of the parameter
    """

    # get table name
    sql = f"""
    SELECT result_table_name FROM runs WHERE run_id = {run_id}
    """
    c = conn.execute(sql)
    result_table_name = one(c, 'result_table_name')

    # get the data type
    sql = f"""
    PRAGMA TABLE_INFO("{result_table_name}")
    """
    c = conn.execute(sql)
    for row in c.fetchall():
        if row['name'] == param_name:
            param_type = row['type']
            break

    # get everything else

    sql = f"""
    SELECT * FROM layouts
    WHERE parameter="{param_name}" and run_id={run_id}
    """
    c = conn.execute(sql)
    resp = many(c, 'layout_id', 'run_id', 'parameter', 'label', 'unit',
                'inferred_from')
    (layout_id, _, _, label, unit, inferred_from_string) = resp

    if inferred_from_string:
        inferred_from = inferred_from_string.split(', ')
    else:
        inferred_from = []

    deps = get_dependencies(conn, layout_id)
    depends_on: Optional[List[str]]
    if len(deps) == 0:
        depends_on = None
    else:
        dps: List[int] = [dp[0] for dp in deps]
        ax_nums: List[int] = [dp[1] for dp in deps]
        depends_on = []
        for _, dp in sorted(zip(ax_nums, dps)):
            sql = f"""
            SELECT parameter FROM layouts WHERE layout_id = {dp}
            """
            c = conn.execute(sql)
            depends_on.append(one(c, 'parameter'))

    parspec = ParamSpec(param_name, param_type, label, unit,
                        inferred_from,
                        depends_on)
    return parspec


def add_parameter(conn: SomeConnection,
                  formatted_name: str,
                  *parameter: ParamSpec):
    """ Add parameters to the dataset

    This will update the layouts and dependencies tables

    NOTE: two parameters with the same name are not allowed
    Args:
        - conn: the connection to the sqlite database
        - formatted_name: name of the table
        - parameter: the paraemters to add
    """
    with atomic(conn) as conn:
        p_names = []
        for p in parameter:
            insert_column(conn, formatted_name, p.name, p.type)
            p_names.append(p.name)
        # get old parameters column from run table
        sql = f"""
        SELECT parameters FROM runs
        WHERE result_table_name=?
        """
        with atomic(conn) as conn:
            c = transaction(conn, sql, formatted_name)
        old_parameters = one(c, 'parameters')
        if old_parameters:
            new_parameters = ",".join([old_parameters] + p_names)
        else:
            new_parameters = ",".join(p_names)
        sql = "UPDATE runs SET parameters=? WHERE result_table_name=?"
        with atomic(conn) as conn:
            transaction(conn, sql, new_parameters, formatted_name)

        # Update the layouts table
        c = _add_parameters_to_layout_and_deps(conn, formatted_name,
                                               *parameter)


def _add_parameters_to_layout_and_deps(conn: SomeConnection,
                                       formatted_name: str,
                                       *parameter: ParamSpec) -> sqlite3.Cursor:
    # get the run_id
    sql = f"""
    SELECT run_id FROM runs WHERE result_table_name="{formatted_name}";
    """
    run_id = one(transaction(conn, sql), 'run_id')
    layout_args = []
    for p in parameter:
        layout_args.append(run_id)
        layout_args.append(p.name)
        layout_args.append(p.label)
        layout_args.append(p.unit)
        layout_args.append(p.inferred_from)
    rowplaceholder = '(?, ?, ?, ?, ?)'
    placeholder = ','.join([rowplaceholder] * len(parameter))
    sql = f"""
    INSERT INTO layouts (run_id, parameter, label, unit, inferred_from)
    VALUES {placeholder}
    """

    with atomic(conn) as conn:
        c = transaction(conn, sql, *layout_args)

        for p in parameter:

            if p.depends_on != '':

                layout_id = get_layout_id(conn, p, run_id)

                deps = p.depends_on.split(', ')
                for ax_num, dp in enumerate(deps):

                    sql = """
                    SELECT layout_id FROM layouts
                    WHERE run_id=? and parameter=?;
                    """

                    c = transaction(conn, sql, run_id, dp)
                    dep_ind = one(c, 'layout_id')

                    sql = """
                    INSERT INTO dependencies (dependent, independent, axis_num)
                    VALUES (?,?,?)
                    """

                    c = transaction(conn, sql, layout_id, dep_ind, ax_num)
    return c


def _validate_table_name(table_name: str) -> bool:
    valid = True
    for i in table_name:
        if unicodedata.category(i) not in _unicode_categories:
            valid = False
            raise RuntimeError("Invalid table name "
                               "{} starting at {}".format(table_name, i))
    return valid


def _create_run_table(conn: SomeConnection,
                      formatted_name: str,
                      parameters: Optional[List[ParamSpec]] = None,
                      values: Optional[VALUES] = None
                      ) -> None:
    """Create run table with formatted_name as name

    Args:
        conn: database connection
        formatted_name: the name of the table to create
    """
    _validate_table_name(formatted_name)

    with atomic(conn) as conn:

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


def create_run(conn: SomeConnection, exp_id: int, name: str,
               guid: str,
               parameters: Optional[List[ParamSpec]]=None,
               values:  List[Any] = None,
               metadata: Optional[Dict[str, Any]]=None)->Tuple[int, int, str]:
    """ Create a single run for the experiment.


    This will register the run in the runs table, the counter in the
    experiments table and create a new table with the formatted name.

    Args:
        - conn: the connection to the sqlite database
        - exp_id: the experiment id we want to create the run into
        - name: a friendly name for this run
        - guid: the guid adhering to our internal guid format
        - parameters: optional list of parameters this run has
        - values:  optional list of values for the parameters
        - metadata: optional metadata dictionary

    Returns:
        - run_counter: the id of the newly created run (not unique)
        - run_id: the row id of the newly created run
        - formatted_name: the name of the newly created table
    """

    run_counter, formatted_name, run_id = _insert_run(conn,
                                                      exp_id,
                                                      name,
                                                      guid,
                                                      parameters)
    if metadata:
        add_meta_data(conn, run_id, metadata)
    _update_experiment_run_counter(conn, exp_id, run_counter)
    _create_run_table(conn, formatted_name, parameters, values)

    return run_counter, run_id, formatted_name


def get_metadata(conn: SomeConnection, tag: str, table_name: str):
    """ Get metadata under the tag from table
    """
    return select_one_where(conn, "runs", tag,
                            "result_table_name", table_name)


def insert_meta_data(conn: SomeConnection, row_id: int, table_name: str,
                     metadata: Dict[str, Any]) -> None:
    """
    Insert new metadata column and add values

    Args:
        - conn: the connection to the sqlite database
        - row_id: the row to add the metadata at
        - table_name: the table to add to, defaults to runs
        - metadata: the metadata to add
    """
    for key in metadata.keys():
        insert_column(conn, table_name, key)
    update_meta_data(conn, row_id, table_name, metadata)


def update_meta_data(conn: SomeConnection, row_id: int, table_name: str,
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


def add_meta_data(conn: SomeConnection,
                  row_id: int,
                  metadata: Dict[str, Any],
                  table_name: str = "runs") -> None:
    """
    Add metadata data (updates if exists, create otherwise).

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


def get_user_version(conn: SomeConnection) -> int:

    curr = atomic_transaction(conn, 'PRAGMA user_version')
    res = one(curr, 0)
    return res


def set_user_version(conn: SomeConnection, version: int) -> None:

    atomic_transaction(conn, 'PRAGMA user_version({})'.format(version))


def get_experiment_name_from_experiment_id(
        conn: SomeConnection, exp_id: int) -> str:
    return select_one_where(
        conn, "experiments", "name", "exp_id", exp_id)


def get_sample_name_from_experiment_id(
        conn: SomeConnection, exp_id: int) -> str:
    return select_one_where(
        conn, "experiments", "sample_name", "exp_id", exp_id)


def get_run_timestamp_from_run_id(conn: SomeConnection,
                                  run_id: int) -> float:
    return select_one_where(conn, "runs", "run_timestamp", "run_id", run_id)


def update_GUIDs(conn: SomeConnection) -> None:
    """
    Update all GUIDs in this database where either the location code or the
    work_station code is zero to use the location and work_station code from
    the qcodesrc.json file in home. Runs where it is not true that both codes
    are zero are skipped.
    """

    log.info('Commencing update of all GUIDs in database')

    cfg = qc.Config()

    location = cfg['GUID_components']['location']
    work_station = cfg['GUID_components']['work_station']

    if location == 0:
        log.warning('The location is still set to the default (0). Can not '
                    'proceed. Please configure the location before updating '
                    'the GUIDs.')
        return
    if work_station == 0:
        log.warning('The work_station is still set to the default (0). Can not'
                    ' proceed. Please configure the location before updating '
                    'the GUIDs.')
        return

    query = f"select MAX(run_id) from runs"
    c = atomic_transaction(conn, query)
    no_of_runs = c.fetchall()[0][0]

    # now, there are four actions we can take

    def _both_nonzero(run_id: int, *args) -> None:
        log.info(f'Run number {run_id} already has a valid GUID, skipping.')

    def _location_only_zero(run_id: int, *args) -> None:
        log.warning(f'Run number {run_id} has a zero (default) location '
                    'code, but a non-zero work station code. Please manually '
                    'resolve this, skipping the run now.')

    def _workstation_only_zero(run_id: int, *args) -> None:
        log.warning(f'Run number {run_id} has a zero (default) work station'
                    ' code, but a non-zero location code. Please manually '
                    'resolve this, skipping the run now.')

    def _both_zero(run_id: int, conn, guid_comps) -> None:
        guid_str = generate_guid(timeint=guid_comps['time'],
                                 sampleint=guid_comps['sample'])
        with atomic(conn) as conn:
            sql = f"""
                   UPDATE runs
                   SET guid = ?
                   where run_id == {run_id}
                   """
            cur = conn.cursor()
            cur.execute(sql, (guid_str,))

        log.info(f'Succesfully updated run number {run_id}.')

    actions: Dict[Tuple[bool, bool], Callable]
    actions = {(True, True): _both_zero,
               (False, True): _workstation_only_zero,
               (True, False): _location_only_zero,
               (False, False): _both_nonzero}

    for run_id in range(1, no_of_runs+1):
        guid_str = get_guid_from_run_id(conn, run_id)
        guid_comps = parse_guid(guid_str)
        loc = guid_comps['location']
        ws = guid_comps['work_station']

        log.info(f'Updating run number {run_id}...')
        actions[(loc == 0, ws == 0)](run_id, conn, guid_comps)
